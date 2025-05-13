import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import etils.epath as epath
import yaml
from torch.utils.data import DataLoader

import minari

from algorithms.utils.wrapper_gym import get_env
from algorithms.utils.dataset import qlearning_dataset

TensorBatch = List[torch.Tensor]

def get_actor_from_checkpoint(checkpoint_path: str, state_dim: int, action_dim: int, max_action: float, checkpoint_id: int = -1):
    ckpt_path = str(epath.Path(checkpoint_path).resolve())
    FINETUNE_PATH = epath.Path(ckpt_path)
    latest_ckpts = list(FINETUNE_PATH.glob("*"))
    latest_ckpts = [ckpt for ckpt in latest_ckpts if not ckpt.is_dir() and ckpt.name.endswith(".pt")]
    latest_ckpts.sort(key=lambda x: int(x.name.split("/")[-1].split("_")[-1].split(".")[0]))
    latest_ckpts = latest_ckpts[checkpoint_id]

    with open(os.path.join(checkpoint_path, "config.yaml")) as f:
        config = yaml.safe_load(f)

    actor = Actor(state_dim, action_dim, max_action).to(config["device"])
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config["discount"],
        "device": config["device"],
    }

    trainer = BC(**kwargs)
    actor = Actor(state_dim, action_dim, max_action).to(config["device"])

    policy_file = Path(latest_ckpts)
    trainer.load_state_dict(torch.load(policy_file))
    actor = trainer.actor
    actor.eval()
    return actor


@dataclass
class TrainConfig:
    # wandb project name
    project: str = "CORL"
    # wandb group name
    group: str = "BC-D4RL"
    # wandb run name
    name: str = "BC"
    # training dataset and evaluation environment
    env: str = "halfcheetah-medium-expert-v2"  # OpenAI gym environment name
    dataset_id: str = "halfcheetah-medium-expert-v2"
    # total gradient updates during training
    max_timesteps: int = int(1e6)
    # training batch size
    batch_size: int = 256
    # maximum size of the replay buffer
    buffer_size: int = 2_000_000
    # what top fraction of the dataset (sorted by return) to use
    frac: float = 0.1
    # maximum possible trajectory length
    max_traj_len: int = 1000
    # whether to normalize states
    normalize: bool = True
    # discount factor
    discount: float = 0.99
    # evaluation frequency, will evaluate eval_freq training steps
    eval_freq: int = int(5e3)
    # number of episodes to run during evaluation
    n_episodes: int = 10
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = None
    # file name for loading a model, optional
    load_model: str = ""
    # training random seed
    seed: int = 0
    # training device
    device: str = "cuda"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

class ReplayBuffer(torch.utils.data.Dataset):
    def __init__(self, dataset_dict):
        self.observations = dataset_dict["observations"]
        self.actions = dataset_dict["actions"]
        self.rewards = dataset_dict["rewards"]
        self.next_observations = dataset_dict["next_observations"]
        self.terminals = dataset_dict["terminals"]
        self.size = len(self.observations)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return [
            torch.from_numpy(self.observations[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.tensor(self.rewards[idx], dtype=torch.float32),
            torch.from_numpy(self.next_observations[idx]),
            torch.tensor(self.terminals[idx], dtype=torch.float32),
        ]


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def keep_best_trajectories(
    dataset: Dict[str, np.ndarray],
    frac: float,
    discount: float,
    max_episode_steps: int = 1000,
):
    ids_by_trajectories = []
    returns = []
    cur_ids = []
    cur_return = 0
    reward_scale = 1.0
    for i, (reward, done) in enumerate(zip(dataset["rewards"], dataset["terminals"])):
        cur_return += reward_scale * reward
        cur_ids.append(i)
        reward_scale *= discount
        if done == 1.0 or len(cur_ids) == max_episode_steps:
            ids_by_trajectories.append(list(cur_ids))
            returns.append(cur_return)
            cur_ids = []
            cur_return = 0
            reward_scale = 1.0

    sort_ord = np.argsort(returns, axis=0)[::-1].reshape(-1)
    top_trajs = sort_ord[: max(1, int(frac * len(sort_ord)))]

    order = []
    for i in top_trajs:
        order += ids_by_trajectories[i]
    order = np.array(order)
    dataset["observations"] = dataset["observations"][order]
    dataset["actions"] = dataset["actions"][order]
    dataset["next_observations"] = dataset["next_observations"][order]
    dataset["rewards"] = dataset["rewards"][order]
    dataset["terminals"] = dataset["terminals"][order]


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class BC:
    def __init__(
        self,
        max_action: np.ndarray,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.max_action = max_action
        self.discount = discount

        self.total_it = 0
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, _, _, _ = batch

        # Compute actor loss
        pi = self.actor(state)
        actor_loss = F.mse_loss(pi, action)
        log_dict["actor_loss"] = actor_loss.item()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    dataset = minari.load_dataset(config.dataset_id)
    qdataset = qlearning_dataset(dataset)        
    env = get_env(config.env, config.device)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    keep_best_trajectories(qdataset, config.frac, config.discount)

    if config.normalize:
        state_mean, state_std = compute_mean_std(qdataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    qdataset["observations"] = normalize_states(
        qdataset["observations"], state_mean, state_std
    )
    qdataset["next_observations"] = normalize_states(
        qdataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(qdataset)
    replay_buffer = DataLoader(
        replay_buffer, batch_size=config.batch_size, shuffle=True
    )

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    max_action = 1.0

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    actor = Actor(state_dim, action_dim, max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "device": config.device,
    }

    print("---------------------------------------")
    print(f"Training BC, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize policy
    trainer = BC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = next(iter(replay_buffer))
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()                
            evaluations.append(eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f}"
            )
            print("---------------------------------------")

            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )

            wandb.log(
                {"eval/score": eval_score},
                step=trainer.total_it,
            )


if __name__ == "__main__":
    train()
