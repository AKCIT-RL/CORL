# Inspired by:
# 1. paper for SAC-N: https://arxiv.org/abs/2110.01548
# 2. implementation: https://github.com/snu-mllab/EDAC
import math
import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import wandb
from torch.distributions import Normal
from tqdm import trange

from torch.utils.data import DataLoader

import minari

from algorithms.utils.wrapper_gym import get_env
from algorithms.utils.dataset import qlearning_dataset, ReplayBuffer
from algorithms.utils.save_video import save_video

import yaml
import etils.epath as epath

def get_actor_from_checkpoint(checkpoint_path: str, state_dim: int, action_dim: int, max_action: float, checkpoint_id: int = -1):
    ckpt_path = str(epath.Path(checkpoint_path).resolve())
    FINETUNE_PATH = epath.Path(ckpt_path)
    latest_ckpts = list(FINETUNE_PATH.glob("*"))
    latest_ckpts = [ckpt for ckpt in latest_ckpts if not ckpt.is_dir() and ckpt.name.endswith(".pt")]
    latest_ckpts.sort(key=lambda x: int(x.name.split("/")[-1].split("_")[-1].split(".")[0]))
    latest_ckpts = latest_ckpts[checkpoint_id]

    with open(os.path.join(checkpoint_path, "config.yaml")) as f:
        config = yaml.safe_load(f)

    actor = Actor(state_dim, action_dim, config["hidden_dim"], config["max_action"])
    actor.to(config["device"])
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config["actor_learning_rate"])
    critic = VectorizedCritic(
        state_dim, action_dim, config["hidden_dim"], config["num_critics"]
    )
    critic.to(config["device"])
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config["critic_learning_rate"]
    )

    trainer = SACN(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic=critic,
        critic_optimizer=critic_optimizer,
        gamma=config["gamma"],
        tau=config["tau"],
        alpha_learning_rate=config["alpha_learning_rate"],
        device=config["device"],
    )
    trainer.load_state_dict(torch.load(latest_ckpts))
    actor = trainer.actor
    actor.eval()
    return actor
   
    return actor


@dataclass
class TrainConfig:
    # wandb project name
    project: str = "CORL"
    # wandb group name
    group: str = "SAC-N"
    # wandb run name
    name: str = "SAC-N"
    # actor and critic hidden dim
    hidden_dim: int = 256
    # critic ensemble size
    num_critics: int = 10
    # discount factor
    gamma: float = 0.99
    # coefficient for the target critic Polyak's update
    tau: float = 5e-3
    # actor learning rate
    actor_learning_rate: float = 3e-4
    # critic learning rate
    critic_learning_rate: float = 3e-4
    # entropy coefficient learning rate for automatic tuning
    alpha_learning_rate: float = 3e-4
    # maximum range for the symmetric actions, [-1, 1]
    max_action: float = 1.0
    # maximum size of the replay buffer
    buffer_size: int = 1_000_000
    # training dataset and evaluation environment
    env: str = "halfcheetah-medium-expert-v2" 
    dataset_id: str = "halfcheetah-medium-expert-v2"
    # training batch size
    batch_size: int = 256
    # total number of training epochs
    num_epochs: int = 3000
    # number of gradient updates during one epoch
    num_updates_on_epoch: int = 1000
    # whether to normalize reward (like in IQL)
    normalize_reward: bool = False
    # number of episodes to run during evaluation
    eval_episodes: int = 10
    # evaluation frequency, will evaluate eval_every training steps
    eval_every: int = 5
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = None
    # configure PyTorch to use deterministic algorithms instead
    # of nondeterministic ones
    deterministic_torch: bool = False
    # training random seed
    train_seed: int = 10
    # evaluation random seed
    eval_seed: int = 42
    # frequency of metrics logging to the wandb
    log_every: int = 100
    # training device
    device: str = "cuda"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# general utils
TensorBatch = List[torch.Tensor]


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


# SAC Actor & Critic implementation
class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias


class Actor(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, max_action: float = 1.0
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # with separate layers works better than with Linear(hidden_dim, 2 * action_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_sigma = nn.Linear(hidden_dim, action_dim)

        # init as in the EDAC paper
        for layer in self.trunk[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.bias, -1e-3, 1e-3)

        self.action_dim = action_dim
        self.max_action = max_action

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden = self.trunk(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)

        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_sigma = torch.clip(log_sigma, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_sigma))

        if deterministic:
            action = mu
        else:
            action = policy_dist.rsample()

        tanh_action, log_prob = torch.tanh(action), None
        if need_log_prob:
            # change of variables formula (SAC paper, appendix C, eq 21)
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)

        return tanh_action * self.max_action, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        deterministic = not self.training
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action = self(state, deterministic=deterministic)[0].cpu().numpy()
        return action


class VectorizedCritic(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, num_critics: int
    ):
        super().__init__()
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, num_critics),
        )
        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [batch_size, state_dim + action_dim]
        state_action = torch.cat([state, action], dim=-1)
        # [num_critics, batch_size, state_dim + action_dim]
        state_action = state_action.unsqueeze(0).repeat_interleave(
            self.num_critics, dim=0
        )
        # [num_critics, batch_size]
        q_values = self.critic(state_action).squeeze(-1)
        return q_values


class SACN:
    def __init__(
        self,
        actor: Actor,
        actor_optimizer: torch.optim.Optimizer,
        critic: VectorizedCritic,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_learning_rate: float = 1e-4,
        device: str = "cpu",
    ):
        self.device = device

        self.actor = actor
        self.critic = critic
        with torch.no_grad():
            self.target_critic = deepcopy(self.critic)

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.tau = tau
        self.gamma = gamma

        # adaptive alpha setup
        self.target_entropy = -float(self.actor.action_dim)
        self.log_alpha = torch.tensor(
            [0.0], dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate)
        self.alpha = self.log_alpha.exp().detach()

    def _alpha_loss(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, action_log_prob = self.actor(state, need_log_prob=True)

        loss = (-self.log_alpha * (action_log_prob + self.target_entropy)).mean()

        return loss

    def _actor_loss(self, state: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        action, action_log_prob = self.actor(state, need_log_prob=True)
        q_value_dist = self.critic(state, action)
        assert q_value_dist.shape[0] == self.critic.num_critics
        q_value_min = q_value_dist.min(0).values
        # needed for logging
        q_value_std = q_value_dist.std(0).mean().item()
        batch_entropy = -action_log_prob.mean().item()

        assert action_log_prob.shape == q_value_min.shape
        loss = (self.alpha * action_log_prob - q_value_min).mean()

        return loss, batch_entropy, q_value_std

    def _critic_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(
                next_state, need_log_prob=True
            )
            q_next = self.target_critic(next_state, next_action).min(0).values
            q_next = q_next - self.alpha * next_action_log_prob

            assert q_next.unsqueeze(-1).shape == done.shape == reward.shape
            q_target = reward + self.gamma * (1 - done) * q_next.unsqueeze(-1)

        q_values = self.critic(state, action)
        # [ensemble_size, batch_size] - [1, batch_size]
        loss = ((q_values - q_target.view(1, -1)) ** 2).mean(dim=1).sum(dim=0)

        return loss

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        state, action, reward, next_state, done = [arr.to(self.device) for arr in batch]
        # Usually updates are done in the following order: critic -> actor -> alpha
        # But we found that EDAC paper uses reverse (which gives better results)

        # Alpha update
        alpha_loss = self._alpha_loss(state)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()

        # Actor update
        actor_loss, actor_batch_entropy, q_policy_std = self._actor_loss(state)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic update
        critic_loss = self._critic_loss(state, action, reward, next_state, done)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #  Target networks soft update
        with torch.no_grad():
            soft_update(self.target_critic, self.critic, tau=self.tau)
            # for logging, Q-ensemble std estimate with the random actions:
            # a ~ U[-max_action, max_action]
            max_action = self.actor.max_action
            random_actions = -max_action + 2 * max_action * torch.rand_like(action)

            q_random_std = self.critic(state, random_actions).std(0).mean().item()

        update_info = {
            "alpha_loss": alpha_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "batch_entropy": actor_batch_entropy,
            "alpha": self.alpha.item(),
            "q_policy_std": q_policy_std,
            "q_random_std": q_random_std,
        }
        return update_info

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic_optim": self.critic_optimizer.state_dict(),
            "alpha_optim": self.alpha_optimizer.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optim"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optim"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optim"])
        self.log_alpha.data[0] = state_dict["log_alpha"]
        self.alpha = self.log_alpha.exp().detach()


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
            action = actor.act(state, device)[0]
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


# normalization like in the IQL paper
# https://github.com/ikostrikov/implicit_q_learning/blob/09d700248117881a75cb21f0adb95c6c8a694cb2/train_offline.py#L35 # noqa
def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    wandb_init(asdict(config))

    # data, evaluation, env setup
    dataset = minari.load_dataset(config.dataset_id)
    qdataset = qlearning_dataset(dataset)        
    env = get_env(config.env, config.device)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if config.normalize_reward:
        modify_reward(qdataset, config.env)

    replay_buffer = ReplayBuffer(qdataset)
    replay_buffer = DataLoader(
        replay_buffer, batch_size=config.batch_size, shuffle=True
    )

    # Actor & Critic setup
    actor = Actor(state_dim, action_dim, config.hidden_dim, config.max_action)
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    critic = VectorizedCritic(
        state_dim, action_dim, config.hidden_dim, config.num_critics
    )
    critic.to(config.device)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config.critic_learning_rate
    )

    trainer = SACN(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic=critic,
        critic_optimizer=critic_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        alpha_learning_rate=config.alpha_learning_rate,
        device=config.device,
    )
    # saving config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    total_updates = 0.0
    for epoch in trange(config.num_epochs, desc="Training"):
        # training
        for _ in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            batch = next(iter(replay_buffer))
            update_info = trainer.update(batch)

            if total_updates % config.log_every == 0:
                wandb.log({"epoch": epoch, **update_info})

            total_updates += 1

        # evaluation
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            eval_returns = eval_actor(
                env=env,
                actor=actor,
                n_episodes=config.eval_episodes,
                seed=config.eval_seed,
                device=config.device,
            )
            eval_log = {
                "eval/score": np.mean(eval_returns),
                "eval/reward_std": np.std(eval_returns),
                "epoch": epoch,
            }
            if hasattr(env, "get_normalized_score"):
                normalized_score = env.get_normalized_score(eval_returns) * 100.0
                eval_log["eval/normalized_score_mean"] = np.mean(normalized_score)
                eval_log["eval/normalized_score_std"] = np.std(normalized_score)

            wandb.log(eval_log)

            if config.checkpoints_path is not None:
                # Save checkpoint every 10 epochs
                if epoch % 500 == 0 or epoch == config.num_epochs - 1:
                    torch.save(
                        trainer.state_dict(),
                        os.path.join(config.checkpoints_path, f"checkpoint_{epoch}.pt"),
                    )

    # save_video(
    #     env_name=config.env,
    #     actor=trainer.actor,
    #     device=config.device,
    #     command=[1.0, 0.0, 0.0],
    #     path_model=config.checkpoints_path,
    # )
    wandb.finish()


if __name__ == "__main__":
    train()
