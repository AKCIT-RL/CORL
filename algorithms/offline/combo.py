# source: Implementation based on https://arxiv.org/pdf/2102.08363.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Tuple as TTuple

import gymnasium as gym 
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
import etils.epath as epath
from torch.utils.data import DataLoader
from tqdm import tqdm

import minari

from algorithms.utils.wrapper_gym import get_env
from algorithms.utils.dataset import qlearning_dataset, ReplayBuffer
from algorithms.utils.save_video import save_video
TensorBatch = List[torch.Tensor]

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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
        "device": config["device"],
    }

    trainer = COMBO(**kwargs)
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
    group: str = "COMBO-D4RL"
    # wandb run name
    name: str = "COMBO"
    # training dataset and evaluation environment
    env: str = "halfcheetah-medium-expert-v2"  # OpenAI gym environment name
    dataset_id: str = "halfcheetah-medium-expert-v2"
    # conservative coefficient
    beta: float = 0.5
    # mix ratio f for d_f
    f: float = 0.5
    # discount factor
    discount: float = 0.99
    # standard deviation for the gaussian exploration noise
    expl_noise: float = 0.1
    # coefficient for the target critic Polyak's update
    tau: float = 0.005
    # scalig coefficient for the noise added to
    # target actor during critic update
    policy_noise: float = 0.2
    # range for the target actor noise clipping
    noise_clip: float = 0.5
    # actor update delay
    policy_freq: int = 2
    # total gradient updates during training
    max_timesteps: int = int(1e6)
    # maximum size of the replay buffer
    buffer_size: int = 2_000_000
    # model buffer size
    model_buffer_size: int = 100_000
    # training batch size
    batch_size: int = 256
    # whether to normalize states
    normalize: bool = True
    # whether to normalize reward (like in IQL)
    normalize_reward: bool = False
    # evaluation frequency, will evaluate every eval_freq training steps
    eval_freq: int = int(5e3)
    # number of episodes to run during evaluation
    n_episodes: int = 10
    # model generation frequency
    gen_freq: int = 1000
    # model rollout horizon
    model_horizon: int = 5
    # number of trajectories to generate per update
    model_num_trajectories: int = 1000
    # dynamics training epochs
    dynamics_epochs: int = 100
    # dynamics learning rate
    model_lr: float = 3e-4
    # number of ensemble models
    ensemble_size: int = 5
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
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state, env.observation_space)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

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
        settings=wandb.Settings(save_code=True), # alterei
    )
    #wandb.run.save()
    # Se quiser subir checkpoints à medida que são criados:
    ckpt_dir = config.get("checkpoints_path")
    if ckpt_dir:
        wandb.save(os.path.join(ckpt_dir, "checkpoint_*.pt"), policy="live")  # glob obrigatório


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    actor.eval()
    episode_rewards = []
    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep) # Seed per episode for reproducibility
        done = False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


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


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        return self.net(sa)


class Dynamics(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Dynamics, self).__init__()
        input_dim = state_dim + action_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_net = nn.Linear(hidden_dim, state_dim)
        self.log_std_net = nn.Linear(hidden_dim, state_dim)
        self.reward_net = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> TTuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        mean = self.mean_net(x)
        log_std = torch.clamp(self.log_std_net(x), -20, 2)
        reward = self.reward_net(x)
        return mean, log_std, reward


def train_dynamics(dynamics: Dynamics, optimizer: torch.optim.Optimizer, dataset: dict, device: str, epochs: int, batch_size: int):
    dynamics.train()
    buffer = ReplayBuffer(dataset)
    loader = DataLoader(buffer, batch_size=batch_size, shuffle=True)
    print('\n\nPROGRESS - DYNAMICS TRAINING:\n\n')
    for _ in tqdm(range(epochs)):
        for batch in loader:
            state, action, reward, next_state, _ = [b.to(device) for b in batch]
            mu, log_std, pred_reward = dynamics(state, action)
            std = torch.exp(0.5 * log_std)
            dist = torch.distributions.Normal(mu, std)
            nll = -dist.log_prob(next_state).sum(dim=-1).mean()
            r_loss = F.mse_loss(pred_reward, reward.unsqueeze(-1))
            loss = nll + r_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def sample_batch(buffer: ReplayBuffer, batch_size: int, device: str) -> TensorBatch:
    indices = np.random.choice(len(buffer), batch_size, replace=True)
    batch = [buffer[idx] for idx in indices]
    state = torch.stack([item[0] for item in batch]).to(device)
    action = torch.stack([item[1] for item in batch]).to(device)
    reward = torch.stack([item[2] for item in batch]).to(device)
    next_state = torch.stack([item[3] for item in batch]).to(device)
    done = torch.stack([item[4] for item in batch]).to(device)
    return [state, action, reward, next_state, done]


def generate_model_data(dynamics_ensemble: List[Dynamics], actor: Actor, offline_buffer: ReplayBuffer, num_trajectories: int, horizon: int, device: str, max_action: float) -> TTuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dynamics_ensemble = [dyn.eval() for dyn in dynamics_ensemble]
    actor.eval()
    with torch.no_grad():
        indices = np.random.choice(len(offline_buffer), num_trajectories, replace=False)
        start_states = []
        for idx in indices:
            obs, _, _, _, _ = offline_buffer[idx]
            start_states.append(obs)
        start_states = torch.tensor(np.array(start_states), device=device, dtype=torch.float32)
        obs_list, act_list, rew_list, next_obs_list, done_list = [], [], [], [], []
        for i in range(num_trajectories):
            s = start_states[i].unsqueeze(0)
            for step in range(horizon):
                a = actor(s).clamp(-max_action, max_action)
                # Average predictions from ensemble
                mu_list = []
                log_std_list = []
                r_list = []
                for dyn in dynamics_ensemble:
                    mu, log_std, r = dyn(s, a)
                    mu_list.append(mu)
                    log_std_list.append(log_std)
                    r_list.append(r)
                mu = torch.mean(torch.stack(mu_list), dim=0)
                log_std = torch.mean(torch.stack(log_std_list), dim=0)
                r = torch.mean(torch.stack(r_list), dim=0)
                std = torch.exp(0.5 * log_std)
                z = torch.randn_like(mu)
                next_s = mu + std * z
                obs_list.append(s.squeeze(0).cpu().numpy())
                act_list.append(a.squeeze(0).cpu().numpy())
                rew_list.append(r.squeeze(0).cpu().numpy())
                next_obs_list.append(next_s.squeeze(0).cpu().numpy())
                done_list.append(0.0)
                s = next_s
        return (np.array(obs_list), np.array(act_list), np.array(rew_list).squeeze(), np.array(next_obs_list), np.array(done_list))


class COMBO:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        beta: float = 0.5,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.actor_optimizer = actor_optimizer
        self.critic_1 = critic_1
        self.critic_1_target = copy.deepcopy(critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2 = critic_2
        self.critic_2_target = copy.deepcopy(critic_2)
        self.critic_2_optimizer = critic_2_optimizer

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.beta = beta

        self.total_it = 0
        self.device = device

    def train(
        self,
        off_state: torch.Tensor,
        off_action: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        has_model: bool = False,
        model_state: Optional[torch.Tensor] = None,
        model_action: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        not_done = 1 - done

        # Conservative term
        q1_off = self.critic_1(off_state, off_action)
        q2_off = self.critic_2(off_state, off_action)
        q_off = torch.min(q1_off, q2_off).mean()
        cons_loss = torch.tensor(0.0, device=self.device)
        if has_model:
            q1_model = self.critic_1(model_state, model_action)
            q2_model = self.critic_2(model_state, model_action)
            q_model = torch.min(q1_model, q2_model).mean()
            cons_loss = self.beta * (q_model - q_off)
        log_dict["cons_loss"] = cons_loss.item()

        with torch.no_grad():
            # Select action according to target actor and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q

        # Get current Q estimates
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        # Compute critic loss
        bellman_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )
        critic_loss = cons_loss + bellman_loss
        log_dict["critic_loss"] = critic_loss.item()
        # Optimize the critics
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            pi = self.actor(state)
            q = self.critic_1(state, pi)
            actor_loss = -q.mean()
            log_dict["actor_loss"] = actor_loss.item()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "critic_1": self.critic_1.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.critic_1.load_state_dict(state_dict["critic_1"])
        self.critic_1_optimizer.load_state_dict(state_dict["critic_1_optimizer"])
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2.load_state_dict(state_dict["critic_2"])
        self.critic_2_optimizer.load_state_dict(state_dict["critic_2_optimizer"])
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)

        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    dataset = minari.load_dataset(config.dataset_id)
    qdataset = qlearning_dataset(dataset)

    env = get_env(config.env, config.device)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if config.normalize_reward:
        modify_reward(qdataset, config.env)

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

    offline_buffer = ReplayBuffer(qdataset)

    max_action = 1.0

    # Train dynamics ensemble
    dynamics_ensemble = [Dynamics(state_dim, action_dim).to(config.device) for _ in range(config.ensemble_size)]
    print("\n\n-------------------------------------------")
    print(f"Training dynamics ensemble of {config.ensemble_size} models (warm start)...")
    print("-------------------------------------------\n\n")
    for i, dynamics in enumerate(dynamics_ensemble):
        print(f"Training ensemble member {i+1}/{config.ensemble_size}")
        dynamics_optimizer = torch.optim.Adam(dynamics.parameters(), lr=config.model_lr)
        train_dynamics(dynamics, dynamics_optimizer, qdataset, config.device, config.dynamics_epochs, config.batch_size)
    print("Done training dynamics ensemble.")

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    actor = Actor(state_dim, action_dim, max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    critic_1 = Critic(state_dim, action_dim).to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=3e-4)
    critic_2 = Critic(state_dim, action_dim).to(config.device)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=3e-4)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "critic_1": critic_1,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2": critic_2,
        "critic_2_optimizer": critic_2_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # TD3
        "policy_noise": config.policy_noise * max_action,
        "noise_clip": config.noise_clip * max_action,
        "policy_freq": config.policy_freq,
        # COMBO
        "beta": config.beta,
    }

    print("---------------------------------------")
    print(f"Training COMBO, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize trainer
    trainer = COMBO(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    # Initialize model data
    model_obs = np.empty((0, state_dim))
    model_acts = np.empty((0, action_dim))
    model_rews = np.empty((0,))
    model_next_obs = np.empty((0, state_dim))
    model_dones = np.empty((0,))
    model_buffer = None
    has_model = False

    evaluations = []
    for t in range(int(config.max_timesteps)):
        # Sample offline batch for conservatism
        off_batch = sample_batch(offline_buffer, config.batch_size, config.device)
        off_state, off_action, _, _, _ = off_batch

        # Sample mixed batch for Bellman and actor
        num_off = int(config.batch_size * config.f)
        num_model = config.batch_size - num_off
        if not has_model or (model_buffer is not None and len(model_buffer) < num_model):
            num_model = 0
            num_off = config.batch_size
        state, action, reward, next_state, done = sample_batch(offline_buffer, num_off, config.device)
        model_state = None
        model_action = None
        if num_model > 0:
            model_bell = sample_batch(model_buffer, num_model, config.device)
            state = torch.cat([state, model_bell[0]], dim=0)
            action = torch.cat([action, model_bell[1]], dim=0)
            reward = torch.cat([reward, model_bell[2]], dim=0)
            next_state = torch.cat([next_state, model_bell[3]], dim=0)
            done = torch.cat([done, model_bell[4]], dim=0)
            model_state = model_bell[0]
            model_action = model_bell[1]

        log_dict = trainer.train(
            off_state, off_action, state, action, reward, next_state, done,
            has_model=has_model, model_state=model_state, model_action=model_action
        )
        wandb.log(log_dict, step=trainer.total_it)

        # Generate model data periodically
        if (t + 1) % config.gen_freq == 0:
            new_obs, new_acts, new_rews, new_next_obs, new_dones = generate_model_data(
                dynamics_ensemble, actor, offline_buffer, config.model_num_trajectories,
                config.model_horizon, config.device, max_action
            )
            if len(model_obs) > 0:
                model_obs = np.concatenate([model_obs, new_obs], axis=0)
                model_acts = np.concatenate([model_acts, new_acts], axis=0)
                model_rews = np.concatenate([model_rews, new_rews], axis=0)
                model_next_obs = np.concatenate([model_next_obs, new_next_obs], axis=0)
                model_dones = np.concatenate([model_dones, new_dones], axis=0)
            else:
                model_obs = new_obs
                model_acts = new_acts
                model_rews = new_rews
                model_next_obs = new_next_obs
                model_dones = new_dones
            if len(model_obs) > config.model_buffer_size:
                model_obs = model_obs[-config.model_buffer_size:]
                model_acts = model_acts[-config.model_buffer_size:]
                model_rews = model_rews[-config.model_buffer_size:]
                model_next_obs = model_next_obs[-config.model_buffer_size:]
                model_dones = model_dones[-config.model_buffer_size:]
            model_dataset = {
                "observations": model_obs,
                "actions": model_acts,
                "rewards": model_rews,
                "next_observations": model_next_obs,
                "terminals": model_dones,
            }
            model_buffer = ReplayBuffer(model_dataset)
            has_model = True

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
            print(f"Evaluation over {config.n_episodes} episodes: " f"{eval_score:.3f}")
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


    # save_video(
    #     env_name=config.env,
    #     actor=trainer.actor,
    #     device=config.device,
    #     command=[1.0, 0.0, 0.0],
    #     path_model=config.checkpoints_path,
    # )

if __name__ == "__main__":
    train()