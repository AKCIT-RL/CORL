# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

import minari
from algorithms.utils.wrapper_gym import get_env
from algorithms.utils.dataset import qlearning_dataset, ReplayBuffer
from torch.utils.data import TensorDataset, DataLoader

from algorithms.utils.common import soft_update, set_seed, wandb_init, compute_mean_std, normalize_states, wrap_env

TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    # wandb project name
    project: str = "CORL"
    # wandb group name
    group: str = "IQL-Minari"
    # wandb run name
    name: str = "IQL"
    # environment identifier for get_env
    env: str = "halfcheetah-medium-expert-v2"
    # dataset id for Minari
    dataset_id: str = "halfcheetah-medium-expert-v2"
    # discount factor
    discount: float = 0.99
    # coefficient for the target critic Polyak's update
    tau: float = 0.005
    # actor update inverse temperature, similar to AWAC
    # small beta -> BC, big beta -> maximizing Q-value
    beta: float = 3.0
    # coefficient for asymmetric critic loss
    iql_tau: float = 0.7
    # whether to use deterministic actor
    iql_deterministic: bool = False
    # total gradient updates during training
    max_timesteps: int = int(1e6)
    # maximum size of the replay buffer
    buffer_size: int = 2_000_000
    # training batch size
    batch_size: int = 256
    # whether to normalize states
    normalize: bool = True
    # whether to normalize reward (like in IQL)
    normalize_reward: bool = False
    # V-critic function learning rate
    vf_lr: float = 3e-4
    # Q-critic learning rate
    qf_lr: float = 3e-4
    # actor learning rate
    actor_lr: float = 3e-4
    #  where to use dropout for policy network, optional
    actor_dropout: Optional[float] = None
    # evaluation frequency, will evaluate every eval_freq training steps
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



class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)

class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



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
def eval_actor(env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int) -> np.ndarray:
    # 1) Semear o ambiente (se suportado)
    try:
        env.seed(seed)
    except Exception:
        pass
    if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
        try:
            env.action_space.seed(seed)
        except Exception:
            pass

    actor.eval()
    episode_rewards = []

    for _ in range(n_episodes):
        # 2) RESET: pode vir como (obs, info) ou dicionário. Pegamos apenas a parte "obs".
        raw = env.reset()
        if isinstance(raw, tuple):
            state = raw[0]
        else:
            state = raw

        episode_reward = 0.0
        done = False

        while not done:
            # 3) Obter ação a partir de state (que agora é de fato um array)
            state_np = state if isinstance(state, np.ndarray) else np.array(state)
            action = actor.act(state_np, device)

            # 4) STEP: pode retornar 4‐tuple (obs, reward, done, info)
            #    ou 5‐tuple (obs, reward, terminated, truncated, info).
            out = env.step(action)
            if len(out) == 4:
                next_raw, reward, done_flag, info = out
                done = bool(done_flag)
            elif len(out) == 5:
                next_raw, reward, term_flag, trunc_flag, info = out
                done = bool(term_flag or trunc_flag)
            else:
                raise RuntimeError(
                    f"env.step() retornou {len(out)} elementos, mas esperava 4 ou 5"
                )

            # 5) Extrair a parte “obs” de next_raw
            if isinstance(next_raw, tuple):
                next_state = next_raw[0]
            else:
                next_state = next_raw

            episode_reward += float(reward)
            state = next_state

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


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    # 1) Carregar dataset Minari e converter para qdataset
    dataset_raw = minari.load_dataset(config.dataset_id)
    qdataset = qlearning_dataset(dataset_raw)

    # 2) Criar o ambiente via get_env e ajustar seed
    env = get_env(config.env, config.device)
    set_seed(config.seed, env)

    # 3) Normalizar recompensa se necessário
    if config.normalize_reward:
        modify_reward(qdataset, config.env)

    # 4) Calcular mean/std e normalizar observações
    if config.normalize:
        state_mean, state_std = compute_mean_std(qdataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    qdataset["observations"] = normalize_states(qdataset["observations"], state_mean, state_std)
    qdataset["next_observations"] = normalize_states(qdataset["next_observations"], state_mean, state_std)

    # 5) Aplicar wrapper de normalização no ambiente
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    # 6) Agora que “env” existe, defina state_dim e action_dim
    #    Note que, em ambientes Minari, pode haver espaços compostos, mas
    #    aqui assumimos que `.observation_space.shape` e `.action_space.shape` existem:
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 7) Montar DataLoader a partir de qdataset (sem usar replay_buffer.sample)
    from torch.utils.data import TensorDataset, DataLoader

    obs_tensor       = torch.tensor(qdataset["observations"],       dtype=torch.float32)
    actions_tensor   = torch.tensor(qdataset["actions"],            dtype=torch.float32)
    rewards_tensor   = torch.tensor(qdataset["rewards"].reshape(-1, 1),    dtype=torch.float32)
    next_obs_tensor  = torch.tensor(qdataset["next_observations"], dtype=torch.float32)
    dones_tensor     = torch.tensor(qdataset["terminals"].reshape(-1, 1),   dtype=torch.float32)

    full_dataset = TensorDataset(obs_tensor, actions_tensor, rewards_tensor, next_obs_tensor, dones_tensor)
    dataloader = DataLoader(full_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    # 8) Determinar max_action (com fallback se action_space.high não existir)
    try:
        max_action = float(env.action_space.high[0])
    except Exception:
        max_action = 1.0

    # 9) Criar redes e otimizadores (state_dim/action_dim usados aqui)
    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(state_dim, action_dim, max_action, dropout=config.actor_dropout)
        if config.iql_deterministic
        else GaussianPolicy(state_dim, action_dim, max_action, dropout=config.actor_dropout)
    ).to(config.device)

    v_optimizer     = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer     = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    # 10) Montar kwargs para o construtor ImplicitQLearning
    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
    }

    # 11) Instanciar o trainer
    trainer = ImplicitQLearning(**kwargs)
    if config.load_model:
        trainer.load_state_dict(torch.load(config.load_model))
        actor = trainer.actor

    # 12) Inicializar wandb e iniciar o loop de treino
    wandb_init(asdict(config))

    step = 0
    while step < config.max_timesteps:
        for batch_tensors in dataloader:
            obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = [
                t.to(config.device) for t in batch_tensors
            ]

            log_dict = trainer.train([obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch])
            wandb.log(log_dict, step=trainer.total_it)

            step += 1
            if step % config.eval_freq == 0:
                # Avaliação
                eval_scores = eval_actor(env, actor, device=config.device, n_episodes=config.n_episodes, seed=config.seed)
                eval_score = eval_scores.mean()
                if hasattr(env, "get_normalized_score"):
                    norm_score = env.get_normalized_score(eval_score) * 100.0
                    print(f"Evaluation over {config.n_episodes} episodes: {eval_score:.3f} , Normalized: {norm_score:.3f}")
                    wandb.log({"d4rl_normalized_score": norm_score}, step=trainer.total_it)
                else:
                    print(f"Evaluation over {config.n_episodes} episodes: {eval_score:.3f}")

                if config.checkpoints_path is not None:
                    torch.save(trainer.state_dict(), os.path.join(config.checkpoints_path, f"checkpoint_{step}.pt"))

                if step >= config.max_timesteps:
                    break
        if step >= config.max_timesteps:
            break

    wandb.finish()



if __name__ == "__main__":
    train()
