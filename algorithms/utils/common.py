import os
import random
import uuid
from typing import Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import wandb


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def set_seed(seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

    if env is not None:
        if hasattr(env, "seed") and callable(env.seed):
            try:
                env.seed(seed)
            except Exception as e:
                print(f"[set_seed] env.seed({seed}) falhou: {e}")
        if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
            try:
                env.action_space.seed(seed)
            except Exception as e:
                print(f"[set_seed] action_space.seed({seed}) falhou: {e}")


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
        save_code=True,
    )
    wandb.run.save()


def compute_mean_std(states: np.ndarray, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
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
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env
