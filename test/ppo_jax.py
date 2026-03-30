from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import pyrallis

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params

from algorithms.utils.wrapper_gym import GymWrapper, get_env

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"


@dataclass
class PPOTestConfig:
   env: str = "Go2JoystickFlatTerrain"
   load_checkpoint_path: Optional[str] = None
   command_type: Optional[str] = None
   n_episodes: int = 10
   seed: int = 0
   device: str = "cuda"


def _get_rl_config(env_name: str):
   if env_name in mujoco_playground.manipulation._envs:
      return manipulation_params.brax_ppo_config(env_name)
   if env_name in mujoco_playground.locomotion._envs:
      return locomotion_params.brax_ppo_config(env_name)
   if env_name in mujoco_playground.dm_control_suite._envs:
      return dm_control_suite_params.brax_ppo_config(env_name)
   raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def _resolve_checkpoint_path(load_checkpoint_path: Optional[str]) -> Path:
   if not load_checkpoint_path:
      raise ValueError("--load_checkpoint_path is required for PPO test.")

   ckpt_path = Path(load_checkpoint_path).expanduser().resolve()
   if not ckpt_path.exists():
      raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")

   if not ckpt_path.is_dir():
      raise ValueError(f"Checkpoint path must be a directory, got: {ckpt_path}")

   subdirs = [d for d in ckpt_path.iterdir() if d.is_dir() and d.name.isdigit()]
   if subdirs:
      subdirs.sort(key=lambda d: int(d.name))
      return subdirs[-1]
   return ckpt_path


def _to_policy_obs(observation):
   obs = jnp.asarray(observation)
   # PPO configs in this repo typically read state observations via policy_obs_key.
   return {"state": obs}


def load(config: PPOTestConfig) -> Callable[[np.ndarray], np.ndarray]:
   checkpoint_path = _resolve_checkpoint_path(config.load_checkpoint_path)
   ppo_params = _get_rl_config(config.env)

   training_params = dict(ppo_params)
   training_params["num_timesteps"] = 0
   if "network_factory" in training_params:
      del training_params["network_factory"]

   network_factory_cfg: Any = ppo_params.get("network_factory", {})
   if hasattr(network_factory_cfg, "to_dict"):
      network_kwargs = network_factory_cfg.to_dict()
   elif isinstance(network_factory_cfg, dict):
      network_kwargs = network_factory_cfg
   else:
      network_kwargs = {}

   train_fn = functools.partial(
      ppo.train,
      **training_params,
      network_factory=functools.partial(
         ppo_networks.make_ppo_networks,
         **network_kwargs,
      ),
      seed=config.seed,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      restore_checkpoint_path=str(checkpoint_path),
   )

   env_cfg = registry.get_default_config(config.env)
   train_env = registry.load(config.env, config=env_cfg)
   eval_env = registry.load(config.env, config=env_cfg)

   print(f"Loading PPO checkpoint from: {checkpoint_path}")
   make_inference_fn, params, _ = train_fn(environment=train_env, eval_env=eval_env)
   inference_fn = make_inference_fn(params, deterministic=True)

   rng = jax.random.PRNGKey(config.seed)

   def policy_fn(observation: np.ndarray) -> np.ndarray:
      nonlocal rng
      rng, act_rng = jax.random.split(rng)
      action, _ = inference_fn(_to_policy_obs(observation), act_rng)
      return np.asarray(action)

   return policy_fn


render_trajectory = []


def render_callback(_, state):
   render_trajectory.append(state)


def evaluate(
   policy_fn: Callable[[np.ndarray], np.ndarray],
   env: GymWrapper,
   num_episodes: int,
   render: bool = False,
):
   episode_returns = []
   for _ in range(num_episodes):
      episode_return = 0
      observation, _ = env.reset()
      done = truncated = False
      while not done and not truncated:
         action = policy_fn(observation)
         observation, reward, done, truncated, _ = env.step(action)
         if render:
               env.render()
         episode_return += float(np.asarray(reward).mean())
      episode_returns.append(episode_return)

   return float(np.mean(episode_returns))


@pyrallis.wrap()  # type: ignore
def test(config: PPOTestConfig):
   env = get_env(
      config.env,
      device=config.device,
      render_callback=render_callback,
      command_type=config.command_type,
   )

   policy_fn = load(config)
   mean_return = evaluate(policy_fn, env, config.n_episodes, render=True)
   print(f"Mean Return over {config.n_episodes} episodes: {mean_return}")

   timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
   env.save_video(render_trajectory, save_path=f"ppo-{config.env}-{timestamp}.mp4")


if __name__ == "__main__":
   test()  # type: ignore
