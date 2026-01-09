import mujoco
from mujoco import mjx

from mujoco_playground import wrapper_torch, wrapper
from mujoco_playground import registry
import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jp
import torch
import mediapy as media

from collections.abc import Mapping
try:
    from flax.core import frozen_dict
except ImportError:
    frozen_dict = None

from .space import NumpySpace


def get_env(env_name: str, device: str, render_callback=None, command_type=None):
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)

    env = GymWrapper(
        env,
        env_cfg,
        seed=1,
        num_actors=1,
        device=device,
        command_type=command_type,
        render_callback=render_callback,
    )

    return env


class GymWrapper(gym.Env):
    def __init__(
        self,
        env,
        env_cfg,
        seed,
        num_actors=1,
        device="cpu",
        command_type=None,
        render_callback=None,
    ):
        super().__init__()
        self.command_type = command_type
        self.env = env
        self.device = device
        self._reset_fn = jax.jit(jax.vmap(self.env.reset))
        self._step_fn = jax.jit(jax.vmap(self.env.step))
        self.rng = jax.random.PRNGKey(seed)
        self.render_callback = render_callback
        self.episode_length = env_cfg.episode_length

        # Handle both dict-based and int-based observation_size
        obs_size = self.env.observation_size
        if isinstance(obs_size, dict):
            obs_size = obs_size["state"]
        
        if isinstance(obs_size, tuple):
            self.observation_space = NumpySpace(shape=obs_size, dtype=np.float32)
        else:
            self.observation_space = NumpySpace(shape=(obs_size,), dtype=np.float32)
        self.action_space = NumpySpace(shape=(self.env.action_size,), dtype=np.float32)

        self.num_envs = num_actors

        self.timesteps = 0

    def _maybe_unfreeze(self, tree):
        if frozen_dict and isinstance(tree, frozen_dict.FrozenDict):
            return tree.unfreeze()
        if isinstance(tree, Mapping):
            return dict(tree)
        return tree

    def _tree_to_numpy(self, tree):
        if isinstance(tree, Mapping):
            return {k: self._tree_to_numpy(v) for k, v in tree.items()}
        if isinstance(tree, (list, tuple)):
            return type(tree)(self._tree_to_numpy(v) for v in tree)
        return np.asarray(tree)

    def _apply_command_override(self, env_state):
        if self.command_type is None or "command" not in env_state.info:
            return env_state

        commands = env_state.info["command"]
        zeros = jp.zeros_like(commands)

        if self.command_type == "fowardbackward":
            command = zeros.at[..., 0].set(commands[..., 0])
        elif self.command_type == "foward":
            command = zeros.at[..., 0].set(jp.abs(commands[..., 0]))
        elif self.command_type == "fowardfixed":
            command = zeros.at[..., 0].set(1.0)
        else:
            return env_state

        obs = self._maybe_unfreeze(env_state.obs)
        if isinstance(obs, dict):
            obs["state"] = obs["state"].at[..., -3:].set(command)
        else:
            obs = obs.at[..., -3:].set(command)

        info = self._maybe_unfreeze(env_state.info)
        info["command"] = command

        return env_state.replace(obs=obs, info=info)

    def reset(self, *, seed=None, options=None):
        self.rng, reset_rng = jax.random.split(self.rng)
        reset_keys = jax.random.split(reset_rng, self.num_envs)
        self.env_state = self._reset_fn(reset_keys)
        self.env_state = self._apply_command_override(self.env_state)
        self.timesteps = 0
        obs_field = self.env_state.obs
        if isinstance(obs_field, dict):
            obs = np.asarray(obs_field["state"])
        else:
            obs = np.asarray(obs_field)
        return obs, {}

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = jp.asarray(action)
        if len(action.shape) == 1:
            action = action[None, ...]

        self.env_state = self._step_fn(self.env_state, action)
        self.env_state = self._apply_command_override(self.env_state)
        self.timesteps += 1
        obs_field = self.env_state.obs
        if isinstance(obs_field, dict):
            obs = np.asarray(obs_field["state"])
        else:
            obs = np.asarray(obs_field)
        rew = np.asarray(self.env_state.reward)
        done = np.asarray(self.env_state.done)
        truncated = np.asarray([self.timesteps >= self.episode_length for _ in range(self.num_envs)])
        info = self._tree_to_numpy(self.env_state.info)
        return obs, rew, done, truncated, info

    def render(self):  # pylint: disable=unused-argument
        if self.render_callback is not None:
            self.render_callback(self.env, self.env_state)
        else:
            raise ValueError("No render callback specified")

    def save_video(self, render_trajectory, save_path=None):
        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[2] = True
        scene_option.geomgroup[3] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

        render_every = 2
        fps = 1.0 / self.env.dt / render_every
        traj = render_trajectory[::render_every]
        frames = self.env.render(
            traj,
            camera="track",
            height=480,
            width=640,
            scene_option=scene_option,
        )
        if save_path is not None:
            media.write_video(save_path, frames, fps=fps)