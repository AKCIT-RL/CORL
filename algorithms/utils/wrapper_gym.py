from pyparsing import Callable, Optional
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

from .space import NumpySpace

try:
    from algorithms.utils.save_video import render_callback
except Exception:
    render_callback = None

def get_env(env_name: str, device: str, num_actors: int = 1, render_callback=None, command_type=None):
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)
    randomizer = registry.get_domain_randomizer(env_name)

    env = GymWrapper(
        env,
        num_actors=num_actors,
        seed=1,
        episode_length=env_cfg.episode_length,
        action_repeat=1,
        randomization_fn=randomizer,
        device=device,
        render_callback=render_callback,
        command_type=command_type,
    )

    return env


class GymWrapper(wrapper_torch.RSLRLBraxWrapper, gym.Env):
    def __init__(
        self,
        env,
        num_actors,
        seed,
        episode_length,
        action_repeat,
        randomization_fn=None,
        render_callback=None,
        device_rank=None,
        device="cpu",
        command_type=None,
    ):
        super().__init__(
            env,
            num_actors,
            seed,
            episode_length,
            action_repeat,
            randomization_fn,
            render_callback,
            device_rank,
        )

        self.env_unwrapped = env
        self.command_type = command_type
        self.device = device
        if isinstance(self.num_obs, tuple):
            self.observation_space = NumpySpace(shape=self.num_obs, dtype=np.float32)
        else:
            self.observation_space = NumpySpace(shape=(self.num_obs,), dtype=np.float32)
        self.action_space = NumpySpace(shape=(self.num_actions,), dtype=np.float32)

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.to(self.device, dtype=torch.float32)
        else:
            action = torch.as_tensor(action, device=self.device, dtype=torch.float32)

        if action.dim() == 1:
            action = action.unsqueeze(0)                 # (1, A)
        elif action.dim() > 2:
            action = action.view(-1, action.shape[-1])   # (B, A)

        action = torch.clamp(action, -1.0, 1.0)
        self.env_state = self.step_fn(self.env_state, action)
        critic_obs = None
        if self.asymmetric_obs:
            obs = wrapper_torch._jax_to_torch(self.env_state.obs["state"])
            critic_obs = wrapper_torch._jax_to_torch(
                self.env_state.obs["privileged_state"]
            )
        else:
            obs = wrapper_torch._jax_to_torch(self.env_state.obs)
        reward = wrapper_torch._jax_to_torch(self.env_state.reward)
        done = wrapper_torch._jax_to_torch(self.env_state.done)
        info = self.env_state.info
        truncation = wrapper_torch._jax_to_torch(info["truncation"])

        info_ret = {
            "time_outs": truncation,
            "observations": {"critic": critic_obs},
            "log": {},
        }

        if "last_episode_success_count" in info:
            last_episode_success_count = (
                wrapper_torch._jax_to_torch(info["last_episode_success_count"])[
                    done > 0
                ]  # pylint: disable=unsubscriptable-object
                .float()
                .tolist()
            )
            if len(last_episode_success_count) > 0:
                self.success_queue.extend(last_episode_success_count)
            info_ret["log"]["last_episode_success_count"] = np.mean(self.success_queue)

        for k, v in self.env_state.metrics.items():
            if k not in info_ret["log"]:
                info_ret["log"][k] = (
                    wrapper_torch._jax_to_torch(v).float().mean().item()
                )

        # next_observation, reward, terminal, truncated, info = env.step(action)
        obs_np = obs.cpu().numpy()
        if obs_np.ndim > 1 and obs_np.shape[0] == 1:
            obs_np = obs_np.squeeze(0)

        reward_np = reward.cpu().numpy()
        if reward_np.size == 1:
            reward_np = float(reward_np.reshape(-1)[0])

        done_np = done.cpu().numpy()
        if done_np.size == 1:
            done_np = bool(done_np.reshape(-1)[0])

        trunc_np = truncation.cpu().numpy()
        if trunc_np.size == 1:
            trunc_np = bool(trunc_np.reshape(-1)[0])

        return (
            obs_np,
            reward_np,
            done_np,
            trunc_np,
            info_ret,
        )

    def reset(self, *, seed=None, options=None):
        # Generate fresh reset keys each call (collab example style)
        # Ensures different initial states across resets without extra overhead.
        self.key, reset_key = jax.random.split(self.key)
        self.key_reset = jax.random.split(reset_key, self.num_envs)
        self.env_state = self.reset_fn(self.key_reset)
        obs = self.env_state.obs
        if self.command_type == "fowardbackward":
            command = jp.concatenate([
                self.env_state.info["command"][:, [0]],  # shape (batch, 1)
                jp.zeros((self.env_state.info["command"].shape[0], 2), dtype=self.env_state.info["command"].dtype)
            ], axis=1)
            self.env_state.info["command"] = command
            obs["state"] = obs["state"].at[..., -3:].set(command)
        elif self.command_type == "foward":
            command = jp.concatenate([
                jp.abs(self.env_state.info["command"][:, [0]]),  # shape (batch, 1)
                jp.zeros((self.env_state.info["command"].shape[0], 2), dtype=self.env_state.info["command"].dtype)
            ], axis=1)
            self.env_state.info["command"] = command
            obs["state"] = obs["state"].at[..., -3:].set(command)
        elif self.command_type == "fowardfixed":
            command = jp.array([[1.0, 0.0, 0.0]] * self.env_state.info["command"].shape[0])
            self.env_state.info["command"] = command
            obs["state"] = obs["state"].at[..., -3:].set(command)

        if self.asymmetric_obs:
            obs = wrapper_torch._jax_to_torch(obs["state"])
        # critic_obs = jax_to_torch(self.env_state.obs["privileged_state"])
        else:
            obs = wrapper_torch._jax_to_torch(obs)
        obs_np = obs.cpu().numpy()
        if obs_np.ndim > 1 and obs_np.shape[0] == 1:
            obs_np = obs_np.squeeze(0)
        return obs_np, {}

    def save_video(self, render_trajectory, save_path=None):
        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[2] = True
        scene_option.geomgroup[3] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

        render_every = 2
        fps = 1.0 / self.env_unwrapped.dt / render_every
        traj = render_trajectory[::render_every]
        frames = self.env_unwrapped.render(
            traj,
            camera="track",
            height=480,
            width=640,
            scene_option=scene_option,
        )
        if save_path is not None:
            media.write_video(save_path, frames, fps=fps)