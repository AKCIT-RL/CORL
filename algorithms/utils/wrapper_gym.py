import mujoco
from mujoco import mjx

from mujoco_playground import wrapper_torch, wrapper
from mujoco_playground import registry
import gymnasium as gym
import numpy as np
import torch

from .space import NumpySpace


def get_env(env_name: str, device: str):
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)
    randomizer = registry.get_domain_randomizer(env_name)

    render_trajectory = []

    def render_callback(_, state):
        render_trajectory.append(state)

    env = GymWrapper(
        env,
        num_actors=1,
        seed=1,
        episode_length=env_cfg.episode_length,
        action_repeat=1,
        render_callback=render_callback,
        randomization_fn=randomizer,
        device=device,
    )

    return env


class GymWrapper(wrapper_torch.RSLRLBraxWrapper):
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

        self.device = device
        if isinstance(self.num_obs, tuple):
            self.observation_space = NumpySpace(shape=self.num_obs, dtype=np.float32)
        else:
            self.observation_space = NumpySpace(shape=(self.num_obs,), dtype=np.float32)
        self.action_space = NumpySpace(shape=(self.num_actions,), dtype=np.float32)

    def step(self, action):
        action = np.array([action])  # Convert to numpy array first
        action = torch.from_numpy(action).to(self.device)
        action = torch.clip(action, -1.0, 1.0)  # pytype: disable=attribute-error
        action = wrapper_torch._torch_to_jax(action)
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
        return (
            obs.cpu().numpy(),
            reward.cpu().numpy(),
            done.cpu().numpy(),
            truncation.cpu().numpy(),
            info_ret,
        )

    def reset(self):
        # todo add random init like in collab examples?
        self.env_state = self.reset_fn(self.key_reset)

        if self.asymmetric_obs:
            obs = wrapper_torch._jax_to_torch(self.env_state.obs["state"])
        # critic_obs = jax_to_torch(self.env_state.obs["privileged_state"])
        else:
            obs = wrapper_torch._jax_to_torch(self.env_state.obs)
        return obs.cpu().numpy(), {}
