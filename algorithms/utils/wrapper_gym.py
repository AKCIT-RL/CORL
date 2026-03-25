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


def get_env(
    env_name: str,
    device: str,
    render_callback=None,
    command_type=None,
    randomize: bool = False,
):
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)

    randomization_fn = None
    if randomize:
        randomization_fn = registry.get_domain_randomizer(env_name)

    env = GymWrapper(
        env,
        env_cfg,
        seed=1,
        num_actors=1,
        device=device,
        command_type=command_type,
        render_callback=render_callback,
        randomization_fn=randomization_fn,
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
        randomization_fn=None,
        randomize_every_episode: bool = True,
    ):
        super().__init__()
        self.command_type = command_type
        self.env = env
        self.device = device
        self.rng = jax.random.PRNGKey(seed)
        self.render_callback = render_callback
        self.episode_length = env_cfg.episode_length
        self._randomize_every_episode = randomize_every_episode

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

        if randomization_fn is not None:
            self._setup_domain_randomization(randomization_fn, num_actors)
        else:
            self._randomization_fn = None
            self._base_mjx_model = None
            self._mjx_model_v = None
            self._in_axes = None
            self._reset_fn = jax.jit(jax.vmap(self.env.reset))
            self._step_fn = jax.jit(jax.vmap(self.env.step))

    def _setup_domain_randomization(self, randomization_fn, num_actors):
        """Build JIT-compiled vmapped reset/step functions with domain randomization.

        At each reset the physics parameters (mass, friction, armature, etc.) are
        re-sampled so every evaluation episode runs in a different simulated world.
        The JIT cache is reused across episodes because the structure (in_axes) of
        the randomized model never changes, only its values do.
        """
        self._randomization_fn = randomization_fn
        self._base_mjx_model = self.env.mjx_model

        # Compute the initial randomized model batch to determine in_axes.
        init_keys = jax.random.split(self.rng, num_actors)
        self._mjx_model_v, self._in_axes = randomization_fn(
            self._base_mjx_model, init_keys
        )

        # The context-manager swap pattern mirrors BraxDomainRandomizationVmapWrapper:
        # during JAX tracing, _mjx_model is temporarily replaced with the abstract
        # vmapped tracer so the env's reset/step record the correct computation graph.
        env_inner = self.env.unwrapped

        def dr_reset(mjx_model, rng):
            old = env_inner._mjx_model
            env_inner._mjx_model = mjx_model
            try:
                return env_inner.reset(rng)
            finally:
                env_inner._mjx_model = old

        def dr_step(mjx_model, state, action):
            old = env_inner._mjx_model
            env_inner._mjx_model = mjx_model
            try:
                return env_inner.step(state, action)
            finally:
                env_inner._mjx_model = old

        self._reset_fn = jax.jit(
            jax.vmap(dr_reset, in_axes=[self._in_axes, 0])
        )
        self._step_fn = jax.jit(
            jax.vmap(dr_step, in_axes=[self._in_axes, 0, 0])
        )

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
        elif self.command_type == "foward_realrobot":
            command = zeros.at[..., 0].set(0.2)
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

        if self._randomization_fn is not None:
            if self._randomize_every_episode:
                self.rng, dr_rng = jax.random.split(self.rng)
                dr_keys = jax.random.split(dr_rng, self.num_envs)
                self._mjx_model_v, _ = self._randomization_fn(
                    self._base_mjx_model, dr_keys
                )
            self.env_state = self._reset_fn(self._mjx_model_v, reset_keys)
        else:
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

        if self._randomization_fn is not None:
            self.env_state = self._step_fn(self._mjx_model_v, self.env_state, action)
        else:
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
        # Visual mesh geoms (group 2) are stripped at compile time by _strip_obj_meshes
        # and discardvisual="true" in the scene XML.  Collision capsules live in group 3
        # and are the only robot geometry left in the compiled model, so we must enable
        # that group to make the robot visible.
        scene_option.geomgroup[2] = True
        scene_option.geomgroup[3] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

        # Improve lighting so the floor and capsules are visible even when the
        # scene XML does not define explicit lights or uses a washed-out texture.
        mj_model = self.env.mj_model
        mj_model.vis.headlight.active = 1
        mj_model.vis.headlight.ambient[:] = [0.3, 0.3, 0.3]
        mj_model.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]
        if mj_model.nmat > 0:
            mj_model.mat_emission[:] = 0.3

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