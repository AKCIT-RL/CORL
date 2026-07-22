import mujoco

from mujoco_playground import registry
import gymnasium as gym
import numpy as np
import os
import ast
import json
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
    num_actors: int = 1,
    dataset=None,
    config_overrides=None,
):
    # ``config_overrides`` is a flattened-dotted-key dict forwarded to
    # ``registry.load`` (applied via ``ConfigDict.update_from_flattened_dict``).
    # It is used by the Tier-5 shifted evaluation to build an env whose
    # perturbation regime differs from the one seen during collection (e.g.
    # stronger push-recovery kicks) while the D4RL reference scores below still
    # come from the (in-distribution) dataset metadata, so the shifted score
    # stays on the same scale as the in-distribution one.
    #
    # Curriculum/rough-terrain envs must use the JAX/MJX collision backend: the
    # warp CCD path is unreliable on the procedural heightfield (it OOMs the GPU
    # or silently drops contacts). Force ``impl=jax`` here so eval dynamics match
    # the dataset, mirroring the collection-time override in
    # Offline-RL-Benchmark/collect_data.py. An explicit caller override still wins.
    if env_name == "Go2RoughCurriculum":
        config_overrides = {"impl": "jax", **(config_overrides or {})}
    env = registry.load(env_name, config_overrides=config_overrides)
    env_cfg = registry.get_default_config(env_name)

    randomization_fn = None
    if randomize:
        randomization_fn = registry.get_domain_randomizer(env_name)

    env = GymWrapper(
        env,
        env_cfg,
        seed=1,
        num_actors=num_actors,
        device=device,
        command_type=command_type,
        render_callback=render_callback,
        randomization_fn=randomization_fn,
        dataset=dataset,
    )

    return env


def maybe_get_shifted_env(env_name, device, command_type=None, dataset=None, eval_shift=None):
    """Build a Tier-5 shifted-evaluation env, or return None when not requested.

    ``eval_shift`` is the per-task ``eval_shift`` block from _datasets.yaml, passed
    as a JSON string (via ``--eval_shift``) or an already-parsed dict. Its keys are
    flattened-dotted env-config overrides (e.g. ``pert_config.velocity_kick``) that
    define a harder / held-out regime than the collection one. The dataset is still
    forwarded so the D4RL reference scores stay in-distribution, keeping the shifted
    score comparable to the in-distribution one.
    """
    if not eval_shift:
        return None
    if isinstance(eval_shift, str):
        try:
            overrides = json.loads(eval_shift)
        except json.JSONDecodeError:
            # pyrallis may round-trip the JSON through yaml/str(), turning it
            # into a single-quoted Python dict repr; fall back to literal_eval.
            overrides = ast.literal_eval(eval_shift)
    else:
        overrides = dict(eval_shift)
    return get_env(
        env_name,
        device,
        command_type=command_type,
        dataset=dataset,
        config_overrides=overrides,
    )


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
        dataset=None,
    ):
        super().__init__()
        self.command_type = command_type
        self.env = env
        self.device = device
        self.rng = jax.random.PRNGKey(seed)
        self.render_callback = render_callback
        self.episode_length = env_cfg.episode_length
        self._randomize_every_episode = randomize_every_episode

        # D4RL-style normalization reference scores, read from the Minari
        # dataset metadata written at collection time (return_min = weakest
        # checkpoint return, return_expert = expert peak). Kept as None when no
        # dataset is provided so get_normalized_score falls back to raw returns.
        self.ref_min_score = None
        self.ref_max_score = None
        if dataset is not None:
            metadata = getattr(dataset.storage, "metadata", None) or {}
            self.ref_min_score = metadata.get(
                "return_min", metadata.get("ref_min_score")
            )
            self.ref_max_score = metadata.get(
                "return_expert", metadata.get("ref_max_score")
            )

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

        # Curriculum envs (e.g. Go2RoughCurriculum) expose ``reset_to(rng, level,
        # col)`` and a ``num_rows x num_cols`` terrain grid. Their plain ``reset``
        # always spawns on the easiest level (row 0), so evaluating through it
        # would only ever measure tier-0 terrain. Detect the curriculum interface
        # here so ``reset`` can instead spread episodes across *all* difficulty
        # levels (matching how the dataset was collected), giving an all-level
        # average score. The curriculum auto-reset promotion/regression is a
        # training-only wrapper and is intentionally not used at eval time.
        base = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        self._is_curriculum = (
            hasattr(base, "reset_to")
            and hasattr(base, "num_rows")
            and hasattr(base, "num_cols")
        )
        if self._is_curriculum:
            self._curriculum_base = base
            self._num_rows = int(base.num_rows)
            self._num_cols = int(base.num_cols)
            self._next_level = 0  # round-robin cursor across successive resets

        if randomization_fn is not None:
            self._setup_domain_randomization(randomization_fn, num_actors)
        else:
            self._randomization_fn = None
            self._base_mjx_model = None
            self._mjx_model_v = None
            self._in_axes = None
            self._reset_fn = jax.jit(jax.vmap(self.env.reset))
            self._step_fn = jax.jit(jax.vmap(self.env.step))
            if self._is_curriculum:
                self._reset_to_fn = jax.jit(jax.vmap(self._curriculum_base.reset_to))

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

        if self.command_type == "forwardbackward":
            command = zeros.at[..., 0].set(commands[..., 0])
        elif self.command_type == "forward":
            command = zeros.at[..., 0].set(jp.abs(commands[..., 0]))
        elif self.command_type == "forwardfixed":
            command = zeros.at[..., 0].set(1.0)
        elif self.command_type == "forward_realrobot":
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

    def reset_rng(self, seed: int = 0):
        """Reseed the internal RNG so evaluation rollouts are reproducible.

        Calling this at the start of an evaluation makes every eval use the same
        sequence of reset states, so scores are comparable across checkpoints
        and the final evaluation matches the intermediate ones for a fixed policy.
        """
        self.rng = jax.random.PRNGKey(seed)
        # Restart the curriculum level round-robin so every evaluation covers the
        # difficulty levels in the same, balanced order.
        if getattr(self, "_is_curriculum", False):
            self._next_level = 0

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
        elif self._is_curriculum:
            # Spread episodes across all difficulty levels via ``reset_to`` in a
            # round-robin over successive resets (random terrain column each
            # time). Over a full evaluation every level gets an equal share of
            # episodes, so the aggregate score reflects all tiers instead of just
            # the easiest one.
            levels = (self._next_level + jp.arange(self.num_envs)) % self._num_rows
            self._next_level = int(
                (self._next_level + self.num_envs) % self._num_rows
            )
            self.rng, col_rng = jax.random.split(self.rng)
            cols = jax.random.randint(
                col_rng, (self.num_envs,), 0, self._num_cols
            )
            self._last_levels = np.asarray(levels)
            self.env_state = self._reset_to_fn(
                reset_keys, levels.astype(jp.int32), cols.astype(jp.int32)
            )
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

    def get_normalized_score(self, score):
        """D4RL-style normalized score, where ~1.0 corresponds to the expert.

            (score - ref_min) / (ref_max - ref_min)

        Returns None when reference scores are unavailable (no dataset was
        passed) so callers can fall back to the raw return.
        """
        if (
            self.ref_min_score is None
            or self.ref_max_score is None
            or self.ref_max_score <= self.ref_min_score
        ):
            return None
        return (score - self.ref_min_score) / (self.ref_max_score - self.ref_min_score)

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


def record_policy_video(
    env_name: str,
    act,
    obs_mean,
    obs_std,
    device: str,
    save_path: str,
    command_type=None,
    seed: int = 1,
):
    """Roll out a policy for a single episode and save an mp4 of the rollout.

    Args:
        env_name: mujoco_playground env id (same value used for evaluation).
        act: callable mapping a normalized observation (np.ndarray) to an action.
        obs_mean, obs_std: observation normalization stats used during training.
        device: jax device string (e.g. "cuda:0").
        save_path: destination path for the .mp4 file.
        command_type: optional command override for joystick envs.
        seed: env reset seed.

    Returns:
        The (raw) episode return obtained during the recorded rollout.
    """
    # Headless rendering backend (matches algorithms/utils/save_video.py).
    os.environ.setdefault("MUJOCO_GL", "egl")

    render_trajectory = []

    def render_callback(_, state):
        render_trajectory.append(state)

    env = get_env(
        env_name,
        device,
        render_callback=render_callback,
        command_type=command_type,
    )

    observation, _ = env.reset()
    done = truncated = False
    episode_return = 0.0
    while not done and not truncated:
        obs_n = (observation - obs_mean) / (obs_std + 1e-5)
        action = np.asarray(act(obs_n))
        observation, reward, done, truncated, _ = env.step(action)
        env.render()
        episode_return += float(np.asarray(reward).reshape(-1)[0])
        done = bool(np.asarray(done).reshape(-1)[0])
        truncated = bool(np.asarray(truncated).reshape(-1)[0])

    # Establish a headless GL context before rendering (mirrors save_video.py).
    try:
        import mujoco.egl

        gl_context = mujoco.egl.GLContext(1024, 1024)
        gl_context.make_current()
    except Exception as e:  # pragma: no cover - depends on GPU/driver
        print(f"[record_policy_video] could not create EGL context: {e}")

    save_dir = os.path.dirname(os.path.abspath(save_path))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    env.save_video(render_trajectory, save_path=save_path)
    return episode_return