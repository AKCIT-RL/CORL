from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import mujoco

from mujoco_playground import registry
import gymnasium as gym
import numpy as np
import os
import ast
import json
import jax
import jax.numpy as jnp
import torch
import mediapy as media
import shutil

# Compute nodes have no system ffmpeg; fall back to the binary bundled with
# imageio-ffmpeg so mediapy can still encode videos.
if shutil.which("ffmpeg") is None:
    try:
        import imageio_ffmpeg

        media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    except (ImportError, RuntimeError):
        pass

from collections.abc import Mapping
from collections import deque

import yaml
try:
    from flax.core import frozen_dict
except ImportError:
    frozen_dict = None

from .space import NumpySpace

ENV_NAME = "Go2JoystickFlatTerrain"    

def locate_command_slice(obs_example, command_example, atol=1e-4):
    """Return ``(start, stop)`` of the contiguous command block inside a 1-D obs.

    The env writes ``info["command"]`` into the observation at a fixed offset that
    is NOT always the tail: e.g. ``H1JoystickGaitTracking`` appends gait features
    (contact/phase/gait_freq/gait/foot_height) AFTER the command, so a blind
    ``[..., -3:]`` slice overwrites ``foot_height`` instead of the command and
    destabilizes the gait policy. We locate the command by matching its values in
    the observation (last match wins). Falls back to the tail when nothing matches
    (command-at-tail envs like the Go2 joystick). Host-side only (numpy).
    """
    obs = np.asarray(obs_example).ravel()
    cmd = np.asarray(command_example).ravel()
    cd = cmd.shape[0]
    found = None
    for i in range(obs.shape[0] - cd + 1):
        if np.allclose(obs[i:i + cd], cmd, atol=atol):
            found = i
    if found is None:
        return int(obs.shape[0] - cd), int(obs.shape[0])
    return int(found), int(found + cd)

@dataclass
class RandomizeFunctions:
    domain_randomize: Callable[[Any, Any], Tuple[Any, Any]]
    observation_randomize: Callable[[Any, Any], Any]
    action_step:Callable[[Any], Any]
    observation_factory: Optional[Callable[[Any, Any], Callable[[Any, Any], Any]]] = None
    # Optional factory: given an action size and an RNG key returns a per-actor action function
    # with signature `action_fn(action) -> action`.
    action_factory: Optional[Callable[[Any, Any], Callable[[Any], Any]]] = None
    randomize_every_episode: bool = True

@dataclass(frozen=True)
class ObsLayout:
    """Where the perturbable sensor readings sit in an env's policy observation.

    ``key`` is the observation dict entry the policy sees ("state"), or None for a
    flat observation. ``slices`` maps each component to its [start, stop) range;
    anything not listed (last action, command, phase, goal, ...) is never touched.
    ``mirrors`` lists extra copies of the same reading: they get the same
    perturbation, so a policy cannot bypass the noise through a clean duplicate.
    """
    key: Optional[str]
    size: int
    slices: Dict[str, Tuple[int, int]]
    mirrors: Dict[str, Tuple[Tuple[int, int], ...]] = field(default_factory=dict)


_OBS_NOISE_FIELDS = ("linvel", "gyro", "gravity", "joint_angles", "joint_vel")
_GO2_LOCOMOTION = {
    "linvel": (0, 3), "gyro": (3, 6), "gravity": (6, 9),
    "joint_angles": (9, 21), "joint_vel": (21, 33),
}

# Read off each env's _get_obs in mujoco_playground; checked against the simulator
# state by scripts/check_randomize_envs.py.
OBS_LAYOUTS: Dict[str, ObsLayout] = {
    # linvel, gyro, gravity, joints, joint vel, last act + command (3)
    "Go2JoystickFlatTerrain": ObsLayout("state", 48, _GO2_LOCOMOTION),
    "Go2PushRecovery": ObsLayout("state", 48, _GO2_LOCOMOTION),
    "Go2RoughCurriculum": ObsLayout("state", 48, _GO2_LOCOMOTION),
    # ... + last act, no command
    "Go2Handstand": ObsLayout("state", 45, _GO2_LOCOMOTION),
    "Go2Footstand": ObsLayout("state", 45, _GO2_LOCOMOTION),
    # ... + last act, stood (1), local goal (2): same size as joystick, other tail
    "Go2GetupWalk": ObsLayout("state", 48, _GO2_LOCOMOTION),
    # no linvel: gyro, gravity, joints, joint vel, last act
    "Go2Getup": ObsLayout("state", 42, {
        "gyro": (0, 3), "gravity": (3, 6), "joint_angles": (6, 18), "joint_vel": (18, 30),
    }),
    # linvel, gyro, gravity, command (3), 29 joints, joint vel, last act, phase (4)
    "G1JoystickFlatTerrain": ObsLayout("state", 103, {
        "linvel": (0, 3), "gyro": (3, 6), "gravity": (6, 9),
        "joint_angles": (12, 41), "joint_vel": (41, 70),
    }),
    # flat: gyro, gravity, 19 joints, joint vel, last act, command (3), then a
    # one-step qvel_history and qpos_error_history built from the clean state
    "H1JoystickGaitTracking": ObsLayout(None, 113, {
        "gyro": (0, 3), "gravity": (3, 6), "joint_angles": (6, 25), "joint_vel": (25, 44),
    }, mirrors={"joint_vel": ((66, 85),), "joint_angles": ((85, 104),)}),
}


# Where a `geom_friction` scale lands:
#   floor_contacts -- every floor contact's sliding friction moves by the same factor
#                     as the floor (the default);
#   floor_geom     -- only geom_friction[0, 0], as the suites evaluated before this
#                     option existed did. G1 and H1 then never feel the low end of
#                     the range (see floor_friction_targets); kept to reproduce them.
FRICTION_TARGETS = ("floor_contacts", "floor_geom")


def floor_friction_targets(model) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(pair_ids, geom_ids)`` whose friction must follow the floor's.

    MuJoCo does not always use the floor's own coefficient for a floor contact:
      * an explicit <pair> carries its own friction (G1: both feet, mu = 0.6);
      * for geom contacts, the higher-priority geom wins, and equal priorities take
        the max (H1: floor and feet both mu = 1.0, so the max floors any decrease).
    Scaling those pairs, and the partners that are not outranked by the floor, by
    the floor's factor gives every floor contact s * its nominal coefficient:
    max(s*a, s*b) = s*max(a, b). Lower-priority partners (Go2 feet) need nothing.

    Read from the static model, as MJX's collision driver builds its contacts, so a
    partner's priority cannot itself be randomized.
    """
    # The pair list MJX itself collides, with excludes and pair overrides applied.
    from mujoco.mjx._src.collision_driver import geom_pairs

    pairs = np.asarray(list(geom_pairs(model)), dtype=int).reshape(-1, 3)
    floor = (pairs[:, 0] == 0) | (pairs[:, 1] == 0)
    pair_ids = np.unique(pairs[floor & (pairs[:, 2] >= 0), 2])
    geom = pairs[floor & (pairs[:, 2] < 0)]
    partners = np.where(geom[:, 0] == 0, geom[:, 1], geom[:, 0])
    priority = np.asarray(model.geom_priority)
    geom_ids = np.unique(partners[priority[partners] >= priority[0]])
    # Scaling a partner's own friction would leak into its non-floor contacts.
    others = pairs[~floor & (pairs[:, 2] < 0)][:, :2]
    leaked = np.intersect1d(geom_ids, others)
    if leaked.size:
        raise ValueError(
            f"floor friction partners {leaked.tolist()} also touch other geoms; "
            "scaling them would change contacts away from the floor"
        )
    return pair_ids, geom_ids


def _concat_observation_dict(comp_dict: Dict[str, Any]) -> np.ndarray:
    """Concatenate component dictionary into the standard 48-dim observation vector.
    
    Order: linvel (3), gyro (3), gravity (3), joint_angles (12), joint_vel (12), last_act (12), command (3).
    """
    keys = ["linvel", "gyro", "gravity", "joint_angles", "joint_vel", "last_act", "command"]
    components = [np.asarray(comp_dict[k]) for k in keys if k in comp_dict]
    return np.concatenate(components, axis=-1)


def _extract_observation_components(obs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract clean (unrandomized) and noisy (randomized) observation component dicts.
    
    Returns:
        (clean_dict, noisy_dict) containing keys:
        'linvel', 'gyro', 'gravity', 'joint_angles', 'joint_vel', 'last_act', 'command'
    """
    if isinstance(obs, dict):
        if "privileged_state" in obs:
            privileged_state = np.asarray(obs["privileged_state"])
            noisy_state = privileged_state[..., :48]
            suffix = privileged_state[..., 48:]
            
            clean_dict = {
                "linvel": suffix[..., 9:12],
                "gyro": suffix[..., 0:3],
                "gravity": suffix[..., 6:9],
                "joint_angles": suffix[..., 15:27],
                "joint_vel": suffix[..., 27:39],
                "last_act": noisy_state[..., 33:45],
                "command": noisy_state[..., 45:48],
            }
            noisy_dict = {
                "linvel": noisy_state[..., 0:3],
                "gyro": noisy_state[..., 3:6],
                "gravity": noisy_state[..., 6:9],
                "joint_angles": noisy_state[..., 9:21],
                "joint_vel": noisy_state[..., 21:33],
                "last_act": noisy_state[..., 33:45],
                "command": noisy_state[..., 45:48],
            }
            return clean_dict, noisy_dict
        elif "state" in obs:
            state = np.asarray(obs["state"])
        else:
            state = np.asarray(obs)
    else:
        state = np.asarray(obs)

    comp_dict = {
        "linvel": state[..., 0:3],
        "gyro": state[..., 3:6],
        "gravity": state[..., 6:9],
        "joint_angles": state[..., 9:21],
        "joint_vel": state[..., 21:33],
        "last_act": state[..., 33:45],
        "command": state[..., 45:48],
    }
    return comp_dict, dict(comp_dict)


@dataclass
class RandomizeConfigs:
    class RandomizeType(Enum):
        DISABLED = "disabled"
        FULL = "full"
        ONLY_DOMAIN = "only_domain"
        ONLY_OBSERVATION = "only_observation" # default
        DEFAULT = "default"
        CUSTOM = "custom"
    
    type: RandomizeType = RandomizeType.DEFAULT
    
    @dataclass
    class RandomizeOptions:
        # Domain
        geom_friction: Optional[Callable[[Any, Any], jax.Array]] = None
        # One of FRICTION_TARGETS; set from geom_friction's `target` in the YAML.
        friction_target: str = "floor_contacts"
        dof_frictionloss: Optional[Callable[[Any, Any], jax.Array]] = None
        dof_armature: Optional[Callable[[Any, Any], jax.Array]] = None
        body_ipos: Optional[Callable[[Any, Any], jax.Array]] = None
        body_mass: Optional[Callable[[Any, Any], jax.Array]] = None
        qpos0: Optional[Callable[[Any, Any], jax.Array]] = None
        actuator_gainprm: Optional[Callable[[Any, Any], jax.Array]] = None
        motor_strength: Optional[Callable[[Any, Any], jax.Array]] = None
        
        # Observation
        gyro: Optional[Callable[[Any, Any], jax.Array]] = None
        gravity: Optional[Callable[[Any, Any], jax.Array]] = None
        joint_angles: Optional[Callable[[Any, Any], jax.Array]] = None
        joint_vel: Optional[Callable[[Any, Any], jax.Array]] = None
        linvel: Optional[Callable[[Any, Any], jax.Array]] = None

        # Observation factory: a callable that receives an observation and an RNG key and returns a randomized observation.
        observation_factory: Optional[Callable[[Any, Any], Callable[[Any, Any], jax.Array]]] = None

        # Action factory: a callable that receives an action size and and RNG key and returns a
        # per-actor action function with signature `action_fn(action) -> action`.
        action_factory: Optional[Callable[[Any, Any], Callable[[Any], jax.Array]]] = None
    
    configs: Optional[RandomizeOptions] = None
    env_name: str = ENV_NAME
    
    def __post_init__(self):
        if self.type != self.RandomizeType.CUSTOM and self.configs is not None:
            raise ValueError("configs should be None when type is not CUSTOM")
        
        if self.type == self.RandomizeType.CUSTOM and self.configs is None:
            raise ValueError("configs should not be None when type is CUSTOM")
    
    def _get_domain_randomize(
        self, enabled=True
    ) -> Callable[[Any, Any], Tuple[Any, Any]]:
        if enabled:
            return registry.get_domain_randomizer(self.env_name) # type: ignore
        
        def disabled_domain_randomizer(model, _):
            return model, None

        return disabled_domain_randomizer

    def _get_observation_randomize(
        self, enabled=True
    ) -> Callable[[Any, Any], Any]:
        if enabled:
            def observation_randomize(obs, _): # type: ignore
                if isinstance(obs, dict) and "state" in obs:
                    return np.asarray(obs["state"])
                # A flat observation has no component layout to decompose, and the
                # split below would reassemble it as the 48-dim Go2 vector: H1's 113
                # dims would be silently truncated instead of forwarded untouched.
                if not isinstance(obs, dict):
                    return np.asarray(obs)
                _, noisy_dict = _extract_observation_components(obs)
                return _concat_observation_dict(noisy_dict)
        else:
            def observation_randomize(obs, _):
                clean_dict, _ = _extract_observation_components(obs)
                return _concat_observation_dict(clean_dict)
        
        return observation_randomize

    def _get_action_function(self) -> Callable[[Any], Any]:
        def action_step(action): return action
        return action_step
    
    def _get_disabled_functions(self) -> RandomizeFunctions:
        domain_randomize = self._get_domain_randomize(enabled=False)
        observation_randomize = self._get_observation_randomize(enabled=False)
        action_step = self._get_action_function()
        
        return RandomizeFunctions(
            domain_randomize=domain_randomize,
            observation_randomize=observation_randomize,
            action_step=action_step,
            observation_factory=None,
            action_factory=None,
        )
    
    def _get_full_functions(self) -> RandomizeFunctions:
        domain_randomize = self._get_domain_randomize(enabled=True)
        observation_randomize = self._get_observation_randomize(enabled=True)
        action_step = self._get_action_function()
        
        return RandomizeFunctions(
            domain_randomize=domain_randomize,
            observation_randomize=observation_randomize,
            action_step=action_step,
            observation_factory=None,
            action_factory=None,
        )
    
    def _get_only_domain_functions(self) -> RandomizeFunctions:
        domain_randomize = self._get_domain_randomize(enabled=True)
        observation_randomize = self._get_observation_randomize(enabled=False)
        action_step = self._get_action_function()
        
        return RandomizeFunctions(
            domain_randomize=domain_randomize,
            observation_randomize=observation_randomize,
            action_step=action_step,
            observation_factory=None,
            action_factory=None,
        )
    
    def _get_default_functions(self) -> RandomizeFunctions:
        domain_randomize = self._get_domain_randomize(enabled=False)
        observation_randomize = self._get_observation_randomize(enabled=True)
        action_step = self._get_action_function()
        
        return RandomizeFunctions(
            domain_randomize=domain_randomize,
            observation_randomize=observation_randomize,
            action_step=action_step,
            observation_factory=None,
            action_factory=None,
        )
    
    def _get_custom_functions(self) -> RandomizeFunctions:
        domain_randomize = self._get_domain_randomize(enabled=False)
        observation_randomize = self._get_observation_randomize(enabled=True)
        action_step = self._get_action_function()
        
        def custom_domain_randomize(model, rng):
            if self.configs is None:
                return domain_randomize(model, rng)
            
            in_axes_replace = dict()
            model_replace_keys = []
            
            friction_pairs = friction_geoms = np.zeros(0, dtype=int)
            if self.configs.geom_friction is not None:
                model_replace_keys.append("geom_friction")
                if self.configs.friction_target == "floor_contacts":
                    friction_pairs, friction_geoms = floor_friction_targets(model)
                    if friction_pairs.size:
                        model_replace_keys.append("pair_friction")
            if self.configs.dof_frictionloss is not None:
                model_replace_keys.append("dof_frictionloss")
            if self.configs.dof_armature is not None:
                model_replace_keys.append("dof_armature")
            if self.configs.body_ipos is not None:
                model_replace_keys.append("body_ipos")
            if self.configs.body_mass is not None:
                model_replace_keys.append("body_mass")
            if self.configs.qpos0 is not None:
                model_replace_keys.append("qpos0")
            if self.configs.actuator_gainprm is not None:
                model_replace_keys.append("actuator_gainprm")
            if self.configs.motor_strength is not None:
                model_replace_keys.extend(["actuator_gainprm", "actuator_biasprm"])

            def rand_dynamics_single(single_rng):
                assert self.configs is not None

                res = {}
                rng_key = single_rng
                if self.configs.geom_friction is not None:
                    rng_key, key = jax.random.split(rng_key)
                    mu0 = model.geom_friction[0, 0]
                    mu = self.configs.geom_friction(mu0, key)
                    friction = model.geom_friction.at[0, 0].set(mu)
                    # Empty under floor_geom; otherwise every floor contact
                    # follows the floor's factor (see floor_friction_targets).
                    scale = mu / mu0
                    friction = friction.at[friction_geoms, 0].multiply(scale)
                    res["geom_friction"] = friction
                    if friction_pairs.size:
                        # both tangential slots: a pair stores 5 friction values
                        res["pair_friction"] = model.pair_friction.at[
                            friction_pairs, :2
                        ].multiply(scale)
                if self.configs.dof_frictionloss is not None:
                    rng_key, key = jax.random.split(rng_key)
                    res["dof_frictionloss"] = model.dof_frictionloss.at[6:].set(
                        self.configs.dof_frictionloss(
                            model.dof_frictionloss[6:], key
                        )
                    )
                if self.configs.dof_armature is not None:
                    rng_key, key = jax.random.split(rng_key)
                    res["dof_armature"] = model.dof_armature.at[6:].set(
                        self.configs.dof_armature(
                            model.dof_armature[6:], key
                        )
                    )
                if self.configs.body_ipos is not None:
                    rng_key, key = jax.random.split(rng_key)
                    res["body_ipos"] = model.body_ipos.at[1].set(
                        self.configs.body_ipos(
                            model.body_ipos[1], key
                        )
                    )
                if self.configs.body_mass is not None:
                    rng_key, key = jax.random.split(rng_key)
                    res["body_mass"] = self.configs.body_mass(
                        model.body_mass, key
                    )
                if self.configs.qpos0 is not None:
                    rng_key, key = jax.random.split(rng_key)
                    res["qpos0"] = model.qpos0.at[7:].set(
                        self.configs.qpos0(
                            model.qpos0[7:], key
                        )
                    )
                if self.configs.actuator_gainprm is not None:
                    rng_key, key = jax.random.split(rng_key)
                    res["actuator_gainprm"] = self.configs.actuator_gainprm(
                        model.actuator_gainprm, key
                    )
                if self.configs.motor_strength is not None:
                    rng_key, key = jax.random.split(rng_key)
                    # PD torque is gainprm[:,0]*ctrl + biasprm[:,1]*qpos with
                    # biasprm[:,1] == -gainprm[:,0]; scaling only the gain shifts the
                    # setpoint instead of scaling torque, so both move together.
                    kp = self.configs.motor_strength(
                        model.actuator_gainprm[:, 0], key
                    )
                    res["actuator_gainprm"] = model.actuator_gainprm.at[:, 0].set(kp)
                    res["actuator_biasprm"] = model.actuator_biasprm.at[:, 1].set(-kp)
                return res

            if getattr(rng, "ndim", 0) == 1:
                model_replace = rand_dynamics_single(rng)
            else:
                model_replace = jax.vmap(rand_dynamics_single)(rng)
                for k in model_replace_keys:
                    in_axes_replace[k] = 0
            
            in_axes = jax.tree_util.tree_map(lambda x: None, model)
            in_axes = in_axes.tree_replace(in_axes_replace)
            model = model.tree_replace(model_replace)
            
            return model, in_axes
        
        def custom_observation_randomize(obs, rng):
            if self.configs is None:
                return observation_randomize(obs, rng)

            if rng is None:
                return observation_randomize(obs, rng)

            layout = OBS_LAYOUTS[self.env_name]

            def randomize_single(single_obs, single_rng):
                # Compose on the env's own noisy observation so a custom suite is a
                # strict superset of `default`; starting from the clean state would
                # make weak YAML noise act as a de-noising instead of a perturbation.
                # Perturb in place so components outside the layout pass untouched.
                state = jnp.asarray(
                    single_obs[layout.key] if layout.key is not None else single_obs
                )
                rng_key = single_rng
                for k in _OBS_NOISE_FIELDS:
                    fn = getattr(self.configs, k, None)
                    if fn is None or k not in layout.slices:
                        continue
                    rng_key, key = jax.random.split(rng_key)
                    start, stop = layout.slices[k]
                    old = state[start:stop]
                    new = fn(old, key)
                    state = state.at[start:stop].set(new)
                    for m_start, m_stop in layout.mirrors.get(k, ()):
                        state = state.at[m_start:m_stop].add(new - old)
                return np.asarray(state)

            if getattr(rng, "ndim", 0) == 1:
                return randomize_single(obs, rng)
            else:
                if isinstance(obs, dict):
                    first_val = next(iter(obs.values()))
                    batch_size = len(first_val)
                    results = [
                        randomize_single({k: v[i] for k, v in obs.items()}, rng[i])
                        for i in range(batch_size)
                    ]
                else:
                    obs_np = np.asarray(obs)
                    results = [
                        randomize_single(obs_np[i], rng[i])
                        for i in range(len(obs_np))
                    ]
                return np.stack(results, axis=0)

        # For custom configs, `configs.action` is expected to be a factory that
        # receives an RNG key and returns an `action_fn(action)->action` for
        # each actor. We store that factory in `action_factory` so the wrapper
        # (`GymWrapper`) can create per-actor action functions during
        # `_setup_randomization` where per-actor RNG keys are available.
        def custom_action_step(action):
            # fallback to default action step when configs.action is not provided
            return action_step(action)

        return RandomizeFunctions(
            domain_randomize=custom_domain_randomize,
            observation_randomize=custom_observation_randomize,
            action_step=custom_action_step,
            observation_factory=(
                self.configs.observation_factory if self.configs is not None else None
            ),
            action_factory=(self.configs.action_factory if self.configs is not None else None),
        )

    def get_functions(self) -> RandomizeFunctions:        
        if self.type == self.RandomizeType.DISABLED:
            return self._get_disabled_functions()
        if self.type == self.RandomizeType.FULL:
            return self._get_full_functions()
        elif self.type == self.RandomizeType.ONLY_DOMAIN:
            return self._get_only_domain_functions()
        elif self.type in [self.RandomizeType.DEFAULT, self.RandomizeType.ONLY_OBSERVATION]:
            return self._get_default_functions()
        
        if self.configs is None:
            raise ValueError("configs should not be None when type is CUSTOM")
        return self._get_custom_functions()

def get_predefined_randomize_configs(
    randomize_type: str,
    options: Optional[dict[str, Any] | str] = None,
    env_name: str = ENV_NAME
) -> RandomizeConfigs:
    """Return a RandomizeConfigs object for the given randomization type."""

    def get_fn(
        dist: Literal["uniform", "normal"],
        type: Literal["additive", "scale", "absolute", "additive_total"],
        range: Tuple[float, float],
        indices: Optional[np.ndarray] = None
    ) -> Callable[[jax.Array, jax.Array], jax.Array]:
        low, high = range

        if dist == "uniform":
            def sample(current, key):
                return jax.random.uniform(
                    key, shape=jnp.shape(current), minval=low, maxval=high
                )
        else:
            mean = (low + high) / 2.0
            std = (high - low) / 2.0
            def sample(current, key):
                value = jax.random.normal(key, shape=jnp.shape(current)) * std + mean
                return jnp.clip(value, low, high)

        def fn(current, key):
            value = sample(current, key)
            if type == "additive":
                out = current + value
            elif type == "additive_total":
                # value is a fraction of the summed array: for body_mass, a payload
                # expressed as a share of the robot's total mass.
                out = current + value * jnp.sum(current)
            elif type == "scale":
                out = current * value
            else:
                out = value
            if indices is not None:
                mask = jnp.zeros_like(current).at[indices].set(1.0)
                out = jnp.where(mask > 0, out, current)
            return out

        return fn

    def get_action_factory(
        type: Literal["randomized_delay", "fixed_delay", "smoothing"],
        configs: Optional[Any] = None
    ) -> Callable[[Any, Any], Callable[[Any], Any]]:
        if type == "randomized_delay":
            if configs is None or "max_delay" not in configs:
                raise ValueError("configs must contain 'max_delay' for randomized delay action factory")
            
            max_delay = configs["max_delay"]
            if not isinstance(max_delay, int) or max_delay < 0:
                raise ValueError("'max_delay' must be a non-negative integer for randomized delay action factory")
            
            def action_factory_fn(action_size, key):
                # sample delay in [0, 2) and convert to Python int (0 or 1 step, i.e., 0 or 20 ms delay)
                delay = int(np.asarray(jax.random.uniform(key, shape=(), minval=0, maxval=max_delay + 1)))

                if delay <= 0:
                    def act_identity(action):
                        return jnp.asarray(action, dtype=jnp.float32)
                    return act_identity

                first_action = np.zeros(action_size, dtype=np.float32)
                buf = deque([first_action] * delay, maxlen=delay)

                def act(action):
                    out = jnp.array(buf[0], dtype=jnp.float32)
                    buf.append(np.asarray(action))
                    return out

                return act
        elif type == "fixed_delay":
            if configs is None or "delay" not in configs:
                raise ValueError("configs must contain 'delay' for fixed delay action factory")
            
            delay = configs["delay"]
            if not isinstance(delay, int) or delay < 0:
                raise ValueError("'delay' must be a non-negative integer for fixed delay action factory")

            def action_factory_fn(action_size, key):
                if delay == 0:
                    def act_identity(action):
                        return jnp.asarray(action, dtype=jnp.float32)
                    return act_identity

                first_action = np.zeros(action_size, dtype=np.float32)
                buf = deque([first_action] * delay, maxlen=delay)

                def act(action):
                    out = jnp.array(buf[0], dtype=jnp.float32)
                    buf.append(np.asarray(action))
                    return out

                return act
        elif type == "smoothing":
            if configs is None or "alpha" not in configs:
                raise ValueError("configs must contain 'alpha' for smoothing action factory")
            
            alpha = configs["alpha"]
            if not isinstance(alpha, float) or not (0.0 < alpha < 1.0):
                raise ValueError("'alpha' must be a float in (0, 1) for smoothing action factory")

            def action_factory_fn(action_size, key):
                smoothed_action = np.zeros(action_size, dtype=np.float32)

                def act(action):
                    nonlocal smoothed_action
                    smoothed_action = alpha * np.asarray(action) + (1 - alpha) * smoothed_action
                    return jnp.array(smoothed_action, dtype=jnp.float32)

                return act

        return action_factory_fn

    if randomize_type == "disabled":
        return RandomizeConfigs(type=RandomizeConfigs.RandomizeType.DISABLED, env_name=env_name)
    elif randomize_type == "full":
        return RandomizeConfigs(type=RandomizeConfigs.RandomizeType.FULL, env_name=env_name)
    elif randomize_type == "only_domain":
        return RandomizeConfigs(type=RandomizeConfigs.RandomizeType.ONLY_DOMAIN, env_name=env_name)
    elif randomize_type in ["only_observation", "default"]:
        return RandomizeConfigs(type=RandomizeConfigs.RandomizeType.DEFAULT, env_name=env_name)
    elif randomize_type == "custom":
        opt: dict[str, Any]

        if isinstance(options, str):
            if os.path.exists(options):
                with open(options, "r") as f:
                    opt = yaml.safe_load(f)
            else:
                opt = yaml.safe_load(options)
        elif isinstance(options, dict):
            opt = options
        else:
            raise ValueError("options must be provided for 'custom' randomization type")

        obs_fields = ["gyro", "gravity", "joint_angles", "joint_vel", "linvel"]
        domain_fields = ["geom_friction", "dof_frictionloss", "dof_armature", "body_ipos", "body_mass", "qpos0", "actuator_gainprm", "motor_strength"]
        noise_fields = obs_fields + domain_fields
        functions = {}

        if opt.get("noise") is not None:
            noise = opt["noise"]

            def get_noise_fn(component):
                if component.get("dist") is None:
                    raise ValueError(f"Missing 'dist' for component {component}")
                elif component["dist"] not in ["uniform", "normal"]:
                    raise ValueError(f"Unknown 'dist' for component {component}: {component['dist']}")
                
                if component.get("type") is None:
                    raise ValueError(f"Missing 'type' for component {component}")
                elif component["type"] not in ["additive", "scale", "absolute", "additive_total"]:
                    raise ValueError(f"Unknown 'type' for component {component}: {component['type']}")
                
                if component.get("min") is None:
                    raise ValueError(f"Missing 'min' for component {component}")
                
                if component.get("max") is None:
                    raise ValueError(f"Missing 'max' for component {component}")
                elif component["min"] >= component["max"]:
                    raise ValueError(f"'min' must be less than 'max' for component {component}: min={component['min']}, max={component['max']}")

                indices = None
                if component.get("indices") is not None:
                    indices = np.asarray(component["indices"])
                    if not np.issubdtype(indices.dtype, np.integer):
                        raise ValueError(f"'indices' must be an array of integers for component {component}")

                return get_fn(
                    component["dist"], 
                    component["type"], 
                    (component["min"], component["max"]),
                    indices
                )

            if not isinstance(noise, dict):
                raise ValueError(f"Unknown 'noise' type for component {noise}: {type(noise)}")

            for key, cfg in noise.items():
                if not isinstance(cfg, dict):
                    raise ValueError(f"Unknown 'noise' config type for component {key}: {type(cfg)}")
                if key not in noise_fields:
                    raise ValueError(f"Unknown component for noise randomization: {key}")
                if cfg.get("type") == "additive_total" and key != "body_mass":
                    raise ValueError(f"'additive_total' is only defined for body_mass, got {key}")
                if "target" in cfg:
                    if key != "geom_friction":
                        raise ValueError(f"'target' is only defined for geom_friction, got {key}")
                    if cfg["target"] not in FRICTION_TARGETS:
                        raise ValueError(f"Unknown geom_friction 'target': {cfg['target']} (expected one of {FRICTION_TARGETS})")
                    functions["friction_target"] = cfg["target"]

                functions[key] = get_noise_fn(cfg)

            if "motor_strength" in functions and "actuator_gainprm" in functions:
                raise ValueError(
                    "'motor_strength' and 'actuator_gainprm' both write actuator_gainprm; use only one"
                )
        
        if opt.get("bias") is not None:
            bias = opt["bias"]

            if not isinstance(bias, dict):
                raise ValueError(f"Unknown 'bias' type: {type(bias)}")
        
            if bias.get("dist") is None:
                raise ValueError(f"Missing 'dist' for component {bias}")
            elif bias["dist"] not in ["uniform", "normal"]:
                raise ValueError(f"Unknown 'dist' for component {bias}: {bias['dist']}")
            
            if bias.get("min") is None:
                raise ValueError(f"Missing 'min' for component {bias}")
            
            if bias.get("max") is None:
                raise ValueError(f"Missing 'max' for component {bias}")
            elif bias["min"] >= bias["max"]:
                raise ValueError(f"'min' must be less than 'max' for component {bias}: min={bias['min']}, max={bias['max']}")

            if bias["dist"] == "uniform":
                dist = lambda rng, shape: jax.random.uniform(rng, shape=shape, minval=bias["min"], maxval=bias["max"])
            else:
                mean = (bias["min"] + bias["max"]) / 2.0
                std = (bias["max"] - bias["min"]) / 2.0
                dist = lambda rng, shape: jax.random.normal(rng, shape=shape) * std + mean

            def get_bias_fn(obs_space, rng):
                bias_value = dist(rng, shape=obs_space.shape)
                
                def bias_fn(obs, key):
                    return obs + bias_value
                
                return bias_fn
            
            functions["observation_factory"] = get_bias_fn

        if opt.get("quantization") is not None:
            value = opt["quantization"]

            if not isinstance(value, int):
                raise ValueError(f"Unknown 'quantization' type: {type(value)}")
            
            if value <= 0:
                raise ValueError(f"'quantization' must be a positive integer, got {value}")

            for field in obs_fields:
                if field not in functions:
                    functions[field] = lambda x, key, val=value: jnp.round(x * val) / val
                else:
                    prev_fn = functions[field]
                    functions[field] = lambda x, key, f=prev_fn, val=value: jnp.round(f(x, key) * val) / val

        if opt.get("action_factory") is not None:
            value = opt["action_factory"]

            if not isinstance(value, dict):
                raise ValueError(f"Unknown 'action_factory' type: {type(value)}")

            if value.get("type") is None:
                raise ValueError(f"Missing 'type' for action_factory: {value}")
            elif value["type"] not in ["randomized_delay", "fixed_delay", "smoothing"]:
                raise ValueError(f"Unknown 'type' for action_factory: {value['type']}")

            functions["action_factory"] = get_action_factory(value["type"], value)

        cfg = RandomizeConfigs.RandomizeOptions()
        for key, fn in functions.items():
            setattr(cfg, key, fn)

        return RandomizeConfigs(
            type=RandomizeConfigs.RandomizeType.CUSTOM,
            configs=cfg,
            env_name=env_name
        )
    else:
        raise ValueError(f"Unknown randomize_type: {randomize_type}")

# `disabled` and `only_domain` rebuild the observation from the *clean* readings in
# privileged_state, which `_extract_observation_components` reads at the Go2 joystick
# offsets. Getup (42/91), Handstand/Footstand (45/94) and GetupWalk (48/97) reorder it
# -- GetupWalk keeps state at 48, so identify the layout by both sizes -- and H1 has
# no privileged_state at all. `custom` only perturbs the noisy state, through
# OBS_LAYOUTS, and works on every env listed there.
_SUPPORTED_OBS_LAYOUT = {"state": 48, "privileged_state": 123}


def _assert_supported_obs_layout(env, randomize_configs, env_name):
    """Raise unless ``env``'s observation fits the randomize type's assumptions.

    ``default``/``full`` forward the observation untouched and work on any env.
    """
    sizes = env.observation_size
    if isinstance(sizes, dict):
        sizes = {
            k: (v[0] if isinstance(v, tuple) else v) for k, v in sizes.items()
        }

    if randomize_configs.type == RandomizeConfigs.RandomizeType.CUSTOM:
        layout = OBS_LAYOUTS.get(env_name)
        if layout is None:
            raise ValueError(
                f"{env_name} has no observation layout in OBS_LAYOUTS; add one "
                "(and check it with scripts/check_randomize_envs.py) before using "
                "a custom randomization suite."
            )
        actual = sizes.get(layout.key) if layout.key is not None else sizes
        if isinstance(actual, dict) or actual != layout.size:
            raise ValueError(
                f"{env_name} observation {sizes} does not match its OBS_LAYOUTS "
                f"entry ({layout.key or 'flat'}: {layout.size}); the env changed."
            )
        return

    clean_split_types = {
        RandomizeConfigs.RandomizeType.DISABLED,
        RandomizeConfigs.RandomizeType.ONLY_DOMAIN,
    }
    if randomize_configs.type in clean_split_types and sizes != _SUPPORTED_OBS_LAYOUT:
        raise ValueError(
            f"{env_name} has observation layout {sizes}, but randomize type "
            f"'{randomize_configs.type.value}' reads clean readings at the Go2 "
            f"joystick offsets {_SUPPORTED_OBS_LAYOUT}."
        )


def get_env(
    device: str,
    render_callback=None,
    command_type=None,
    num_actors: int = 1,
    dataset=None,
    config_overrides: Optional[Dict[str, str | int | list[Any]]] = None,
    randomize_configs: Optional[RandomizeConfigs | str] = None,
    randomize_options: Optional[dict[str, Any] | str] = None,
    env_name: str = ENV_NAME
):
    config_overrides = {
        "impl": "jax", 
        **config_overrides
    } if config_overrides is not None else {"impl": "jax"}
    env = registry.load(env_name, config_overrides=config_overrides)
    env_cfg = registry.get_default_config(env_name)

    options = None
    if randomize_options is not None and isinstance(randomize_options, dict):
        options = randomize_options
    elif randomize_options is not None and isinstance(randomize_options, str):
        if os.path.exists(randomize_options):
            with open(randomize_options, "r") as f:
                options = yaml.safe_load(f)
        else:
            options = yaml.safe_load(randomize_options)

    if randomize_configs is None:
        if options is None:
            randomize_configs = RandomizeConfigs(
                type=RandomizeConfigs.RandomizeType.DEFAULT,
                env_name=env_name
            )
        else:
            randomize_configs = get_predefined_randomize_configs(
                "custom", options, env_name
            )
    elif isinstance(randomize_configs, str):
        randomize_configs = get_predefined_randomize_configs(
            randomize_configs, options, env_name
        )

    randomize_functions = randomize_configs.get_functions()

    _assert_supported_obs_layout(env, randomize_configs, env_name)

    env = GymWrapper(
        env,
        env_cfg,
        seed=1,
        num_actors=num_actors,
        device=device,
        command_type=command_type,
        render_callback=render_callback,
        randomize_functions=randomize_functions,
        dataset=dataset,
    )

    return env


def maybe_get_shifted_env(
    device,
    command_type=None,
    dataset=None,
    eval_shift=None,
    num_actors=1,
    env_name: str = ENV_NAME
):
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
        device,
        command_type=command_type,
        dataset=dataset,
        config_overrides=overrides,
        num_actors=num_actors,
        env_name=env_name
    )


class GymWrapper(gym.Env):
    def __init__(
        self,
        env,
        env_cfg,
        seed,
        randomize_functions: RandomizeFunctions,
        num_actors=1,
        device="cpu",
        command_type=None,
        render_callback=None,
        dataset=None,
    ):
        super().__init__()
        self.command_type = command_type
        self._cmd_slice = None
        self.env = env
        self.device = device
        self.rng = jax.random.PRNGKey(seed)
        self.render_callback = render_callback
        self.episode_length = env_cfg.episode_length

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
            self.observation_space = NumpySpace(shape=obs_size, dtype=jnp.float32)
        else:
            self.observation_space = NumpySpace(shape=(obs_size,), dtype=jnp.float32)
        self.action_space = NumpySpace(shape=(self.env.action_size,), dtype=jnp.float32)

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

        self._randomize_functions = randomize_functions
        self._need_reset = True
        self._randomize_every_episode = randomize_functions.randomize_every_episode
        
        self._setup_randomization(num_actors)

    def update_randomize_functions(self, randomize_functions: RandomizeFunctions):
        """Update the randomization functions and re-setup the randomization."""
        self._randomize_functions = randomize_functions
        self._need_reset = True
        self._randomize_every_episode = randomize_functions.randomize_every_episode
        self._setup_randomization(self.num_envs)

    def warmup_jit_reset(self):
        """Trigger JAX JIT compilation for reset/step functions.
        
        Call this after update_randomize_functions() to trigger compilation
        in a controlled context, avoiding slow first reset during evaluation.
        This performs a dummy reset/step cycle to compile the vmapped functions.
        """
        try:
            self.reset()
        except Exception:
            # Silently ignore any errors during warmup
            pass

    def _setup_randomization(self, num_actors):
        """Build JIT-compiled vmapped reset/step functions with randomization.

        At each reset the physics parameters (mass, friction, armature, etc.) are
        re-sampled so every evaluation episode runs in a different simulated world.
        The JIT cache is reused across episodes because the structure (in_axes) of
        the randomized model never changes, only its values do.
        """
        # Define functions
        self.domain_randomize_fn = self._randomize_functions.domain_randomize
        self.observation_randomize_fn = self._randomize_functions.observation_randomize
        
        # Only initialize the base model on first setup to avoid unnecessary recreation
        if not hasattr(self, '_base_mjx_model') or self._base_mjx_model is None:
            self._base_mjx_model = self.env.mjx_model

        # Compute the initial randomized model batch to determine in_axes.
        init_keys = jax.random.split(self.rng, num_actors)
        self._mjx_model_v, self._in_axes = self.domain_randomize_fn(
            self._base_mjx_model, init_keys
        )

        # If an action_factory is provided, build per-actor action functions
        # from per-actor RNG keys and create a batched action function that
        # applies each per-actor function to the corresponding action.
        if getattr(self._randomize_functions, "action_factory", None) is not None:
            self.action_fn = self._build_batched_action_fn(init_keys)
        else:
            self.action_fn = self._randomize_functions.action_step

        if getattr(self._randomize_functions, "observation_factory", None) is not None:
            self.observation_fn = self._build_batched_observation_fn(init_keys)
        else:
            self.observation_fn = None

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

    def _build_batched_action_fn(self, actor_keys):
        action_size = self.env.action_size
        self._action_fns = [
            self._randomize_functions.action_factory(action_size, key)  # type: ignore
            for key in actor_keys
        ]

        def batched_action_fn(actions):
            # `actions` is expected shape (num_envs, action_size). We apply
            # per-actor Python action functions and return a jnp array.
            actions_np = np.asarray(actions)
            outs = []
            for index, fn in enumerate(self._action_fns):
                out = fn(actions_np[index])
                outs.append(np.asarray(out))
            return jnp.asarray(np.stack(outs, axis=0))

        return batched_action_fn

    def _build_batched_observation_fn(self, actor_keys):
        observation_space = self.observation_space
        self._observation_fns = [
            self._randomize_functions.observation_factory(observation_space, key)  # type: ignore
            for key in actor_keys
        ]

        def batched_observation_fn(obs):
            obs_np = np.asarray(obs)
            if obs_np.ndim == 1:
                return jnp.asarray(self._observation_fns[0](obs_np, None))

            outs = []
            for index, fn in enumerate(self._observation_fns):
                out = fn(obs_np[index], None)
                outs.append(np.asarray(out))
            return jnp.asarray(np.stack(outs, axis=0))

        return batched_observation_fn

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
        obs = self._maybe_unfreeze(env_state.obs)

        info = self._maybe_unfreeze(env_state.info)
        if "rng" in info:
            rng = info["rng"]
            if getattr(rng, "ndim", 0) == 1:
                next_rng, obs_rng = jax.random.split(rng)
            else:
                split_rngs = jax.vmap(lambda key: jax.random.split(key, 2))(rng)
                next_rng, obs_rng = split_rngs[:, 0], split_rngs[:, 1]
            info["rng"] = next_rng
        else:
            obs_rng = None

        state_obs = jnp.asarray(self.observation_randomize_fn(obs, obs_rng))

        observation_fn = getattr(self, "observation_fn", None)
        if observation_fn is not None:
            state_obs = jnp.asarray(observation_fn(state_obs))

        if self.command_type is not None and "command" in info:
            commands = info["command"]
            zeros = jnp.zeros_like(commands)

            if self.command_type == "forwardbackward":
                command = zeros.at[..., 0].set(commands[..., 0])
            elif self.command_type == "forward":
                command = zeros.at[..., 0].set(jnp.abs(commands[..., 0]))
            elif self.command_type == "forwardfixed":
                command = zeros.at[..., 0].set(1.0)
            elif self.command_type == "forward_realrobot":
                command = zeros.at[..., 0].set(0.2)
            else:
                command = None

            if command is not None:
                if self._cmd_slice is None:
                    self._cmd_slice = locate_command_slice(
                        state_obs[0] if state_obs.ndim > 1 else state_obs,
                        commands[0] if commands.ndim > 1 else commands,
                    )
                start, stop = self._cmd_slice
                state_obs = state_obs.at[..., start:stop].set(command)
                info["command"] = command

        return env_state.replace(obs=state_obs, info=info)

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
        self._need_reset = False
        self.rng, reset_rng = jax.random.split(self.rng)
        reset_keys = jax.random.split(reset_rng, self.num_envs)

        if getattr(self._randomize_functions, "action_factory", None) is not None:
            self.action_fn = self._build_batched_action_fn(reset_keys)
        if getattr(self._randomize_functions, "observation_factory", None) is not None:
            self.observation_fn = self._build_batched_observation_fn(reset_keys)

        if self.domain_randomize_fn is not None:
            if self._randomize_every_episode:
                self.rng, dr_rng = jax.random.split(self.rng)
                dr_keys = jax.random.split(dr_rng, self.num_envs)
                self._mjx_model_v, _ = self.domain_randomize_fn(
                    self._base_mjx_model, dr_keys
                )
            self.env_state = self._reset_fn(self._mjx_model_v, reset_keys)
        else:
            self.env_state = self._reset_fn(reset_keys)

        self.env_state = self._apply_command_override(self.env_state)
        self.timesteps = 0
        obs_field = self.env_state.obs
        if isinstance(obs_field, (dict, Mapping)):
            obs = np.asarray(obs_field["state"])
        else:
            obs = np.asarray(obs_field)
        return obs, {}

    def step(self, action):
        if self._need_reset:
            raise RuntimeError(
                "Environment must be reset before stepping. Call `reset()` first."
            )
        
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = jnp.asarray(action)
        if len(action.shape) == 1:
            action = action[None, ...]

        action = self.action_fn(action)

        if self.domain_randomize_fn is not None:
            self.env_state = self._step_fn(self._mjx_model_v, self.env_state, action)
        else:
            self.env_state = self._step_fn(self.env_state, action)
        self.env_state = self._apply_command_override(self.env_state)
        self.timesteps += 1
        obs_field = self.env_state.obs
        if isinstance(obs_field, (dict, Mapping)):
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
        device,
        render_callback=render_callback,
        command_type=command_type,
        env_name=env_name,
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