from dataclasses import dataclass
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
        dof_frictionloss: Optional[Callable[[Any, Any], jax.Array]] = None
        dof_armature: Optional[Callable[[Any, Any], jax.Array]] = None
        body_ipos: Optional[Callable[[Any, Any], jax.Array]] = None
        body_mass: Optional[Callable[[Any, Any], jax.Array]] = None
        qpos0: Optional[Callable[[Any, Any], jax.Array]] = None
        actuator_gainprm: Optional[Callable[[Any, Any], jax.Array]] = None
        
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
    
    def __post_init__(self):
        if self.type != self.RandomizeType.CUSTOM and self.configs is not None:
            raise ValueError("configs should be None when type is not CUSTOM")
        
        if self.type == self.RandomizeType.CUSTOM and self.configs is None:
            raise ValueError("configs should not be None when type is CUSTOM")
    
    def _get_domain_randomize(
        self, enabled=True
    ) -> Callable[[Any, Any], Tuple[Any, Any]]:
        if enabled:
            return registry.get_domain_randomizer(ENV_NAME) # type: ignore
        
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
        observation_randomize = self._get_observation_randomize(enabled=False)
        action_step = self._get_action_function()
        
        def custom_domain_randomize(model, rng):
            if self.configs is None:
                return domain_randomize(model, rng)
            
            in_axes_replace = dict()
            model_replace_keys = []
            
            if self.configs.geom_friction is not None:
                model_replace_keys.append("geom_friction")
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

            def rand_dynamics_single(single_rng):
                assert self.configs is not None

                res = {}
                rng_key = single_rng
                if self.configs.geom_friction is not None:
                    rng_key, key = jax.random.split(rng_key)
                    res["geom_friction"] = model.geom_friction.at[0, 0].set(
                        self.configs.geom_friction(
                            model.geom_friction[0, 0], key
                        )
                    )
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

            def randomize_single(single_obs, single_rng):
                clean_dict, _ = _extract_observation_components(single_obs)
                comp_dict = dict(clean_dict)
                
                rng_key = single_rng
                for k in ["linvel", "gyro", "gravity", "joint_angles", "joint_vel"]:
                    fn = getattr(self.configs, k, None)
                    if fn is not None:
                        rng_key, key = jax.random.split(rng_key)
                        comp_dict[k] = fn(clean_dict[k], key)
                return _concat_observation_dict(comp_dict)

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
    options: Optional[dict[str, Any] | str] = None
) -> RandomizeConfigs:
    """Return a RandomizeConfigs object for the given randomization type."""

    def get_fn(
        dist: Literal["uniform", "normal"],
        type: Literal["additive", "scale"],
        range: Tuple[float, float],
        indices: Optional[np.ndarray] = None
    ) -> Callable[[jax.Array, jax.Array], jax.Array]:
        if dist == "uniform":
            if type == "additive":
                def fn(current, key):
                    noise = jax.random.uniform(
                        key, 
                        shape=current.shape, 
                        minval=range[0], 
                        maxval=range[1]
                    )
                    if indices is not None:
                        mask = jnp.zeros_like(noise).at[indices].set(1.0)
                        noise = noise * mask
                    return current + noise
            else:
                def fn(current, key):
                    scale = jax.random.uniform(
                        key, 
                        shape=current.shape, 
                        minval=range[0], 
                        maxval=range[1]
                    )
                    if indices is not None:
                        mask = jnp.zeros_like(scale).at[indices].set(1.0)
                        scale = scale * mask + (1.0 - mask)
                    return current * scale
        else:
            mean = (range[0] + range[1]) / 2.0
            std = (range[1] - range[0]) / 2.0
            if type == "additive":
                def fn(current, key):
                    noise = jax.random.normal(
                        key, shape=current.shape
                    ) * std + mean
                    noise = jnp.clip(noise, range[0], range[1])
                    if indices is not None:
                        mask = jnp.zeros_like(noise).at[indices].set(1.0)
                        noise = noise * mask
                    return current + noise
            else:
                def fn(current, key):
                    scale = jax.random.normal(
                        key, 
                        shape=current.shape
                    ) * std + mean
                    scale = jnp.clip(scale, range[0], range[1])
                    if indices is not None:
                        mask = jnp.zeros_like(scale).at[indices].set(1.0)
                        scale = scale * mask + (1.0 - mask)
                    return current * scale
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
        return RandomizeConfigs(type=RandomizeConfigs.RandomizeType.DISABLED)
    elif randomize_type == "full":
        return RandomizeConfigs(type=RandomizeConfigs.RandomizeType.FULL)
    elif randomize_type == "only_domain":
        return RandomizeConfigs(type=RandomizeConfigs.RandomizeType.ONLY_DOMAIN)
    elif randomize_type in ["only_observation", "default"]:
        return RandomizeConfigs(type=RandomizeConfigs.RandomizeType.DEFAULT)
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
        domain_fields = ["geom_friction", "dof_frictionloss", "dof_armature", "body_ipos", "body_mass", "qpos0", "actuator_gainprm"]
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
                elif component["type"] not in ["additive", "scale"]:
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
                
                functions[key] = get_noise_fn(cfg)
        
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
            configs=cfg
        )
    else:
        raise ValueError(f"Unknown randomize_type: {randomize_type}")

def get_env(
    device: str,
    render_callback=None,
    command_type=None,
    num_actors: int = 1,
    dataset=None,
    config_overrides: Optional[Dict[str, str | int | list[Any]]] = None,
    randomize_configs: Optional[RandomizeConfigs | str] = None,
    randomize_options: Optional[dict[str, Any] | str] = None
):
    config_overrides = {
        "impl": "jax", 
        **config_overrides
    } if config_overrides is not None else {"impl": "jax"}
    env = registry.load(ENV_NAME, config_overrides=config_overrides)
    env_cfg = registry.get_default_config(ENV_NAME)

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
                type=RandomizeConfigs.RandomizeType.DEFAULT
            )
        else:
            randomize_configs = get_predefined_randomize_configs("custom", options)
    elif isinstance(randomize_configs, str):
        randomize_configs = get_predefined_randomize_configs(randomize_configs, options)

    randomize_functions = randomize_configs.get_functions()

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


def maybe_get_shifted_env(device, command_type=None, dataset=None, eval_shift=None, num_actors=1):
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
        num_actors=num_actors
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