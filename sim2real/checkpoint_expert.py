import argparse
import abc
import dataclasses
import enum
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Optional, Protocol, Sequence, Tuple, Union

import numpy as np

import jax
import jax.numpy as jnp
import flax
from flax import linen, struct
from etils import epath
from orbax import checkpoint as ocp

from portable_actor import PortableActor, load_actor, ordered_dense_from_flax, save_actor

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

STATE_DIM = 48
ACTION_DIM = 12
MAX_ACTION = 1.0

DEFAULT_NETWORK_CONFIG = {
    "policy_hidden_layer_sizes": (512, 256, 128),
    "policy_obs_key": "state",
    "value_hidden_layer_sizes": (512, 256, 128),
    "value_obs_key": "state",
}

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ObservationSize = Union[int, Mapping[str, Union[Tuple[int, ...], int]]]
Observation = Union[jnp.ndarray, Mapping[str, jnp.ndarray]]
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]
Params = Any
PRNGKey = jnp.ndarray
Action = jnp.ndarray
Extra = Mapping[str, Any]
PreprocessorParams = Any


# ---------------------------------------------------------------------------
# UInt64 (JAX x64-safe counter)
# ---------------------------------------------------------------------------


@flax.struct.dataclass
class UInt64:
    hi: Union[int, np.ndarray, jax.Array]
    lo: Union[int, np.ndarray, jax.Array]

    def to_numpy(self):
        hi_np = np.array(self.hi, dtype=np.uint64)
        lo_np = np.array(self.lo, dtype=np.uint64)
        return (hi_np << np.uint64(32)) | lo_np

    def __post_init__(self):
        if isinstance(self.hi, (int, np.integer, np.ndarray, jax.Array)):
            object.__setattr__(self, "hi", jnp.uint32(self.hi))
        if isinstance(self.lo, (int, np.integer, np.ndarray, jax.Array)):
            object.__setattr__(self, "lo", jnp.uint32(self.lo))

    def __add__(self, other):
        other = _sanitize_uint64_input(other)
        return _add_uint64(self, other)

    def __repr__(self):
        return f"UInt64(hi={self.hi}, lo={self.lo})"

    def __int__(self):
        return int(self.to_numpy())


def _sanitize_uint64_input(other):
    if isinstance(other, (int, np.ndarray, jax.Array)):
        other_lo = other & jnp.array(0xFFFFFFFF, dtype=jnp.uint32)
        other_hi = other >> 32
        return UInt64(
            hi=jnp.array(other_hi, dtype=jnp.uint32),
            lo=jnp.array(other_lo, dtype=jnp.uint32),
        )
    elif isinstance(other, UInt64):
        return other
    raise NotImplementedError(f"Cannot perform op on {type(other)} and UInt64.")


def _add_uint64(a: UInt64, b: UInt64) -> UInt64:
    result_lo = a.lo + b.lo
    carry_bit_pattern = (a.lo & b.lo) | ((a.lo | b.lo) & (~result_lo))
    carry = carry_bit_pattern >> 31
    result_hi = a.hi + b.hi + carry
    return UInt64(hi=result_hi, lo=result_lo)


# ---------------------------------------------------------------------------
# Running statistics
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Array:
    shape: Tuple[int, ...]
    dtype: jnp.dtype


NestedArray = jnp.ndarray
NestedTensor = Any
NestedSpec = Union[Array, Iterable["NestedSpec"], Mapping[Any, "NestedSpec"]]
Nest = Union[NestedArray, NestedTensor, NestedSpec]


@struct.dataclass
class NestedMeanStd:
    mean: Nest
    std: Nest


class NormalizationMode(enum.IntEnum):
    WELFORD = 0
    EMA = 1


@struct.dataclass
class RunningStatisticsState(NestedMeanStd):
    count: Union[jnp.ndarray, UInt64]
    summed_variance: Nest
    std_eps: float = 0.0
    mode: int = struct.field(pytree_node=False, default=NormalizationMode.WELFORD)


def normalize(batch, mean_std, max_abs_value=None):
    def normalize_leaf(data, mean, std):
        if not jnp.issubdtype(data.dtype, jnp.inexact):
            return data
        data = (data - mean) / std
        if max_abs_value is not None:
            data = jnp.clip(data, -max_abs_value, max_abs_value)
        return data

    return jax.tree_util.tree_map(normalize_leaf, batch, mean_std.mean, mean_std.std)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def load_orbax_checkpoint(path):
    path = epath.Path(path)
    if not path.exists():
        raise ValueError(f"Checkpoint path does not exist: {path.as_posix()}")

    metadata = ocp.PyTreeCheckpointer().metadata(path).item_metadata
    restore_args = jax.tree.map(
        lambda _: ocp.RestoreArgs(restore_type=np.ndarray), metadata
    )
    target = ocp.PyTreeCheckpointer().restore(
        path, ocp.args.PyTreeRestore(restore_args=restore_args), item=None
    )

    state_dict = target[0]
    if isinstance(state_dict["count"], dict) and "hi" in state_dict["count"]:
        state_dict["count"] = UInt64(**state_dict["count"])
    target[0] = RunningStatisticsState(**state_dict)
    return target


def find_checkpoint(checkpoints_dir: str, step: Optional[int] = None) -> epath.Path:
    """Return the checkpoint directory for a given step, or the latest one.

    Args:
        checkpoints_dir: Directory containing numbered checkpoint sub-folders.
        step: Specific step number to load. Pass ``None`` (default) to load
              the checkpoint with the highest step number.

    Raises:
        FileNotFoundError: If no checkpoints are found or the requested step
            does not exist.
    """
    ckpt_path = epath.Path(checkpoints_dir).resolve()
    ckpts = [c for c in ckpt_path.glob("*") if c.is_dir() and c.name.isdigit()]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint directories found in {ckpt_path}")
    ckpts.sort(key=lambda x: int(x.name))

    if step is None:
        return ckpts[-1]

    matches = [c for c in ckpts if int(c.name) == step]
    if not matches:
        available = [int(c.name) for c in ckpts]
        raise FileNotFoundError(
            f"Checkpoint step {step} not found in {ckpt_path}. "
            f"Available steps: {available}"
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


class ParametricDistribution(abc.ABC):
    def __init__(self, param_size, postprocessor, event_ndims, reparametrizable):
        self._param_size = param_size
        self._postprocessor = postprocessor
        self._event_ndims = event_ndims
        self._reparametrizable = reparametrizable
        assert event_ndims in [0, 1]

    @abc.abstractmethod
    def create_dist(self, parameters):
        pass

    @property
    def param_size(self):
        return self._param_size

    @property
    def reparametrizable(self):
        return self._reparametrizable

    def postprocess(self, event):
        return self._postprocessor.forward(event)

    def inverse_postprocess(self, event):
        return self._postprocessor.inverse(event)

    def sample_no_postprocessing(self, parameters, seed):
        return self.create_dist(parameters).sample(seed=seed)

    def sample(self, parameters, seed):
        return self.postprocess(self.sample_no_postprocessing(parameters, seed))

    def mode(self, parameters):
        return self.postprocess(self.create_dist(parameters).mode())

    def log_prob(self, parameters, actions):
        dist = self.create_dist(parameters)
        log_probs = dist.log_prob(actions)
        log_probs -= self._postprocessor.forward_log_det_jacobian(actions)
        if self._event_ndims == 1:
            log_probs = jnp.sum(log_probs, axis=-1)
        return log_probs

    def entropy(self, parameters, seed):
        dist = self.create_dist(parameters)
        entropy = dist.entropy()
        entropy += self._postprocessor.forward_log_det_jacobian(dist.sample(seed=seed))
        if self._event_ndims == 1:
            entropy = jnp.sum(entropy, axis=-1)
        return entropy


class _NormalDistribution:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def sample(self, seed):
        return jax.random.normal(seed, shape=self.loc.shape) * self.scale + self.loc

    def mode(self):
        return self.loc

    def log_prob(self, x):
        log_unnormalized = -0.5 * jnp.square(x / self.scale - self.loc / self.scale)
        log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale)
        return log_unnormalized - log_normalization

    def entropy(self):
        log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale)
        return (0.5 + log_normalization) * jnp.ones_like(self.loc)

    def kl_divergence(self, old_dist: "_NormalDistribution"):
        return jnp.sum(
            jnp.log(self.scale / old_dist.scale + 1e-5)
            + (jnp.square(old_dist.scale) + jnp.square(old_dist.loc - self.loc))
            / (2.0 * jnp.square(self.scale))
            - 0.5,
            axis=-1,
        )


class TanhBijector:
    def forward(self, x):
        return jnp.tanh(x)

    def inverse(self, y):
        return jnp.arctanh(y)

    def forward_log_det_jacobian(self, x):
        return 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))


class IdentityPostprocessor:
    def forward(self, x):
        return x

    def inverse(self, x):
        return x

    def forward_log_det_jacobian(self, x):
        return jnp.zeros_like(x)


class NormalTanhDistribution(ParametricDistribution):
    def __init__(self, event_size, min_std=0.001, var_scale=1):
        super().__init__(
            param_size=2 * event_size,
            postprocessor=TanhBijector(),
            event_ndims=1,
            reparametrizable=True,
        )
        self._min_std = min_std
        self._var_scale = var_scale

    def create_dist(self, parameters):
        loc, scale = jnp.split(parameters, 2, axis=-1)
        scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale
        return _NormalDistribution(loc=loc, scale=scale)


class NormalDistribution(ParametricDistribution):
    def __init__(self, event_size: int) -> None:
        super().__init__(
            param_size=event_size,
            postprocessor=IdentityPostprocessor(),
            event_ndims=1,
            reparametrizable=True,
        )

    def create_dist(self, parameters):
        return _NormalDistribution(*parameters)


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------


class PreprocessObservationFn(Protocol):
    def __call__(
        self, observation: Observation, preprocessor_params: PreprocessorParams
    ) -> jnp.ndarray:
        ...


def identity_observation_preprocessor(
    observation: Observation, preprocessor_params: PreprocessorParams
):
    del preprocessor_params
    return observation


class MLP(linen.Module):
    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    layer_norm: bool = False

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = linen.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                hidden = self.activation(hidden)
                if self.layer_norm:
                    hidden = linen.LayerNorm()(hidden)
        return hidden


class Param(linen.Module):
    init_value: float = 0.0
    size: int = 1

    @linen.compact
    def __call__(self):
        return self.param(
            "value", init_fn=lambda keys: jnp.full((self.size,), self.init_value)
        )


class LogParam(linen.Module):
    init_value: float = 1.0
    size: int = 1

    @linen.compact
    def __call__(self):
        log_value = self.param(
            "log_value",
            init_fn=lambda key: jnp.full((self.size,), jnp.log(self.init_value)),
        )
        return jnp.exp(log_value)


class PolicyModuleWithStd(linen.Module):
    param_size: int
    hidden_layer_sizes: Sequence[int]
    activation: ActivationFn
    kernel_init: jax.nn.initializers.Initializer
    layer_norm: bool
    noise_std_type: Literal["scalar", "log"]
    init_noise_std: float
    state_dependent_std: bool = False
    mean_clip_scale: Optional[float] = None
    mean_kernel_init: Optional[jax.nn.initializers.Initializer] = None

    @linen.compact
    def __call__(self, obs):
        if self.noise_std_type not in ["scalar", "log"]:
            raise ValueError(f"Unsupported noise_std_type: {self.noise_std_type}")

        outputs = MLP(
            layer_sizes=list(self.hidden_layer_sizes),
            activation=self.activation,
            kernel_init=self.kernel_init,
            layer_norm=self.layer_norm,
            activate_final=True,
        )(obs)

        mean_kernel_init = (
            self.mean_kernel_init if self.mean_kernel_init is not None else self.kernel_init
        )
        mean_params = linen.Dense(self.param_size, kernel_init=mean_kernel_init)(outputs)
        if self.mean_clip_scale is not None:
            mean_params = self.mean_clip_scale * (
                mean_params / (1.0 + jnp.abs(mean_params))
            )

        if self.state_dependent_std:
            log_std_output = linen.Dense(self.param_size, kernel_init=self.kernel_init)(outputs)
            std_params = jnp.exp(log_std_output) if self.noise_std_type == "log" else log_std_output
        else:
            std_module = (
                LogParam(self.init_noise_std, size=self.param_size, name="std_logparam")
                if self.noise_std_type == "log"
                else Param(self.init_noise_std, size=self.param_size, name="std_param")
            )
            std_params = std_module()

        return mean_params, jnp.broadcast_to(std_params, mean_params.shape)


@dataclasses.dataclass
class FeedForwardNetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]


def _get_obs_state_size(obs_size: ObservationSize, obs_key: str) -> int:
    obs_size = obs_size[obs_key] if isinstance(obs_size, Mapping) else obs_size
    return jax.tree_util.tree_flatten(obs_size)[0][-1]


def normalizer_select(
    processor_params: RunningStatisticsState, obs_key: str
) -> RunningStatisticsState:
    return RunningStatisticsState(
        count=processor_params.count,
        mean=processor_params.mean[obs_key],
        summed_variance=processor_params.summed_variance[obs_key],
        std=processor_params.std[obs_key],
    )


def make_policy_network(
    param_size: int,
    obs_size: ObservationSize,
    preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm: bool = False,
    obs_key: str = "state",
    distribution_type: Literal["normal", "tanh_normal"] = "tanh_normal",
    noise_std_type: Literal["scalar", "log"] = "scalar",
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
    mean_clip_scale: Optional[float] = None,
    mean_kernel_init: Optional[Initializer] = None,
):
    if distribution_type == "tanh_normal":
        policy_module = MLP(
            layer_sizes=list(hidden_layer_sizes) + [param_size],
            activation=activation,
            kernel_init=kernel_init,
            layer_norm=layer_norm,
        )
    elif distribution_type == "normal":
        policy_module = PolicyModuleWithStd(
            param_size=param_size,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            kernel_init=kernel_init,
            layer_norm=layer_norm,
            noise_std_type=noise_std_type,
            init_noise_std=init_noise_std,
            state_dependent_std=state_dependent_std,
            mean_clip_scale=mean_clip_scale,
            mean_kernel_init=mean_kernel_init,
        )
    else:
        raise ValueError(f"Unsupported distribution_type: {distribution_type}")

    def apply(processor_params, policy_params, obs):
        if isinstance(obs, Mapping):
            obs = preprocess_observations_fn(
                obs[obs_key], normalizer_select(processor_params, obs_key)
            )
        else:
            obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs)

    dummy_obs = jnp.zeros((1, _get_obs_state_size(obs_size, obs_key)))

    return FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs),
        apply=apply,
    )


def make_value_network(
    obs_size: ObservationSize,
    preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    obs_key: str = "state",
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
) -> FeedForwardNetwork:
    value_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [1],
        activation=activation,
        kernel_init=kernel_init,
    )

    def apply(processor_params, value_params, obs):
        if isinstance(obs, Mapping):
            obs = preprocess_observations_fn(
                obs[obs_key], normalizer_select(processor_params, obs_key)
            )
        else:
            obs = preprocess_observations_fn(obs, processor_params)
        return jnp.squeeze(value_module.apply(value_params, obs), axis=-1)

    dummy_obs = jnp.zeros((1, _get_obs_state_size(obs_size, obs_key)))
    return FeedForwardNetwork(
        init=lambda key: value_module.init(key, dummy_obs),
        apply=apply,
    )


@flax.struct.dataclass
class PPONetworks:
    policy_network: FeedForwardNetwork
    value_network: FeedForwardNetwork
    parametric_action_distribution: ParametricDistribution


def make_ppo_networks(
    observation_size: ObservationSize,
    action_size: int,
    preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: ActivationFn = linen.swish,
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
    distribution_type: Literal["normal", "tanh_normal"] = "tanh_normal",
    noise_std_type: Literal["scalar", "log"] = "scalar",
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
    policy_network_kernel_init_fn: Initializer = jax.nn.initializers.lecun_uniform,
    policy_network_kernel_init_kwargs: Optional[Mapping[str, Any]] = None,
    value_network_kernel_init_fn: Initializer = jax.nn.initializers.lecun_uniform,
    value_network_kernel_init_kwargs: Optional[Mapping[str, Any]] = None,
    mean_clip_scale: Optional[float] = None,
    mean_kernel_init_fn: Optional[Initializer] = None,
    mean_kernel_init_kwargs: Optional[Mapping[str, Any]] = None,
):
    policy_kernel_init_kwargs = policy_network_kernel_init_kwargs or {}
    value_kernel_init_kwargs = value_network_kernel_init_kwargs or {}
    mean_kernel_init_kwargs_ = mean_kernel_init_kwargs or {}

    if distribution_type == "normal":
        dist = NormalDistribution(event_size=action_size)
    elif distribution_type == "tanh_normal":
        dist = NormalTanhDistribution(event_size=action_size)
    else:
        raise ValueError(f"Unsupported distribution_type: {distribution_type}")

    policy_network = make_policy_network(
        dist.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
        obs_key=policy_obs_key,
        distribution_type=distribution_type,
        noise_std_type=noise_std_type,
        init_noise_std=init_noise_std,
        state_dependent_std=state_dependent_std,
        kernel_init=policy_network_kernel_init_fn(**policy_kernel_init_kwargs),
        mean_clip_scale=mean_clip_scale,
        mean_kernel_init=(
            mean_kernel_init_fn(**mean_kernel_init_kwargs_)
            if mean_kernel_init_fn is not None
            else None
        ),
    )
    value_network = make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        obs_key=value_obs_key,
        kernel_init=value_network_kernel_init_fn(**value_kernel_init_kwargs),
    )

    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=dist,
    )


def make_inference_fn(ppo_networks: PPONetworks, compute_value: bool = False):
    def make_policy(params: Params, deterministic: bool = False):
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def policy(observations: Observation, key_sample: PRNGKey) -> Tuple[Action, Extra]:
            param_subset = (params[0], params[1])
            logits = policy_network.apply(*param_subset, observations)
            if deterministic:
                return ppo_networks.parametric_action_distribution.mode(logits), {}
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(raw_actions)
            extras = {
                "log_prob": log_prob,
                "raw_action": raw_actions,
                "distribution_params": logits,
            }
            if compute_value:
                extras["value"] = ppo_networks.value_network.apply(
                    params[0], params[2], observations
                )
            return postprocessed_actions, extras

        return policy

    return make_policy


# ---------------------------------------------------------------------------
# Main checkpoint-loading helper
# ---------------------------------------------------------------------------


def load_expert_checkpoint(
    checkpoints_dir: str,
    observation_size: int = STATE_DIM,
    action_size: int = ACTION_DIM,
    network_config: Optional[dict] = None,
    checkpoint_step: Optional[int] = None,
) -> dict:
    """Load a PPO expert checkpoint and return inference utilities.

    Args:
        checkpoints_dir: Directory containing numbered checkpoint sub-folders.
        observation_size: Observation space dimension.
        action_size: Action space dimension.
        network_config: Network architecture kwargs forwarded to
            ``make_ppo_networks`` (defaults to ``DEFAULT_NETWORK_CONFIG``).
        checkpoint_step: Specific training step to load. Pass ``None``
            (default) to load the latest available checkpoint.

    Returns:
        Dictionary with keys:
            - ``actor_fn``: Compiled deterministic policy callable.
            - ``get_action``: Wrapper accepting a numpy array or a Brax-style dict.
            - ``actor_params``: Raw checkpoint params tuple.
            - ``network_config``: Network config used.
    """
    if network_config is None:
        network_config = DEFAULT_NETWORK_CONFIG

    selected_ckpt = find_checkpoint(checkpoints_dir, step=checkpoint_step)
    print(f"Loading checkpoint: {selected_ckpt}")

    params = load_orbax_checkpoint(selected_ckpt)
    print(f"Checkpoint loaded — {len(params)} elements (normalizer, policy, value)")

    ppo_network = make_ppo_networks(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=normalize,
        **network_config,
    )

    _make_policy = make_inference_fn(ppo_network)
    policy_fn = _make_policy(params, deterministic=True)
    print("Deterministic inference function created.")

    def get_action(obs) -> np.ndarray:
        """Accept a numpy array or a Brax-style dict and return an action."""
        if not isinstance(obs, dict):
            obs = {"state": jnp.array(obs)}
        action, _ = policy_fn(obs, jax.random.PRNGKey(0))
        return np.array(action)

    return {
        "actor_fn": policy_fn,
        "get_action": get_action,
        "actor_params": params,
        "network_config": network_config,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a PPO expert checkpoint to a pickle file."
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        required=True,
        help="Directory containing the numbered orbax checkpoint sub-folders.",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        required=True,
        help="Environment name (used for the output pickle filename).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run identifier appended to the output filename (e.g. 20250904-225910).",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help=(
            "Training step of the checkpoint to load. "
            "Omit to load the latest available checkpoint."
        ),
    )
    parser.add_argument("--state-dim", type=int, default=STATE_DIM)
    parser.add_argument("--action-dim", type=int, default=ACTION_DIM)
    parser.add_argument("--max-action", type=float, default=MAX_ACTION)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    checkpoints_dir = args.checkpoints_dir
    env_name = args.env_name
    run_id = args.run_id
    state_dim = args.state_dim
    action_dim = args.action_dim
    max_action = args.max_action

    actor = load_expert_checkpoint(
        checkpoints_dir=checkpoints_dir,
        observation_size=state_dim,
        action_size=action_dim,
        network_config=DEFAULT_NETWORK_CONFIG,
        checkpoint_step=args.checkpoint_step,
    )

    # Extract weights into a pure-NumPy portable actor (no JAX/Flax/orbax needed
    # to load it later). The deterministic Brax PPO policy is:
    #   logits = swish-MLP(normalize(obs))            # last layer = 2*action_dim
    #   loc    = logits[..., :action_dim]
    #   action = tanh(loc)
    # Observations are normalised with the running-statistics state stored in
    # params[0] (mean/std per obs_key), with no extra epsilon (std already has
    # std_eps baked in) and no clipping.
    params = actor["actor_params"]
    norm_state = params[0]
    policy_params = params[1]
    policy_tree = policy_params.get("params", policy_params)
    layers = ordered_dense_from_flax(policy_tree, "hidden")

    obs_key = DEFAULT_NETWORK_CONFIG["policy_obs_key"]
    norm_mean = norm_state.mean
    norm_std = norm_state.std
    obs_mean = norm_mean[obs_key] if isinstance(norm_mean, Mapping) else norm_mean
    obs_std = norm_std[obs_key] if isinstance(norm_std, Mapping) else norm_std

    portable = PortableActor(
        layers=layers,
        activation="swish",
        output={"type": "split_tanh", "action_dim": action_dim},
        obs_mean=np.asarray(obs_mean),
        obs_std=np.asarray(obs_std),
        obs_norm_eps=0.0,
        meta={
            "algo": "PPOExpert",
            "env_name": env_name,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "obs_key": obs_key,
        },
    )

    pickle_path = f"./actor-PPOExpert-{env_name}-{run_id}.pkl"
    save_actor(pickle_path, portable)
    print(f"Saved actor to: {pickle_path}")
    print(f"File size: {os.path.getsize(pickle_path) / 1024 / 1024:.2f} MB")

    # Verify round-trip against the original JAX actor (array and dict inputs).
    loaded_actor = load_actor(pickle_path)

    test_obs_array = np.random.randn(state_dim).astype(np.float32)
    test_obs_dict = {"state": jnp.array(test_obs_array)}

    action_array = actor["get_action"](test_obs_array)
    loaded_action = loaded_actor["get_action"](obs=test_obs_array)
    loaded_action_dict = loaded_actor["get_action"](obs={"state": test_obs_array})

    print(
        f"Array input  — action shape: {action_array.shape}, "
        f"range: [{action_array.min():.3f}, {action_array.max():.3f}]"
    )
    print(
        f"Portable dict input identical to array input: "
        f"{np.allclose(loaded_action, loaded_action_dict, atol=1e-6)}"
    )
    print(
        f"Loaded actor — actions match original: "
        f"{np.allclose(action_array, loaded_action, atol=1e-5)}"
    )


if __name__ == "__main__":
    main()
