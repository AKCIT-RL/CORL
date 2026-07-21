import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import distrax

from portable_actor import (
    PortableActor,
    load_actor,
    ordered_dense_from_flax,
    save_actor,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

STATE_DIM = 48
ACTION_DIM = 12
MAX_ACTION = 1.0

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    kernel_init: Callable[[Any, Sequence[int], Any], jnp.ndarray] = default_init()
    add_layer_norm: bool = False
    layer_norm_final: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if self.add_layer_norm:
                if self.layer_norm_final or i + 1 < len(self.hidden_dims):
                    x = nn.LayerNorm()(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
        return x


class GaussianPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -20.0
    log_std_max: Optional[float] = 2.0
    final_fc_init_scale: float = 1e-3

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0
    ) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims, activate_final=True)(observations)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)

        log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        return distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_awac_checkpoint(checkpoint_path: str, state_dim: int = STATE_DIM) -> dict:
    """Load an AWAC actor from a JAX checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        state_dim: Dimension of the observation space (used for model init).

    Returns:
        Dictionary with keys:
            - ``actor_fn_deterministic``: JIT-compiled deterministic policy.
            - ``actor_fn_stochastic``: JIT-compiled stochastic policy.
            - ``get_action``: Wrapper with obs normalisation (temperature=0 → deterministic).
            - ``actor_params``: Loaded model parameters.
            - ``obs_mean``: Observation mean used for normalisation.
            - ``obs_std``: Observation std used for normalisation.
            - ``config``: Parsed config.yaml.
    """
    ckpt_path = Path(checkpoint_path).resolve()

    with open(ckpt_path / "config.yaml") as f:
        config_dict = yaml.safe_load(f)

    checkpoint_data = np.load(
        ckpt_path / "checkpoint_final.npz", allow_pickle=True
    )

    raw_actor_params = checkpoint_data["actor_params"].item()
    obs_mean = jnp.array(checkpoint_data["obs_mean"])
    obs_std = jnp.array(checkpoint_data["obs_std"])

    print(f"obs_mean: {obs_mean[:5]}... (first 5)")
    print(f"obs_std:  {obs_std[:5]}... (first 5)")

    params_tree = raw_actor_params.get("params", raw_actor_params)
    action_dim = params_tree["log_stds"].shape[0]

    actor_model = GaussianPolicy(
        hidden_dims=config_dict["actor_hidden_dims"],
        action_dim=action_dim,
    )

    dummy_obs = jnp.zeros((1, state_dim))
    initial_vars = actor_model.init(jax.random.PRNGKey(0), dummy_obs)
    actor_params = flax.serialization.from_state_dict(initial_vars, raw_actor_params)

    @jax.jit
    def actor_fn_deterministic(params, observations):
        return actor_model.apply(params, observations, temperature=0.0).loc

    @jax.jit
    def actor_fn_stochastic(params, observations, temperature):
        dist = actor_model.apply(params, observations, temperature=temperature)
        return dist.sample(seed=jax.random.PRNGKey(0))

    def get_action(obs: np.ndarray, temperature: float = 0.0) -> np.ndarray:
        norm_obs = (obs - obs_mean) / (obs_std + 1e-5)
        if temperature == 0.0:
            action = actor_fn_deterministic(actor_params, norm_obs)
        else:
            action = actor_fn_stochastic(actor_params, norm_obs, temperature)
        return np.array(jnp.clip(action, -1.0, 1.0))

    return {
        "actor_fn_deterministic": actor_fn_deterministic,
        "actor_fn_stochastic": actor_fn_stochastic,
        "get_action": get_action,
        "actor_params": actor_params,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "config": config_dict,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an AWAC checkpoint to a pickle file."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the AWAC checkpoint directory.",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        required=True,
        help="Environment name (used for the output pickle filename).",
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

    checkpoint_path = args.checkpoint_path
    env_name = args.env_name
    state_dim = args.state_dim
    max_action = args.max_action

    actor = load_awac_checkpoint(checkpoint_path=checkpoint_path, state_dim=state_dim)

    # Extract weights into a pure-NumPy portable actor (no JAX/Flax/distrax
    # needed to load it later). GaussianPolicy is deterministic at
    # temperature=0: the action is ``clip(means, -1, 1)``. The means come from a
    # hidden MLP (ReLU after every layer, since activate_final=True) followed by
    # a separate output Dense (no activation). add_layer_norm defaults to False.
    policy_params = actor["actor_params"]["params"]
    real_action_dim = int(policy_params["log_stds"].shape[0])
    hidden_layers = ordered_dense_from_flax(policy_params["MLP_0"], "Dense")
    mean_dense = policy_params["Dense_0"]
    layers = hidden_layers + [
        (np.asarray(mean_dense["kernel"]), np.asarray(mean_dense["bias"]))
    ]
    portable = PortableActor(
        layers=layers,
        activation="relu",
        output={"type": "clip", "low": -1.0, "high": 1.0},
        obs_mean=actor["obs_mean"],
        obs_std=actor["obs_std"],
        obs_norm_eps=1e-5,
        meta={
            "algo": "AWAC",
            "env_name": env_name,
            "state_dim": state_dim,
            "action_dim": real_action_dim,
            "max_action": max_action,
        },
    )

    run_id = Path(checkpoint_path).name
    pickle_path = f"./actor-{run_id}.pkl"
    save_actor(pickle_path, portable)
    print(f"Saved actor to: {pickle_path}")

    # Verify round-trip against the original JAX actor (temperature=0).
    loaded_actor = load_actor(pickle_path)

    test_obs = np.random.randn(1, state_dim).astype(np.float32)
    original_action = actor["get_action"](test_obs)
    loaded_action = loaded_actor["get_action"](obs=test_obs)

    print("Original action:", original_action)
    print("Loaded action:  ", loaded_action)
    print(f"Actions match: {np.allclose(original_action, loaded_action, atol=1e-5)}")


if __name__ == "__main__":
    main()
