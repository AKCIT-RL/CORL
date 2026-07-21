import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
import sys
import warnings

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

from portable_actor import (
    PortableActor,
    load_actor,
    ordered_dense_from_flax,
    save_actor,
)

def _register_numpy2_compat_aliases() -> None:
    """Register aliases so NumPy 1.x can unpickle objects saved with NumPy 2.x."""
    
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r".*numpy\.core.*"
        )
        
        np_core = np.core
        sys.modules.setdefault("numpy._core", np_core)
        for submodule in (
            "_multiarray_umath",
            "multiarray",
            "umath",
            "overrides",
            "numeric",
            "numerictypes",
            "fromnumeric",
            "shape_base",
            "function_base",
            "getlimits",
            "_methods",
        ):
            target = getattr(np_core, submodule, None)
            if target is not None:
                sys.modules.setdefault(f"numpy._core.{submodule}", target)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

STATE_DIM = 48
ACTION_DIM = 12
MAX_ACTION = 1.0

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    kernel_init: Callable[[Any, Sequence[int], Any], jnp.ndarray] = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.layer_norm and i + 1 < len(self.hidden_dims):
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x


class BCActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    max_action: float = 1.0

    @nn.compact
    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        action = MLP((*self.hidden_dims, self.action_dim))(observation)
        return self.max_action * jnp.tanh(action)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def get_actor_from_checkpoint(
    checkpoint_path: str,
    state_dim: int,
    action_dim: int,
    max_action: float = 1.0,
) -> dict:
    """Load a BC actor from a JAX checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        state_dim: Dimension of the observation space.
        action_dim: Dimension of the action space.
        max_action: Maximum action magnitude (default: 1.0).

    Returns:
        Dictionary with keys:
            - ``actor_fn``: JIT-compiled function (obs -> actions).
            - ``get_action``: Wrapper that normalises obs before inference.
            - ``actor_params``: Loaded model parameters.
            - ``obs_mean``: Observation mean used for normalisation.
            - ``obs_std``: Observation std used for normalisation.
            - ``config``: Parsed config.yaml, or ``None`` if unavailable.
    """
    ckpt_path = Path(checkpoint_path).resolve()

    # Load config
    config = None
    hidden_dims = (256, 256)
    config_file = ckpt_path / "config.yaml"
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
        if config and "hidden_dims" in config:
            hidden_dims = tuple(config["hidden_dims"])

    # Locate checkpoint file
    ckpt_files = list(ckpt_path.glob("checkpoint_*.npz"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_path}")

    def _get_step(p: Path) -> float:
        step_str = p.stem.split("_")[-1]
        return float("inf") if step_str == "final" else int(step_str)

    ckpt_files.sort(key=_get_step)

    final_ckpt = ckpt_path / "checkpoint_final.npz"
    selected_ckpt = final_ckpt if final_ckpt.exists() else ckpt_files[-1]
    print(f"[get_actor_from_checkpoint] Loading checkpoint: {selected_ckpt}")

    # Load parameters
    _register_numpy2_compat_aliases()
    checkpoint = np.load(selected_ckpt, allow_pickle=True)
    actor_params = checkpoint["actor_params"].item()
    obs_mean = checkpoint["obs_mean"]
    obs_std = checkpoint["obs_std"]
    print(f"obs_mean: {obs_mean}")
    print(f"obs_std: {obs_std}")

    # Build model and restore params
    actor_model = BCActor(
        hidden_dims=hidden_dims,
        action_dim=action_dim,
        max_action=max_action,
    )
    actor_params = flax.serialization.from_state_dict(
        actor_model.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim))),
        actor_params,
    )

    @jax.jit
    def actor_fn(obs: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(actor_model.apply(actor_params, obs), -max_action, max_action)

    def get_action(obs: np.ndarray) -> np.ndarray:
        obs_normalized = (obs - obs_mean) / (obs_std + 1e-5)
        return np.array(actor_fn(jnp.array(obs_normalized)))

    print(
        f"[get_actor_from_checkpoint] Loaded actor with "
        f"hidden_dims={hidden_dims}, action_dim={action_dim}"
    )

    return {
        "actor_fn": actor_fn,
        "get_action": get_action,
        "actor_params": actor_params,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "config": config,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a BC checkpoint to a pickle file.")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the BC checkpoint directory.",
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


def main() -> None:
    args = parse_args()

    checkpoint_path = args.checkpoint_path
    env_name = args.env_name
    state_dim = args.state_dim
    action_dim = args.action_dim
    max_action = args.max_action

    actor = get_actor_from_checkpoint(
        checkpoint_path=checkpoint_path,
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
    )

    # Extract weights into a pure-NumPy portable actor (no JAX/Flax needed to
    # load it later). BCActor: a single MLP whose Dense layers are hidden +
    # output, ReLU after every layer except the last, then ``max_action*tanh``.
    mlp_params = actor["actor_params"]["params"]["MLP_0"]
    layers = ordered_dense_from_flax(mlp_params, "Dense")
    portable = PortableActor(
        layers=layers,
        activation="relu",
        output={"type": "tanh_scaled", "max_action": max_action},
        obs_mean=actor["obs_mean"],
        obs_std=actor["obs_std"],
        obs_norm_eps=1e-5,
        meta={
            "algo": "BC",
            "env_name": env_name,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
        },
    )

    run_id = Path(checkpoint_path).name.split("-")[-1]
    pickle_path = f"./actor-BC-{env_name}-{run_id}.pkl"
    save_actor(pickle_path, portable)
    print(f"Saved actor to: {pickle_path}")

    # Verify round-trip against the original JAX actor.
    loaded_actor = load_actor(pickle_path)

    test_obs = np.random.randn(1, state_dim).astype(np.float32)
    original_action = actor["get_action"](test_obs)
    loaded_action = loaded_actor["get_action"](obs=test_obs)

    print("Original action:", original_action)
    print("Loaded action:  ", loaded_action)
    print(f"Actions match: {np.allclose(original_action, loaded_action, atol=1e-5)}")


if __name__ == "__main__":
    main()
