"""Portable, dependency-light actor for offline-RL checkpoints.

This module lets policies trained with JAX/Flax (BC, TD3+BC, IQL, AWAC,
Brax-PPO, ...) be exported as *pure data* and evaluated anywhere with only
NumPy installed. No JAX, Flax, distrax, cloudpickle, or matching Python
version is required on the consumer side.

Why this exists
---------------
Previously, checkpoints were exported by ``cloudpickle``-ing a live, jitted
JAX ``get_action`` closure. That couples every consumer (Isaac Sim, the real
robot, ...) to the *exact* Python/JAX/Flax/distrax versions used at export
time, because cloudpickle serialises Python bytecode plus JAX/Flax internals.

Here we instead serialise only the *data* (weight matrices + a tiny spec) and
reconstruct a pure-NumPy forward pass on load. The serialised file contains
only NumPy arrays, lists, floats, strings and ``None``; it is therefore
forward/backward compatible across Python and library versions.

The consumer contract is unchanged and algorithm-agnostic::

    from portable_actor import load_actor
    actor = load_actor("actor.pkl")
    action = actor["get_action"](obs=obs)   # same dict-style call as before

Producer (CORL) side::

    from portable_actor import PortableActor, save_actor, ordered_dense_from_flax
    layers = ordered_dense_from_flax(params["params"]["MLP_0"], "Dense")
    actor = PortableActor(
        layers=layers,
        activation="relu",
        output={"type": "tanh_scaled", "max_action": 1.0},
        obs_mean=obs_mean, obs_std=obs_std, obs_norm_eps=1e-5,
        meta={"env_name": env_name, "algo": "BC"},
    )
    save_actor("actor.pkl", actor)
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

FORMAT = "portable_actor_v1"


# ---------------------------------------------------------------------------
# Activations (pure NumPy)
# ---------------------------------------------------------------------------


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _swish(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid.
    return x * (0.5 * (1.0 + np.tanh(0.5 * x)))


def _identity(x: np.ndarray) -> np.ndarray:
    return x


_ACTIVATIONS = {
    "relu": _relu,
    "swish": _swish,
    "silu": _swish,  # alias
    "tanh": np.tanh,
    "identity": _identity,
}


def _layer_norm(
    x: np.ndarray,
    scale: Optional[np.ndarray],
    bias: Optional[np.ndarray],
    eps: float = 1e-6,
) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + eps)
    if scale is not None:
        y = y * scale
    if bias is not None:
        y = y + bias
    return y


# ---------------------------------------------------------------------------
# Portable actor
# ---------------------------------------------------------------------------


class PortableActor:
    """A feed-forward policy that runs entirely in NumPy.

    The forward pass is::

        x = (obs - obs_mean) / (obs_std + obs_norm_eps)
        [optional clip]
        for each dense layer except the last:
            x = activation(LayerNorm?(x @ W + b))
        x = x @ W_last + b_last
        action = output_op(x)

    ``output_op`` is one of:

    - ``tanh_scaled``: ``max_action * tanh(x)``           (BC, TD3+BC)
    - ``clip``:        ``clip(x, low, high)``             (IQL, AWAC; x is mean)
    - ``split_tanh``:  ``tanh(x[..., :action_dim])``      (Brax PPO tanh-normal)
    - ``identity``:    ``x``
    """

    def __init__(
        self,
        layers: Sequence[Tuple[np.ndarray, np.ndarray]],
        activation: str,
        output: Dict[str, Any],
        obs_mean: Any = 0.0,
        obs_std: Any = 1.0,
        obs_norm_eps: float = 1e-5,
        obs_clip: Optional[float] = None,
        layer_norms: Optional[Sequence[Optional[Tuple[np.ndarray, np.ndarray]]]] = None,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Supported: {sorted(_ACTIVATIONS)}"
            )
        self.layers: List[Tuple[np.ndarray, np.ndarray]] = [
            (np.asarray(w, dtype=np.float32), np.asarray(b, dtype=np.float32))
            for w, b in layers
        ]
        self.activation = activation
        self.output = dict(output)
        self.obs_mean = np.asarray(obs_mean, dtype=np.float32)
        self.obs_std = np.asarray(obs_std, dtype=np.float32)
        self.obs_norm_eps = float(obs_norm_eps)
        self.obs_clip = None if obs_clip is None else float(obs_clip)
        if layer_norms is None:
            layer_norms = [None] * len(self.layers)
        self.layer_norms: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [
            None
            if ln is None
            else (np.asarray(ln[0], dtype=np.float32), np.asarray(ln[1], dtype=np.float32))
            for ln in layer_norms
        ]
        self.meta: Dict[str, Any] = dict(meta or {})

    # -- forward ------------------------------------------------------------

    def _forward(self, obs: Any) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]

        x = (x - self.obs_mean) / (self.obs_std + self.obs_norm_eps)
        if self.obs_clip is not None:
            x = np.clip(x, -self.obs_clip, self.obs_clip)

        act = _ACTIVATIONS[self.activation]
        n = len(self.layers)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < n - 1:
                ln = self.layer_norms[i]
                if ln is not None:
                    x = _layer_norm(x, ln[0], ln[1])
                x = act(x)

        x = self._apply_output(x)
        return x[0] if single else x

    def _apply_output(self, x: np.ndarray) -> np.ndarray:
        t = self.output.get("type")
        if t == "tanh_scaled":
            return float(self.output.get("max_action", 1.0)) * np.tanh(x)
        if t == "clip":
            return np.clip(
                x,
                float(self.output.get("low", -1.0)),
                float(self.output.get("high", 1.0)),
            )
        if t == "split_tanh":
            action_dim = int(self.output["action_dim"])
            return np.tanh(x[..., :action_dim])
        if t == "identity":
            return x
        raise ValueError(f"Unknown output type: {t!r}")

    # -- public API ---------------------------------------------------------

    def get_action(self, obs: Any = None, **kwargs: Any) -> np.ndarray:
        """Return an action for ``obs``.

        ``obs`` may be a 1-D vector, a batched 2-D array, or a Brax-style
        mapping such as ``{"state": array}`` (the observation key defaults to
        ``"state"``; if absent and the mapping has a single entry, that entry is
        used).
        """
        if obs is None:
            if "observation" in kwargs:
                obs = kwargs["observation"]
            elif "observations" in kwargs:
                obs = kwargs["observations"]
            else:
                raise TypeError("get_action() missing required argument 'obs'")
        if isinstance(obs, Mapping):
            obs_key = self.meta.get("obs_key", "state")
            if obs_key in obs:
                obs = obs[obs_key]
            elif len(obs) == 1:
                obs = next(iter(obs.values()))
            else:
                raise KeyError(
                    f"Observation mapping has no key {obs_key!r}; "
                    f"available keys: {list(obs)}"
                )
        return self._forward(obs)

    # Allow ``actor(obs=...)`` just like the old jitted callable.
    __call__ = get_action

    # -- dict-style backward compatibility ---------------------------------
    # The old pickles were plain dicts; consumers do ``actor["get_action"]``,
    # ``actor.get("obs_mean")``, etc. Support that transparently.

    def _as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "get_action": self.get_action,
            "obs_mean": self.obs_mean,
            "obs_std": self.obs_std,
        }
        d.update(self.meta)
        return d

    def __getitem__(self, key: str) -> Any:
        return self._as_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._as_dict().get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._as_dict()

    # -- (de)serialisation as pure data ------------------------------------

    def to_state(self) -> Dict[str, Any]:
        return {
            "format": FORMAT,
            "layers": [(w, b) for w, b in self.layers],
            "activation": self.activation,
            "output": self.output,
            "obs_mean": self.obs_mean,
            "obs_std": self.obs_std,
            "obs_norm_eps": self.obs_norm_eps,
            "obs_clip": self.obs_clip,
            "layer_norms": self.layer_norms,
            "meta": self.meta,
        }

    @classmethod
    def from_state(cls, s: Mapping[str, Any]) -> "PortableActor":
        if s.get("format") != FORMAT:
            raise ValueError(f"Unsupported actor format: {s.get('format')!r}")
        return cls(
            layers=s["layers"],
            activation=s["activation"],
            output=s["output"],
            obs_mean=s.get("obs_mean", 0.0),
            obs_std=s.get("obs_std", 1.0),
            obs_norm_eps=s.get("obs_norm_eps", 1e-5),
            obs_clip=s.get("obs_clip"),
            layer_norms=s.get("layer_norms"),
            meta=s.get("meta"),
        )


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------


def save_actor(path: str, actor: PortableActor) -> None:
    """Serialise *actor* to *path* as pure data (stdlib pickle, protocol 4)."""
    with open(path, "wb") as f:
        pickle.dump(actor.to_state(), f, protocol=4)


def load_actor(path: str) -> Any:
    """Load an actor from *path*.

    Returns a :class:`PortableActor` for the portable format. For legacy
    cloudpickle dicts (containing a live ``get_action``) the dict is returned
    as-is, so old artefacts keep working where the original deps are present.
    """
    with open(path, "rb") as f:
        state = pickle.load(f)
    if isinstance(state, PortableActor):
        return state
    if isinstance(state, Mapping) and state.get("format") == FORMAT:
        return PortableActor.from_state(state)
    if isinstance(state, Mapping) and "get_action" in state:
        return state  # legacy artefact
    raise ValueError(f"Unrecognised actor file: {path}")


# ---------------------------------------------------------------------------
# Producer-only utilities (NumPy-only; safe to import without JAX/Flax)
# ---------------------------------------------------------------------------


def ordered_dense_from_flax(params: Mapping[str, Any], prefix: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract ``[(W, b), ...]`` from a Flax params mapping, ordered by index.

    *params* maps layer names to ``{"kernel": ..., "bias": ...}``. Names that
    match ``<prefix>_<i>`` (e.g. ``Dense_0``, ``hidden_2``) are returned ordered
    by the trailing integer ``i``. Arrays are converted to NumPy, so JAX arrays
    are accepted.
    """
    items: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for name, sub in params.items():
        if not name.startswith(prefix):
            continue
        if not (isinstance(sub, Mapping) and "kernel" in sub and "bias" in sub):
            continue
        try:
            idx = int(str(name).split("_")[-1])
        except ValueError:
            continue
        items.append((idx, np.asarray(sub["kernel"]), np.asarray(sub["bias"])))
    items.sort(key=lambda t: t[0])
    return [(w, b) for _, w, b in items]


def layer_norms_from_flax(
    params: Mapping[str, Any],
    n_layers: int,
    prefix: str = "LayerNorm",
) -> Optional[List[Optional[Tuple[np.ndarray, np.ndarray]]]]:
    """Extract optional LayerNorm ``(scale, bias)`` aligned to dense layers.

    Returns ``None`` if no LayerNorm params are present (the common case), so
    callers can pass the result straight through to :class:`PortableActor`.
    """
    found: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for name, sub in params.items():
        if not name.startswith(prefix):
            continue
        if not isinstance(sub, Mapping):
            continue
        try:
            idx = int(str(name).split("_")[-1])
        except ValueError:
            continue
        scale = sub.get("scale")
        bias = sub.get("bias")
        found[idx] = (
            np.asarray(scale) if scale is not None else None,
            np.asarray(bias) if bias is not None else None,
        )
    if not found:
        return None
    return [found.get(i) for i in range(n_layers)]
