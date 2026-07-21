# source https://github.com/nikhilbarhate99/min-decision-transformer
# https://arxiv.org/abs/2106.01345
import collections
import os
import uuid
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import flax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyrallis
import wandb
from flax import linen as nn
from flax.training.train_state import TrainState
from tqdm import tqdm

import minari

from algorithms.utils.wrapper_gym import get_env, maybe_get_shifted_env

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class DTConfig:
    # wandb project name
    project: str = "CORL"
    # wandb group name
    group: str = "DT-D4RL"
    # wandb run name
    name: str = "DT"
    # transformer hidden dim
    embedding_dim: int = 128
    # depth of the transformer model
    num_layers: int = 3
    # number of heads in the attention
    num_heads: int = 1
    # maximum sequence length during training
    seq_len: int = 20
    # maximum rollout length, needed for the positional embeddings
    episode_len: int = 1000
    # attention dropout
    attention_dropout: float = 0.1
    # residual dropout
    residual_dropout: float = 0.1
    # embeddings dropout
    embedding_dropout: float = 0.1
    # maximum range for the symmetric actions, [-1, 1]
    max_action: float = 1.0
    # training dataset and evaluation environment
    env_name: str = "halfcheetah-medium-v2"
    dataset_id: str = "halfcheetah-medium-v2"
    # AdamW optimizer learning rate
    learning_rate: float = 1e-4
    # AdamW optimizer betas
    betas: Tuple[float, float] = (0.9, 0.999)
    # AdamW weight decay
    weight_decay: float = 1e-4
    # maximum gradient norm during training, optional
    clip_grad: Optional[float] = 0.25
    # training batch size
    batch_size: int = 64
    # total training steps
    update_steps: int = 100_000
    # warmup steps for the learning rate scheduler
    warmup_steps: int = 10_000
    # reward scaling, to reduce the magnitude
    reward_scale: float = 0.001
    # target return-to-go for the prompting during evaluation
    target_returns: Tuple[float, ...] = (12000.0, 6000.0)
    # number of episodes to run during evaluation
    eval_episodes: int = 10
    # number of episodes for the final evaluation (larger -> lower variance)
    n_eval_episodes_final: int = 50
    # fixed seed for evaluation rollouts (reproducible / comparable)
    eval_seed: int = 0
    # evaluation frequency, will evaluate eval_every training steps
    eval_every: int = 10_000
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = None
    # save an intermediate checkpoint every N evaluations (final always saved)
    checkpoints_every: int = 10
    # training random seed
    seed: int = 0
    # training seed (alternative to seed, used in configs)
    train_seed: int = 0
    # evaluation seed
    eval_seed: int = 42
    # training device
    device: str = "cuda"
    # command type for environment (e.g., "direction", "forward", "forwardfixed")
    command_type: Optional[str] = None
    # Optional Tier-5 shifted evaluation: JSON string of flattened env-config
    # overrides (e.g. stronger push-recovery kicks) applied only to a second eval
    # env. Normalization reuses the in-distribution dataset refs, so the shifted
    # score is comparable to the final score (their gap = robustness gap).
    eval_shift: Optional[str] = None
    # deterministic torch (not used in JAX, but present in configs for compatibility)
    deterministic_torch: bool = False
    # number of workers (not used in JAX, but present in configs for compatibility)
    num_workers: int = 4

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
        # Use train_seed if provided, otherwise use seed
        # If train_seed was explicitly set (not default 0), use it; otherwise use seed
        if self.train_seed != 0:
            self.seed = self.train_seed
        elif self.seed != 0:
            self.train_seed = self.seed


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class MaskedCausalAttention(nn.Module):
    h_dim: int
    max_T: int
    n_heads: int
    drop_p: float
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray, training=True) -> jnp.ndarray:
        B, T, C = x.shape
        N, D = self.n_heads, C // self.n_heads
        # rearrange q, k, v as (B, N, T, D)
        q = (
            nn.Dense(self.h_dim, kernel_init=self.kernel_init)(x)
            .reshape(B, T, N, D)
            .transpose(0, 2, 1, 3)
        )
        k = (
            nn.Dense(self.h_dim, kernel_init=self.kernel_init)(x)
            .reshape(B, T, N, D)
            .transpose(0, 2, 1, 3)
        )
        v = (
            nn.Dense(self.h_dim, kernel_init=self.kernel_init)(x)
            .reshape(B, T, N, D)
            .transpose(0, 2, 1, 3)
        )
        # causal mask
        ones = jnp.ones((self.max_T, self.max_T))
        mask = jnp.tril(ones).reshape(1, 1, self.max_T, self.max_T)
        # weights (B, N, T, T) jax
        weights = jnp.einsum("bntd,bnfd->bntf", q, k) / jnp.sqrt(D)
        # causal mask applied to weights
        weights = jnp.where(mask[..., :T, :T] == 0, -jnp.inf, weights[..., :T, :T])
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = jax.nn.softmax(weights, axis=-1)
        # attention (B, N, T, D)
        attention = nn.Dropout(self.drop_p, deterministic=not training)(
            jnp.einsum("bntf,bnfd->bntd", normalized_weights, v)
        )
        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(0, 2, 1, 3).reshape(B, T, N * D)
        out = nn.Dropout(self.drop_p, deterministic=not training)(
            nn.Dense(self.h_dim)(attention)
        )
        return out


class Block(nn.Module):
    h_dim: int
    max_T: int
    n_heads: int
    attention_dropout: float
    residual_dropout: float
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray, training=True) -> jnp.ndarray:
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + MaskedCausalAttention(
            self.h_dim, self.max_T, self.n_heads, self.attention_dropout
        )(
            x, training=training
        )  # residual
        x = nn.LayerNorm()(x)
        # MLP
        out = nn.Dense(4 * self.h_dim, kernel_init=self.kernel_init)(x)
        out = nn.gelu(out)
        out = nn.Dense(self.h_dim, kernel_init=self.kernel_init)(out)
        out = nn.Dropout(self.residual_dropout, deterministic=not training)(out)
        # residual
        x = x + out
        x = nn.LayerNorm()(x)
        return x


class DecisionTransformer(nn.Module):
    state_dim: int
    act_dim: int
    num_layers: int
    h_dim: int
    seq_len: int
    n_heads: int
    attention_dropout: float
    residual_dropout: float
    embedding_dropout: float
    max_timestep: int = 4096
    kernel_init: Callable = default_init()

    def setup(self) -> None:
        self.blocks = [
            Block(self.h_dim, 3 * self.seq_len, self.n_heads, self.attention_dropout, self.residual_dropout)
            for _ in range(self.num_layers)
        ]
        # projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm()
        self.embed_drop = nn.Dropout(self.embedding_dropout)
        self.embed_timestep = nn.Embed(self.max_timestep, self.h_dim)
        self.embed_rtg = nn.Dense(self.h_dim, kernel_init=self.kernel_init)
        self.embed_state = nn.Dense(self.h_dim, kernel_init=self.kernel_init)
        # continuous actions
        self.embed_action = nn.Dense(self.h_dim, kernel_init=self.kernel_init)
        self.use_action_tanh = True
        # prediction heads
        self.predict_rtg = nn.Dense(1, kernel_init=self.kernel_init)
        self.predict_state = nn.Dense(self.state_dim, kernel_init=self.kernel_init)
        self.predict_action = nn.Dense(self.act_dim, kernel_init=self.kernel_init)

    def __call__(
        self,
        timesteps: jnp.ndarray,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        returns_to_go: jnp.ndarray,
        training=True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)
        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings
        # stack rtg, states and actions and reshape sequence as
        # (r1, s1, a1, r2, s2, a2 ...)
        h = (
            jnp.stack((returns_embeddings, state_embeddings, action_embeddings), axis=1)
            .transpose(0, 2, 1, 3)
            .reshape(B, 3 * T, self.h_dim)
        )
        h = self.embed_ln(h)
        h = self.embed_drop(h, deterministic=not training)
        # transformer and prediction
        for block in self.blocks:
            h = block(h, training=training)
        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t, a_t
        h = h.reshape(B, T, 3, self.h_dim).transpose(0, 2, 1, 3)
        # get predictions
        return_preds = self.predict_rtg(h[:, 2])  # predict next rtg given r, s, a
        state_preds = self.predict_state(h[:, 2])  # predict next state given r, s, a
        action_preds = self.predict_action(h[:, 1])
        if self.use_action_tanh:
            action_preds = jnp.tanh(action_preds)

        return state_preds, action_preds, return_preds


def discount_cumsum(x: jnp.ndarray, gamma: float) -> jnp.ndarray:
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t + 1]
    return disc_cumsum


def get_traj(dataset_id: str):
    print("processing: ", dataset_id)
    dataset = minari.load_dataset(dataset_id)
    
    paths = []
    all_observations = []
    
    for episode in dataset.iterate_episodes():
        terminations = np.array(episode.terminations, dtype=bool)
        truncations = np.array(episode.truncations, dtype=bool)
        terminals = np.logical_or(terminations, truncations).astype(np.float32)
        episode_data = {
            "observations": np.array(episode.observations[:-1], dtype=np.float32),
            "next_observations": np.array(episode.observations[1:], dtype=np.float32),
            "actions": np.array(episode.actions, dtype=np.float32),
            "rewards": np.array(episode.rewards, dtype=np.float32),
            "terminals": terminals,
        }
        paths.append(episode_data)
        all_observations.append(episode_data["observations"])
    
    all_obs = np.concatenate(all_observations, axis=0)
    returns = np.array([np.sum(p["rewards"]) for p in paths])
    num_samples = np.sum([p["rewards"].shape[0] for p in paths])
    print(f"Number of samples collected: {num_samples}")
    print(
        f"Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}"
    )
    obs_mean = all_obs.mean(axis=0)
    obs_std = all_obs.std(axis=0) + 1e-6
    return paths, obs_mean, obs_std


class Trajectory(NamedTuple):
    timesteps: np.ndarray  # num_ep x max_len
    states: np.ndarray  # num_ep x max_len x state_dim
    actions: np.ndarray  # num_ep x max_len x act_dim
    returns_to_go: np.ndarray  # num_ep x max_len x 1
    masks: np.ndarray  # num_ep x max_len


def padd_by_zero(arr: jnp.ndarray, pad_to: int) -> jnp.ndarray:
    return np.pad(arr, ((0, pad_to - arr.shape[0]), (0, 0)), mode="constant")


def make_padded_trajectories(
    config: DTConfig,
) -> Tuple[Trajectory, int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    trajectories, mean, std = get_traj(config.dataset_id)
    # Calculate returns to go for all trajectories
    # Normalize states
    max_len = 0
    traj_lengths = []
    for traj in trajectories:
        traj["returns_to_go"] = discount_cumsum(traj["rewards"], 1.0) * config.reward_scale
        traj["observations"] = (traj["observations"] - mean) / std
        max_len = max(max_len, traj["observations"].shape[0])
        traj_lengths.append(traj["observations"].shape[0])
    # Pad trajectories
    padded_trajectories = {key: [] for key in Trajectory._fields}
    for traj in trajectories:
        timesteps = np.arange(0, len(traj["observations"]))
        padded_trajectories["timesteps"].append(
            padd_by_zero(timesteps.reshape(-1, 1), max_len).reshape(-1)
        )
        padded_trajectories["states"].append(
            padd_by_zero(traj["observations"], max_len)
        )
        padded_trajectories["actions"].append(padd_by_zero(traj["actions"], max_len))
        padded_trajectories["returns_to_go"].append(
            padd_by_zero(traj["returns_to_go"].reshape(-1, 1), max_len)
        )
        padded_trajectories["masks"].append(
            padd_by_zero(
                np.ones((len(traj["observations"]), 1)).reshape(-1, 1), max_len
            ).reshape(-1)
        )
    return (
        Trajectory(
            timesteps=np.stack(padded_trajectories["timesteps"]),
            states=np.stack(padded_trajectories["states"]),
            actions=np.stack(padded_trajectories["actions"]),
            returns_to_go=np.stack(padded_trajectories["returns_to_go"]),
            masks=np.stack(padded_trajectories["masks"]),
        ),
        len(trajectories),
        jnp.array(traj_lengths),
        mean,
        std,
    )


def sample_start_idx(
    rng: jax.random.PRNGKey,
    traj_idx: int,
    padded_traj_length: jnp.ndarray,
    seq_len: int,
) -> jnp.ndarray:
    """
    Determine the start_idx for given trajectory, the trajectories are padded to max_len.
    Therefore, naively sample from 0, max_len will produce bunch of all zero data.
    To avoid that, we refer padded_traj_length, the list of actual trajectory length + seq_len
    """
    traj_len = padded_traj_length[traj_idx]
    start_idx = jax.random.randint(rng, (1,), 0, traj_len - seq_len - 1)
    return start_idx


def extract_traj(
    traj_idx: jnp.ndarray, start_idx: jnp.ndarray, traj: Trajectory, seq_len: int
) -> Trajectory:
    """
    Extract the trajectory with seq_len for given traj_idx and start_idx
    """
    return jax.tree_util.tree_map(
        lambda x: jax.lax.dynamic_slice_in_dim(x[traj_idx], start_idx, seq_len),
        traj,
    )


@partial(jax.jit, static_argnums=(2, 3, 4))
def sample_traj_batch(
    rng,
    traj: Trajectory,
    batch_size: int,
    seq_len: int,
    episode_num: int,
    padded_traj_lengths: jnp.ndarray,
) -> Trajectory:
    traj_idx = jax.random.randint(rng, (batch_size,), 0, episode_num)  # B
    start_idx = jax.vmap(sample_start_idx, in_axes=(0, 0, None, None))(
        jax.random.split(rng, batch_size), traj_idx, padded_traj_lengths, seq_len
    ).reshape(
        -1
    )  # B
    return jax.vmap(extract_traj, in_axes=(0, 0, None, None))(
        traj_idx, start_idx, traj, seq_len
    )


class DTTrainState(NamedTuple):
    transformer: TrainState


class DT(object):

    @classmethod
    def update(
        self, train_state: DTTrainState, batch: Trajectory, rng: jax.random.PRNGKey
    ) -> Tuple[Any, jnp.ndarray]:
        timesteps, states, actions, returns_to_go, traj_mask = (
            batch.timesteps,
            batch.states,
            batch.actions,
            batch.returns_to_go,
            batch.masks,
        )

        def loss_fn(params):
            state_preds, action_preds, return_preds = train_state.transformer.apply_fn(
                params, timesteps, states, actions, returns_to_go, rngs={"dropout": rng}
            )  # B x T x state_dim, B x T x act_dim, B x T x 1
            # mask actions
            actions_masked = actions * traj_mask[:, :, None]
            action_preds_masked = action_preds * traj_mask[:, :, None]
            # Calculate mean squared error loss
            action_loss = jnp.mean(jnp.square(action_preds_masked - actions_masked))
            return action_loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(train_state.transformer.params)
        # Apply gradient clipping
        transformer = train_state.transformer.apply_gradients(grads=grad)
        return train_state._replace(transformer=transformer), loss

    @classmethod
    def get_action(
        self,
        train_state: DTTrainState,
        timesteps: jnp.ndarray,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        returns_to_go: jnp.ndarray,
    ) -> jnp.ndarray:
        state_preds, action_preds, return_preds = train_state.transformer.apply_fn(
            train_state.transformer.params,
            timesteps,
            states,
            actions,
            returns_to_go,
            training=False,
        )
        return action_preds


def create_dt_train_state(
    rng: jax.random.PRNGKey, state_dim: int, act_dim: int, config: DTConfig
) -> DTTrainState:
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        num_layers=config.num_layers,
        h_dim=config.embedding_dim,
        seq_len=config.seq_len,
        n_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
    )
    rng, init_rng = jax.random.split(rng)
    # initialize params
    params = model.init(
        init_rng,
        timesteps=jnp.zeros((1, config.seq_len), jnp.int32),
        states=jnp.zeros((1, config.seq_len, state_dim), jnp.float32),
        actions=jnp.zeros((1, config.seq_len, act_dim), jnp.float32),
        returns_to_go=jnp.zeros((1, config.seq_len, 1), jnp.float32),
        training=False,
    )
    # optimizer with warmup
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=config.learning_rate,
        transition_steps=config.warmup_steps,
    )
    decay_fn = optax.constant_schedule(config.learning_rate)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[config.warmup_steps],
    )
    tx = optax.chain(
        optax.clip_by_global_norm(config.clip_grad) if config.clip_grad else optax.identity(),
        optax.adamw(
            learning_rate=schedule_fn,
            weight_decay=config.weight_decay,
            b1=config.betas[0],
            b2=config.betas[1],
        ),
    )
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return DTTrainState(train_state)


def get_dt_from_checkpoint(
    checkpoint_path: str,
    state_dim: int,
    action_dim: int,
) -> Dict[str, Any]:
    """
    Load a Decision Transformer from a JAX checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        state_dim: Dimension of the state/observation space
        action_dim: Dimension of the action space
    
    Returns:
        Dictionary containing:
            - 'transformer_fn': JIT-compiled transformer function
            - 'get_action': Function that takes observations and returns actions
            - 'transformer_params': The loaded model parameters
            - 'obs_mean': Observation mean for normalization
            - 'obs_std': Observation std for normalization
            - 'config': The loaded config dictionary
    """
    from pathlib import Path
    
    ckpt_path = Path(checkpoint_path).resolve()
    
    # Load config if available
    config_file = ckpt_path / "config.yaml"
    config = None
    if config_file.exists():
        if yaml is None:
            raise ImportError("PyYAML is required to load config. Install with: pip install pyyaml")
        with open(config_file) as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    # Extract config parameters
    embedding_dim = config.get("embedding_dim", 128)
    num_layers = config.get("num_layers", 3)
    num_heads = config.get("num_heads", 1)
    seq_len = config.get("seq_len", 20)
    attention_dropout = config.get("attention_dropout", 0.1)
    residual_dropout = config.get("residual_dropout", 0.1)
    embedding_dropout = config.get("embedding_dropout", 0.1)
    
    # Find checkpoint files
    ckpt_files = list(ckpt_path.glob("checkpoint_*.npz"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_path}")
    
    # Sort by step number (extract number from filename)
    def get_step(p):
        name = p.stem  # e.g., "checkpoint_1000" or "checkpoint_final"
        step_str = name.split("_")[-1]
        if step_str == "final":
            return float("inf")
        return int(step_str)
    
    ckpt_files.sort(key=get_step)
    
    # Look for final checkpoint specifically
    final_ckpt = ckpt_path / "checkpoint_final.npz"
    if final_ckpt.exists():
        selected_ckpt = final_ckpt
    else:
        selected_ckpt = ckpt_files[-1]
    
    print(f"[get_dt_from_checkpoint] Loading checkpoint: {selected_ckpt}")
    
    # Load checkpoint
    checkpoint = np.load(selected_ckpt, allow_pickle=True)
    transformer_params = checkpoint["transformer_params"].item()  # .item() to extract dict from 0-d array
    obs_mean = checkpoint["state_mean"]
    obs_std = checkpoint["state_std"]
    
    # Create Decision Transformer model
    dt_model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=action_dim,
        num_layers=num_layers,
        h_dim=embedding_dim,
        seq_len=seq_len,
        n_heads=num_heads,
        attention_dropout=attention_dropout,
        residual_dropout=residual_dropout,
        embedding_dropout=embedding_dropout,
    )
    
    # Convert params back to FrozenDict
    transformer_params = flax.serialization.from_state_dict(
        dt_model.init(
            jax.random.PRNGKey(0),
            timesteps=jnp.zeros((1, seq_len), jnp.int32),
            states=jnp.zeros((1, seq_len, state_dim), jnp.float32),
            actions=jnp.zeros((1, seq_len, action_dim), jnp.float32),
            returns_to_go=jnp.zeros((1, seq_len, 1), jnp.float32),
            training=False,
        ),
        transformer_params
    )
    
    # Create JIT-compiled transformer function
    @jax.jit
    def transformer_fn(
        timesteps: jnp.ndarray,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        returns_to_go: jnp.ndarray,
    ) -> jnp.ndarray:
        state_preds, action_preds, return_preds = dt_model.apply(
            transformer_params,
            timesteps,
            states,
            actions,
            returns_to_go,
            training=False,
        )
        return action_preds
    
    # Create a get_action function for easier inference
    def get_action(
        timesteps: jnp.ndarray,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        returns_to_go: jnp.ndarray,
    ) -> np.ndarray:
        """Get action from the Decision Transformer."""
        action_preds = transformer_fn(timesteps, states, actions, returns_to_go)
        return np.array(action_preds)
    
    print(f"[get_dt_from_checkpoint] Loaded DT with embedding_dim={embedding_dim}, "
          f"num_layers={num_layers}, seq_len={seq_len}")
    
    return {
        "transformer_fn": transformer_fn,
        "get_action": get_action,
        "transformer_params": transformer_params,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "config": config,
    }


def evaluate(
    policy_fn: Callable,
    train_state: DTTrainState,
    env: gym.Env,
    config: DTConfig,
    target_return: float,
    state_mean=0,
    state_std=1,
    seed: int = 0,
    num_episodes: Optional[int] = None,
) -> float:
    env.reset_rng(seed)
    n_eps = num_episodes if num_episodes is not None else config.eval_episodes
    eval_batch_size = 1  # required for forward pass
    total_reward = 0
    total_timesteps = 0
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # Convert state_mean and state_std to JAX arrays and ensure correct shape
    state_mean = jnp.array(state_mean).reshape(-1) if not isinstance(state_mean, (int, float)) else jnp.array([state_mean] * state_dim)
    state_std = jnp.array(state_std).reshape(-1) if not isinstance(state_std, (int, float)) else jnp.array([state_std] * state_dim)
    # same as timesteps used for training the transformer
    timesteps = jnp.arange(0, config.episode_len, 1, jnp.int32)
    # repeat
    timesteps = jnp.repeat(timesteps[None, :], eval_batch_size, axis=0)
    for _ in range(n_eps):
        # zeros place holders
        actions = jnp.zeros(
            (eval_batch_size, config.episode_len, act_dim), dtype=jnp.float32
        )
        states = jnp.zeros(
            (eval_batch_size, config.episode_len, state_dim), dtype=jnp.float32
        )
        rewards_to_go = jnp.zeros(
            (eval_batch_size, config.episode_len, 1), dtype=jnp.float32
        )
        # init episode
        running_state, _ = env.reset()
        # Ensure running_state is 1D
        running_state = np.array(running_state).flatten()
        running_reward = 0
        running_rtg = target_return * config.reward_scale
        for t in range(config.episode_len):
            total_timesteps += 1
            # add state in placeholder and normalize
            normalized_state = (jnp.array(running_state) - state_mean) / state_std
            states = states.at[0, t].set(normalized_state)
            # calculate running rtg and add in placeholder
            running_rtg = running_rtg - (running_reward * config.reward_scale)
            rewards_to_go = rewards_to_go.at[0, t].set(running_rtg)
            if t < config.seq_len:
                act_preds = policy_fn(
                    train_state,
                    timesteps[:, : t + 1],
                    states[:, : t + 1],
                    actions[:, : t + 1],
                    rewards_to_go[:, : t + 1],
                )
                act = act_preds[0, -1]
            else:
                act_preds = policy_fn(
                    train_state,
                    timesteps[:, t - config.seq_len + 1 : t + 1],
                    states[:, t - config.seq_len + 1 : t + 1],
                    actions[:, t - config.seq_len + 1 : t + 1],
                    rewards_to_go[:, t - config.seq_len + 1 : t + 1],
                )
                act = act_preds[0, -1]
            running_state, running_reward, done, truncated, _ = env.step(np.array(act))
            # Ensure running_state is 1D
            running_state = np.array(running_state).flatten()
            # add action in placeholder
            actions = actions.at[0, t].set(act)
            total_reward += running_reward
            if done or truncated:
                break
    mean_reward = total_reward / n_eps
    # Normalize using the env's D4RL-style reference scores (loaded from the
    # dataset metadata). Falls back to the raw return when refs are absent.
    normalized = env.get_normalized_score(mean_reward)
    normalized_score = normalized * 100 if normalized is not None else mean_reward
    return normalized_score, mean_reward


def record_dt_video(
    policy_fn: Callable,
    train_state: DTTrainState,
    config: DTConfig,
    target_return: float,
    save_path: str,
    state_mean=0,
    state_std=1,
):
    """Roll out the DT policy for a single episode and save an mp4 of it."""
    # Headless rendering backend (matches algorithms/utils/save_video.py).
    os.environ.setdefault("MUJOCO_GL", "egl")

    render_trajectory = []

    def render_callback(_, state):
        render_trajectory.append(state)

    env = get_env(
        config.env_name,
        config.device,
        render_callback=render_callback,
        command_type=config.command_type,
    )

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    state_mean = (
        jnp.array(state_mean).reshape(-1)
        if not isinstance(state_mean, (int, float))
        else jnp.array([state_mean] * state_dim)
    )
    state_std = (
        jnp.array(state_std).reshape(-1)
        if not isinstance(state_std, (int, float))
        else jnp.array([state_std] * state_dim)
    )
    timesteps = jnp.arange(0, config.episode_len, 1, jnp.int32)[None, :]

    actions = jnp.zeros((1, config.episode_len, act_dim), dtype=jnp.float32)
    states = jnp.zeros((1, config.episode_len, state_dim), dtype=jnp.float32)
    rewards_to_go = jnp.zeros((1, config.episode_len, 1), dtype=jnp.float32)

    running_state, _ = env.reset()
    running_state = np.array(running_state).flatten()
    running_reward = 0
    running_rtg = target_return * config.reward_scale
    for t in range(config.episode_len):
        normalized_state = (jnp.array(running_state) - state_mean) / state_std
        states = states.at[0, t].set(normalized_state)
        running_rtg = running_rtg - (running_reward * config.reward_scale)
        rewards_to_go = rewards_to_go.at[0, t].set(running_rtg)
        if t < config.seq_len:
            act_preds = policy_fn(
                train_state,
                timesteps[:, : t + 1],
                states[:, : t + 1],
                actions[:, : t + 1],
                rewards_to_go[:, : t + 1],
            )
        else:
            act_preds = policy_fn(
                train_state,
                timesteps[:, t - config.seq_len + 1 : t + 1],
                states[:, t - config.seq_len + 1 : t + 1],
                actions[:, t - config.seq_len + 1 : t + 1],
                rewards_to_go[:, t - config.seq_len + 1 : t + 1],
            )
        act = act_preds[0, -1]
        running_state, running_reward, done, truncated, _ = env.step(np.array(act))
        env.render()
        running_state = np.array(running_state).flatten()
        actions = actions.at[0, t].set(act)
        if bool(np.asarray(done).reshape(-1)[0]) or bool(
            np.asarray(truncated).reshape(-1)[0]
        ):
            break

    # Establish a headless GL context before rendering (mirrors save_video.py).
    try:
        import mujoco.egl

        gl_context = mujoco.egl.GLContext(1024, 1024)
        gl_context.make_current()
    except Exception as e:  # pragma: no cover - depends on GPU/driver
        print(f"[record_dt_video] could not create EGL context: {e}")

    save_dir = os.path.dirname(os.path.abspath(save_path))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    env.save_video(render_trajectory, save_path=save_path)


@pyrallis.wrap()
def train(config: DTConfig):
    wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=config,
        id=str(uuid.uuid4()),
    )

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    env = get_env(config.env_name, config.device, command_type=config.command_type, dataset=minari.load_dataset(config.dataset_id))
    shifted_env = maybe_get_shifted_env(
        config.env_name, config.device, command_type=config.command_type,
        dataset=minari.load_dataset(config.dataset_id), eval_shift=config.eval_shift,
    )
    rng = jax.random.PRNGKey(config.seed)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    trajectories, episode_num, traj_lengths, state_mean, state_std = (
        make_padded_trajectories(config)
    )
    # create trainer
    rng, subkey = jax.random.split(rng)
    train_state = create_dt_train_state(subkey, state_dim, act_dim, config)

    algo = DT()
    update_fn = jax.jit(algo.update)
    for i in tqdm(range(1, config.update_steps + 1), smoothing=0.1, dynamic_ncols=True):
        rng, data_rng, update_rng = jax.random.split(rng, 3)
        traj_batch = sample_traj_batch(
            data_rng,
            trajectories,
            config.batch_size,
            config.seq_len,
            episode_num,
            traj_lengths,
        )  # B x T x D
        train_state, action_loss = update_fn(train_state, traj_batch, update_rng)  # update parameters
        
        wandb.log({"training/action_loss": action_loss}, step=i)

        if i % config.eval_every == 0:
            # evaluate on env for each target return
            for target_return in config.target_returns:
                score, raw_score = evaluate(
                    algo.get_action, train_state, env, config, target_return, state_mean, state_std, seed=config.eval_seed
                )
                wandb.log(
                    {
                        f"eval/{target_return}_score": score,
                        f"eval/{target_return}_raw_score": raw_score,
                    },
                    step=i,
                )
                print(f"Step {i}, Target Return {target_return}: {score}")

            if config.checkpoints_path is not None and (i // config.eval_every) % config.checkpoints_every == 0:
                checkpoint = {
                    "transformer_params": flax.serialization.to_state_dict(train_state.transformer.params),
                    "state_mean": np.array(state_mean),
                    "state_std": np.array(state_std),
                    "step": i,
                }
                checkpoint_path = os.path.join(config.checkpoints_path, f"checkpoint_{i}.npz")
                np.savez(checkpoint_path, **checkpoint)
                print(f"Saved checkpoint to {checkpoint_path}")

    # final evaluation
    for target_return in config.target_returns:
        score, raw_score = evaluate(
            algo.get_action, train_state, env, config, target_return, state_mean, state_std,
            seed=config.eval_seed, num_episodes=config.n_eval_episodes_final,
        )
        wandb.log({
            f"eval/{target_return}_final_score": score,
            f"eval/{target_return}_final_raw_score": raw_score,
        })
        print(f"Final Score for Target Return {target_return}: {score}")

        # Tier-5 shifted evaluation: same policy under a harder / held-out
        # perturbation regime. Reuses the in-distribution refs, so the gap
        # measures robustness.
        if shifted_env is not None:
            shifted_score, shifted_raw = evaluate(
                algo.get_action, train_state, shifted_env, config, target_return, state_mean, state_std,
                seed=config.eval_seed, num_episodes=config.n_eval_episodes_final,
            )
            wandb.log({
                f"eval/shifted_{target_return}_final_score": shifted_score,
                f"eval/shifted_{target_return}_final_raw_score": shifted_raw,
                f"eval/{target_return}_robustness_gap": score - shifted_score,
            })
            print(f"Final Shifted Score for Target Return {target_return}: {shifted_score}")

    # Save final checkpoint
    if config.checkpoints_path is not None:
        checkpoint = {
            "transformer_params": flax.serialization.to_state_dict(train_state.transformer.params),
            "state_mean": np.array(state_mean),
            "state_std": np.array(state_std),
            "step": config.update_steps,
        }
        checkpoint_path = os.path.join(config.checkpoints_path, f"checkpoint_final.npz")
        np.savez(checkpoint_path, **checkpoint)
        print(f"Saved final checkpoint to {checkpoint_path}")

    # Record a rollout video of the final policy (highest target return)
    try:
        video_dir = config.checkpoints_path if config.checkpoints_path is not None else "videos"
        video_path = os.path.join(video_dir, f"{config.name}.mp4")
        record_dt_video(
            algo.get_action,
            train_state,
            config,
            config.target_returns[0],
            video_path,
            state_mean,
            state_std,
        )
        wandb.log({"eval/video": wandb.Video(video_path)})
        print(f"Saved rollout video to {video_path}")
    except Exception as e:
        print(f"[video] failed to record rollout video: {e}")

    wandb.finish()


if __name__ == "__main__":
    train()
