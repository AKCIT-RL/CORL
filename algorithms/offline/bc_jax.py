# Behavioral Cloning implementation in JAX
# Simple supervised learning approach for offline RL
import os
import uuid
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import minari
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyrallis
import tqdm
import wandb
import yaml
from flax.training.train_state import TrainState
import flax.serialization
from flax.core import FrozenDict

from algorithms.utils.wrapper_gym import GymWrapper, get_env, maybe_get_shifted_env, record_policy_video
from algorithms.utils.dataset import qlearning_dataset

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

@dataclass
class BCConfig:
    # wandb project name
    project: str = "train-TD3-BC"
    # wandb group name
    group: str = "BC"
    # wandb run name
    name: str = "BC"
    # training dataset and evaluation environment
    env: str = "halfcheetah-medium-expert-v2"  # OpenAI gym environment name
    dataset_id: str = "halfcheetah-medium-expert-v2"
    command_type: Optional[str] = None
    # Optional Tier-5 shifted evaluation: JSON string of flattened env-config
    # overrides (e.g. stronger push-recovery kicks) applied only to a second eval
    # env. Normalization reuses the in-distribution dataset refs, so the shifted
    # score is comparable to eval/final_score (their gap = robustness gap).
    eval_shift: Optional[str] = None
    # total gradient updates during training
    max_timesteps: int = int(1e6)
    # training batch size
    batch_size: int = 256
    # maximum size of the replay buffer
    buffer_size: int = 2_000_000
    # what top fraction of the dataset (sorted by return) to use
    frac: float = 0.1 # WARNING: NOT USED
    # maximum possible trajectory length
    max_traj_len: int = 1000 # WARNING: NOT USED
    # whether to normalize states
    normalize: bool = True
    # discount factor
    discount: float = 0.99 # WARNING: NOT USED
    # evaluation frequency, will evaluate eval_freq training steps
    eval_freq: int = int(5e3)
    # number of episodes to run during evaluation
    n_episodes: int = 10
    # number of episodes for the final evaluation (larger -> lower variance)
    n_eval_episodes_final: int = 50
    # fixed seed for evaluation rollouts (reproducible / comparable)
    eval_seed: int = 0
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = None
    # save an intermediate checkpoint every N evaluations (final always saved)
    checkpoints_every: int = 10
    # file name for loading a model, optional
    load_model: Optional[str] = None
    # training random seed
    seed: int = 0
    # training device
    device: str = "cuda"
    # NETWORK
    hidden_dims: Tuple[int, ...] = (256, 256)
    actor_lr: float = 1e-3
    n_jitted_updates: int = 8

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

    def __hash__(
        self,
    ):  # make config hashable to be specified as static_argnums in jax.jit.
        return hash(self.__repr__())


def default_init(scale: Optional[float] = jnp.sqrt(2).item()):
    return nn.initializers.orthogonal(scale)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    kernel_init: Callable[[Any, Sequence[int], Any], jnp.ndarray] = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, hidden_dims in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dims, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.layer_norm:  # Add layer norm after activation
                    if i + 1 < len(self.hidden_dims):
                        x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x


class BCActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    max_action: float = 1.0  # In D4RL, action is scaled to [-1, 1]

    @nn.compact
    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        action = MLP((*self.hidden_dims, self.action_dim))(observation)
        action = self.max_action * jnp.tanh(
            action
        )  # scale to [-max_action, max_action]
        return action


class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray


def get_dataset(
    dataset, config: BCConfig, clip_to_eps: bool = True, eps: float = 1e-5
) -> Tuple:
    # dataset = d4rl.qlearning_dataset(env)

    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

    imputed_next_observations = np.roll(dataset["observations"], -1, axis=0)
    same_obs = np.all(
        np.isclose(imputed_next_observations, dataset["next_observations"], atol=1e-5),
        axis=-1,
    )
    dones = 1.0 - same_obs.astype(np.float32)
    dones[-1] = 1

    dataset = Transition(
        observations=jnp.array(dataset["observations"], dtype=jnp.float32),
        actions=jnp.array(dataset["actions"], dtype=jnp.float32),
        rewards=jnp.array(dataset["rewards"], dtype=jnp.float32),
        dones=jnp.array(dones, dtype=jnp.float32),
        next_observations=jnp.array(dataset["next_observations"], dtype=jnp.float32),
    )
    # shuffle data and select the first buffer_size samples
    data_size = min(config.buffer_size, len(dataset.observations))
    rng = jax.random.PRNGKey(config.seed)
    rng, rng_permute = jax.random.split(rng, 2)
    perm = jax.random.permutation(rng_permute, len(dataset.observations))
    dataset = jax.tree_util.tree_map(lambda x: x[perm], dataset)
    assert len(dataset.observations) >= data_size
    dataset = jax.tree_util.tree_map(lambda x: x[:data_size], dataset)
    # normalize states
    obs_mean, obs_std = 0, 1
    if config.normalize:
        obs_mean = dataset.observations.mean(0)
        obs_std = dataset.observations.std(0)
        dataset = dataset._replace(
            observations=(dataset.observations - obs_mean) / (obs_std + 1e-5),
            next_observations=(dataset.next_observations - obs_mean) / (obs_std + 1e-5),
        )
    return dataset, obs_mean, obs_std


def update_by_loss_grad(
    train_state: TrainState, loss_fn: Callable
) -> Tuple[TrainState, jnp.ndarray]:
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(train_state.params)
    new_train_state = train_state.apply_gradients(grads=grad)
    return new_train_state, loss


class BCTrainState(NamedTuple):
    actor: TrainState
    max_action: float = 1.0


class BC(object):
    @classmethod
    def update_actor(
        self,
        train_state: BCTrainState,
        batch: Transition,
        rng: jax.Array,
        config: BCConfig,
    ) -> Tuple["BCTrainState", jnp.ndarray]:
        def actor_loss_fn(actor_params: FrozenDict[str, Any]) -> jnp.ndarray:
            predicted_action = train_state.actor.apply_fn(
                actor_params, batch.observations
            )
            # Simple MSE loss for behavioral cloning
            bc_loss = jnp.square(predicted_action - batch.actions).mean()
            return bc_loss

        new_actor, actor_loss = update_by_loss_grad(train_state.actor, actor_loss_fn)
        return train_state._replace(actor=new_actor), actor_loss

    @classmethod
    def update_n_times(
        self,
        train_state: BCTrainState,
        data: Transition,
        rng: jax.Array,
        config: BCConfig,
    ) -> Tuple["BCTrainState", Dict]:
        actor_loss = 0.0
        for _ in range(
            config.n_jitted_updates
        ):  # we can jit for loop for static unroll
            rng, batch_rng = jax.random.split(rng, 2)
            batch_idx = jax.random.randint(
                batch_rng, (config.batch_size,), 0, len(data.observations)
            )
            batch: Transition = jax.tree_util.tree_map(lambda x: x[batch_idx], data)
            rng, actor_rng = jax.random.split(rng, 2)
            train_state, actor_loss = self.update_actor(
                train_state, batch, actor_rng, config
            )
        return train_state, {
            "actor_loss": actor_loss,
        }

    @classmethod
    def get_action(
        self,
        train_state: BCTrainState,
        obs: jnp.ndarray,
        max_action: float = 1.0,  # In D4RL, action is scaled to [-1, 1]
    ) -> jnp.ndarray:
        action = train_state.actor.apply_fn(train_state.actor.params, obs)
        action = action.clip(-max_action, max_action)
        return action

def load_bc_train_state(
    actions: jnp.ndarray,
    config: BCConfig
) -> BCTrainState:
    if not config.load_model:
        raise ValueError("No checkpoint specified for loading.")
    
    checkpoint = np.load(config.load_model, allow_pickle=True)
    actor_params = checkpoint["actor_params"].item()
    
    actor_model = BCActor(
        action_dim=actions.shape[-1],
        hidden_dims=config.hidden_dims,
    )
    actor_train_state = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_params,
        tx=optax.adam(config.actor_lr),
    )
    
    return BCTrainState(
        actor=actor_train_state,
    )    

def create_bc_train_state(
    rng: jax.Array,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    config: BCConfig,
) -> BCTrainState:
    action_dim = actions.shape[-1]
    actor_model = BCActor(
        action_dim=action_dim,
        hidden_dims=config.hidden_dims,
    )
    rng, actor_rng = jax.random.split(rng, 2)
    # initialize actor
    actor_train_state: TrainState = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, observations),
        tx=optax.adam(config.actor_lr),
    )
    return BCTrainState(
        actor=actor_train_state,
    )


def get_actor_from_checkpoint(
    checkpoint_path: str,
    state_dim: int,
    action_dim: int,
    max_action: float = 1.0,
) -> Dict[str, Any]:
    """
    Load a BC actor from a JAX checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        state_dim: Dimension of the state/observation space
        action_dim: Dimension of the action space
        max_action: Maximum action value (default: 1.0)
        checkpoint_id: Which checkpoint to load (-1 for latest, -2 for final)
        hidden_dims: Hidden layer dimensions (default from config if available)
    
    Returns:
        Dictionary containing:
            - 'actor_fn': JIT-compiled function that takes observations and returns actions
            - 'actor_params': The loaded model parameters
            - 'obs_mean': Observation mean for normalization
            - 'obs_std': Observation std for normalization
            - 'config': The loaded config dictionary
    """
    ckpt_path = Path(checkpoint_path).resolve()
    
    # Load config if available
    config_file = ckpt_path / "config.yaml"
    config = None
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
        # Use hidden_dims from config if available
        if config and "hidden_dims" in config:
            hidden_dims = tuple(config["hidden_dims"])
    
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
    
    print(f"[get_actor_from_checkpoint] Loading checkpoint: {selected_ckpt}")
    
    # Load checkpoint
    checkpoint = np.load(selected_ckpt, allow_pickle=True)
    actor_params = checkpoint["actor_params"].item()  # .item() to extract dict from 0-d array
    obs_mean = checkpoint["obs_mean"]
    obs_std = checkpoint["obs_std"]
    
    # Create actor model
    actor_model = BCActor(
        hidden_dims=hidden_dims,
        action_dim=action_dim,
        max_action=max_action,
    )
    
    # Convert params back to FrozenDict
    actor_params = flax.serialization.from_state_dict(
        actor_model.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim))),
        actor_params
    )
    
    # Create JIT-compiled action function
    @jax.jit
    def actor_fn(obs: jnp.ndarray) -> jnp.ndarray:
        actions = actor_model.apply(actor_params, obs)
        return jnp.clip(jnp.asarray(actions), -max_action, max_action)
    
    # Create a normalized action function that includes obs normalization
    def get_action(obs: np.ndarray) -> np.ndarray:
        obs_normalized = (obs - obs_mean) / (obs_std + 1e-5)
        action = actor_fn(jnp.array(obs_normalized))
        return np.array(action)
    
    print(f"[get_actor_from_checkpoint] Loaded actor with hidden_dims={hidden_dims}, action_dim={action_dim}")
    
    return {
        "actor_fn": actor_fn,
        "get_action": get_action,
        "actor_params": actor_params,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "config": config,
    }


def evaluate(
    policy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    env: GymWrapper,
    num_episodes: int,
    obs_mean,
    obs_std,
    seed: int = 0,
) -> float:
    env.reset_rng(seed)
    episode_returns = []
    for _ in range(num_episodes):
        episode_return = 0
        observation, _ = env.reset()
        done = truncated = False
        while not done and not truncated:
            observation = (observation - obs_mean) / (obs_std + 1e-5)
            action = policy_fn(observation)
            observation, reward, done, truncated, _ = env.step(action)
            episode_return += reward
        episode_returns.append(episode_return)
    
    mean_return = np.mean(episode_returns)
    # Normalize using the env's D4RL-style reference scores (loaded from the
    # dataset metadata). Falls back to the raw return when refs are absent.
    normalized = env.get_normalized_score(mean_return)
    normalized_score = normalized * 100 if normalized is not None else mean_return.item()
    return normalized_score, mean_return.item()

@pyrallis.wrap()  # type: ignore
def train(config: BCConfig):
    wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        id=str(uuid.uuid4()),
    )

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            yaml.safe_dump(asdict(config), f)

    minari_dataset = minari.load_dataset(config.dataset_id)
    dataset = qlearning_dataset(minari_dataset)
    env = get_env(config.env, config.device, command_type=config.command_type, dataset=minari_dataset)
    shifted_env = maybe_get_shifted_env(
        config.env, config.device, command_type=config.command_type,
        dataset=minari_dataset, eval_shift=config.eval_shift,
    )

    rng = jax.random.PRNGKey(config.seed)
    dataset, obs_mean, obs_std = get_dataset(dataset, config)
    # create train_state
    rng, subkey = jax.random.split(rng)
    example_batch: Transition = jax.tree_util.tree_map(lambda x: x[0], dataset)
    if config.load_model:
        print(f"Loading model from checkpoint: {config.load_model}")
        train_state = load_bc_train_state(example_batch.actions,config)
    else:
        train_state = create_bc_train_state(
            subkey, example_batch.observations, example_batch.actions, config
        )
    algo = BC()
    update_fn = jax.jit(algo.update_n_times, static_argnums=(3,))
    act_fn = jax.jit(algo.get_action)

    num_steps = config.max_timesteps // config.n_jitted_updates
    eval_interval = config.eval_freq // config.n_jitted_updates
    for i in tqdm.tqdm(range(1, num_steps + 1), smoothing=0.1, dynamic_ncols=True):
        rng, update_rng = jax.random.split(rng)
        train_state, update_info = update_fn(
            train_state,
            dataset,
            update_rng,
            config,
        )  # update parameters
        train_metrics = {f"training/{k}": v for k, v in update_info.items()}
        wandb.log(train_metrics, step=i)

        if i % eval_interval == 0:
            policy_fn = partial(act_fn, train_state)
            normalized_score, raw_score = evaluate(
                policy_fn,
                env,
                num_episodes=config.n_episodes,
                obs_mean=obs_mean,
                obs_std=obs_std,
                seed=config.eval_seed,
            )
            eval_metrics = {
                "eval/score": normalized_score,
                "eval/raw_score": raw_score,
            }
            wandb.log(eval_metrics, step=i)

            if config.checkpoints_path is not None and (i // eval_interval) % config.checkpoints_every == 0:
                checkpoint = {
                    "actor_params": flax.serialization.to_state_dict(train_state.actor.params),
                    "obs_mean": np.array(obs_mean),
                    "obs_std": np.array(obs_std),
                    "step": i,
                }
                checkpoint_path = os.path.join(config.checkpoints_path, f"checkpoint_{i}.npz")
                np.savez(checkpoint_path, **checkpoint)
                print(f"Saved checkpoint to {checkpoint_path}")

    # final evaluation
    policy_fn = partial(act_fn, train_state)
    normalized_score, raw_score = evaluate(
        policy_fn,
        env,
        num_episodes=config.n_eval_episodes_final,
        obs_mean=obs_mean,
        obs_std=obs_std,
        seed=config.eval_seed,
    )
    print("Final Evaluation Score:", normalized_score)
    wandb.log({
        "eval/final_score": normalized_score,
        "eval/final_raw_score": raw_score,
    })

    # Tier-5 shifted evaluation: same policy under a harder / held-out perturbation
    # regime. Reuses the in-distribution refs, so the gap measures robustness.
    if shifted_env is not None:
        shifted_score, shifted_raw = evaluate(
            policy_fn,
            shifted_env,
            num_episodes=config.n_eval_episodes_final,
            obs_mean=obs_mean,
            obs_std=obs_std,
            seed=config.eval_seed,
        )
        print("Final Shifted Evaluation Score:", shifted_score)
        wandb.log({
            "eval/shifted_final_score": shifted_score,
            "eval/shifted_final_raw_score": shifted_raw,
            "eval/robustness_gap": normalized_score - shifted_score,
        })

    # Save final checkpoint
    if config.checkpoints_path is not None:
        checkpoint = {
            "actor_params": flax.serialization.to_state_dict(train_state.actor.params),
            "obs_mean": np.array(obs_mean),
            "obs_std": np.array(obs_std),
            "step": num_steps,
        }
        checkpoint_path = os.path.join(config.checkpoints_path, "checkpoint_final.npz")
        np.savez(checkpoint_path, **checkpoint)
        print(f"Saved final checkpoint to {checkpoint_path}")
        
    # Record a rollout video of the final policy
    try:
        video_dir = config.checkpoints_path if config.checkpoints_path is not None else "videos"
        video_path = os.path.join(video_dir, f"{config.name}.mp4")
        record_policy_video(
            env_name=config.env,
            act=lambda o: policy_fn(o),
            obs_mean=obs_mean,
            obs_std=obs_std,
            device=config.device,
            save_path=video_path,
            command_type=config.command_type,
        )
        wandb.log({"eval/video": wandb.Video(video_path)})
        print(f"Saved rollout video to {video_path}")
    except Exception as e:
        print(f"[video] failed to record rollout video: {e}")

    wandb.finish()

if __name__ == "__main__":
    train() # type: ignore

