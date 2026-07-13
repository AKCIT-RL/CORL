# Database Aggregation implementation in JAX
# Simple supervised learning approach for offline RL
import os
import time
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

from algorithms.utils.wrapper_gym import GymWrapper, get_env
from algorithms.utils.dataset import qlearning_dataset

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

@dataclass
class DAggerConfig:
    # wandb project name
    project: str = "train-TD3-BC"
    # wandb group name
    group: str = "DAgger"
    # wandb run name
    name: str = "DAgger"
    # training dataset and evaluation environment
    env: str = "halfcheetah-medium-expert-v2"  # OpenAI gym environment name
    dataset_id: str = "halfcheetah-medium-expert-v2"
    command_type: Optional[str] = None
    # initial gradient updates during training
    initial_timesteps: int = int(1e6)
    # gradient updates during DAgger iterations
    dagger_timesteps: int = int(1e5)
    # dagger iterations
    dagger_iterations: int = 10
    # dagger data collection episodes per iteration
    dagger_episodes: int = 100
    # number of parallel envs used only for DAgger data collection
    collect_num_envs: int = 8
    # file name to expert policy checkpoint
    expert_checkpoint: str = ""
    # decay factor for mixing expert and learned policy
    decay_factor: float = 0.9
    # training batch size
    batch_size: int = 256
    # maximum size of the replay buffer
    buffer_size: int = 2_500_000
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
    # directory for saving rollout videos (optional)
    video_dir: Optional[str] = None
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = None
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
        if self.video_dir is not None:
            self.video_dir = os.path.join(self.video_dir, self.name)

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


class DAggerActor(nn.Module):
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

def _get_rl_config(env_name: str):
   if env_name in mujoco_playground.manipulation._envs:
      return manipulation_params.brax_ppo_config(env_name)
   if env_name in mujoco_playground.locomotion._envs:
      return locomotion_params.brax_ppo_config(env_name)
   if env_name in mujoco_playground.dm_control_suite._envs:
      return dm_control_suite_params.brax_ppo_config(env_name)
   raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def _resolve_expert_checkpoint_path(load_checkpoint_path: Optional[str]) -> Path:
   if not load_checkpoint_path:
      raise ValueError("--expert_checkpoint is required for DAgger test.")

   ckpt_path = Path(load_checkpoint_path).expanduser().resolve()
   if not ckpt_path.exists():
      raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")

   if not ckpt_path.is_dir():
      raise ValueError(f"Checkpoint path must be a directory, got: {ckpt_path}")

   subdirs = [d for d in ckpt_path.iterdir() if d.is_dir() and d.name.isdigit()]
   if subdirs:
      subdirs.sort(key=lambda d: int(d.name))
      return subdirs[-1]
   return ckpt_path


def _to_policy_obs(observation):
   obs = jnp.asarray(observation)
   # PPO configs in this repo typically read state observations via policy_obs_key.
   return {"state": obs}

render_trajectory = []


def _first_env_state(state):
    def _select_first(x):
        shape = getattr(x, "shape", None)
        if shape is not None and len(shape) > 0:
            return x[0]
        return x

    return jax.tree_util.tree_map(_select_first, state)


def _render_callback(_, state):
    render_trajectory.append(_first_env_state(state))

def load_expert(config: DAggerConfig) -> Callable[[np.ndarray], np.ndarray]:
   checkpoint_path = _resolve_expert_checkpoint_path(config.expert_checkpoint)
   ppo_params = _get_rl_config(config.env)

   training_params = dict(ppo_params)
   training_params["num_timesteps"] = 0
   if "network_factory" in training_params:
      del training_params["network_factory"]

   network_factory_cfg: Any = ppo_params.get("network_factory", {})
   if hasattr(network_factory_cfg, "to_dict"):
      network_kwargs = network_factory_cfg.to_dict()
   elif isinstance(network_factory_cfg, dict):
      network_kwargs = network_factory_cfg
   else:
      network_kwargs = {}

   train_fn = partial(
        ppo.train,
        **training_params,
        network_factory=partial(
            ppo_networks.make_ppo_networks,
            **network_kwargs,
        ),
        seed=config.seed,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        restore_checkpoint_path=str(checkpoint_path),
   )

   env_cfg = registry.get_default_config(config.env)
   train_env = registry.load(config.env, config=env_cfg)
   eval_env = registry.load(config.env, config=env_cfg)

   print(f"Loading PPO checkpoint from: {checkpoint_path}")
   make_inference_fn, params, _ = train_fn(environment=train_env, eval_env=eval_env)
   inference_fn = make_inference_fn(params, deterministic=True)

   rng = jax.random.PRNGKey(config.seed)

   def policy_fn(observation: np.ndarray) -> np.ndarray:
        nonlocal rng
        rng, act_rng = jax.random.split(rng)
        action, _ = inference_fn(_to_policy_obs(observation), act_rng)
        action_np = np.asarray(action)
        # Brax PPO inference may return a batch dimension of size 1.
        if action_np.ndim > 1 and action_np.shape[0] == 1:
            action_np = action_np[0]
        return action_np

   return policy_fn


def get_dataset(
    dataset, config: DAggerConfig, clip_to_eps: bool = True, eps: float = 1e-5
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


class DAggerTrainState(NamedTuple):
    actor: TrainState
    max_action: float = 1.0


class DAgger(object):
    @classmethod
    def update_actor(
        self,
        train_state: DAggerTrainState,
        batch: Transition,
        rng: jax.Array,
        config: DAggerConfig,
    ) -> Tuple["DAggerTrainState", jnp.ndarray]:
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
        train_state: DAggerTrainState,
        data: Transition,
        rng: jax.Array,
        config: DAggerConfig,
    ) -> Tuple["DAggerTrainState", Dict]:
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
        train_state: DAggerTrainState,
        obs: jnp.ndarray,
        max_action: float = 1.0,  # In D4RL, action is scaled to [-1, 1]
    ) -> jnp.ndarray:
        action = train_state.actor.apply_fn(train_state.actor.params, obs)
        action = action.clip(-max_action, max_action)
        return action

def load_dagger_train_state(
    actions: jnp.ndarray,
    config: DAggerConfig
) -> DAggerTrainState:
    if config.load_model is None:
        raise ValueError("No checkpoint specified for loading.")
    
    checkpoint = np.load(config.load_model, allow_pickle=True)
    actor_params = checkpoint["actor_params"].item()
    
    actor_model = DAggerActor(
        action_dim=actions.shape[-1],
        hidden_dims=config.hidden_dims,
    )
    actor_train_state = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_params,
        tx=optax.adam(config.actor_lr),
    )
    
    return DAggerTrainState(
        actor=actor_train_state,
    )    

def create_dagger_train_state(
    rng: jax.Array,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    config: DAggerConfig,
) -> DAggerTrainState:
    action_dim = actions.shape[-1]
    actor_model = DAggerActor(
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
    return DAggerTrainState(
        actor=actor_train_state,
    )


def get_actor_from_checkpoint(
    checkpoint_path: str,
    state_dim: int,
    action_dim: int,
    max_action: float = 1.0,
) -> Dict[str, Any]:
    """
    Load a DAgger actor from a JAX checkpoint.
    
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
    actor_model = DAggerActor(
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
    record_video: bool = False,
) -> float:  # D4RL specific
    episode_returns = []
    for episode in range(num_episodes):
        render_trajectory.clear()  # Clear previous trajectory
        episode_return = 0
        observation, _ = env.reset()
        done = truncated = False
        while not done and not truncated:
            observation = (observation - obs_mean) / (obs_std + 1e-5)
            action = policy_fn(observation)
            observation, reward, done, truncated, _ = env.step(action)
            if record_video and episode == num_episodes - 1:  # Record video only for the last episode
                env.render()
            episode_return += reward
        episode_returns.append(episode_return)
    
    mean_return = np.mean(episode_returns)
    # Compute normalized score if available (D4RL), otherwise fall back to raw
    if hasattr(env, 'get_normalized_score'):
        normalized_score = env.get_normalized_score(mean_return) * 100 # type: ignore
    else:
        normalized_score = mean_return.item()
    return normalized_score, mean_return.item()

def training_loop(
    config: DAggerConfig,
    update_fn: Callable,
    act_fn: Callable,
    timesteps: int,
    train_state: DAggerTrainState,
    dataset: Transition,
    env: GymWrapper,
    obs_mean,
    obs_std,
    rng: jax.Array,
    index: int = 0, # for DAgger iterations, not used in initial training loop
):
    # Initial training loop
    num_steps = timesteps // config.n_jitted_updates
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
        wandb.log(train_metrics, step=index + i)

        if i % eval_interval == 0:
            policy_fn = partial(act_fn, train_state)
            normalized_score, raw_score = evaluate(
                policy_fn,
                env,
                num_episodes=config.n_episodes,
                obs_mean=obs_mean,
                obs_std=obs_std,
                record_video=config.video_dir is not None,
            )
            eval_metrics = {
                "eval/score": normalized_score,
                "eval/raw_score": raw_score,
            }
            wandb.log(eval_metrics, step=index + i)

            # Save video of the last evaluation episode for this iteration
            if config.video_dir is not None and config.n_episodes > 0:
                video_path = os.path.join(config.video_dir, f"eval/{index + i}.mp4")
                env.save_video(render_trajectory, save_path=video_path)
            
            # Save checkpoint for this iteration
            if config.checkpoints_path is not None:
                checkpoint = {
                    "actor_params": flax.serialization.to_state_dict(train_state.actor.params),
                    "obs_mean": np.array(obs_mean),
                    "obs_std": np.array(obs_std),
                    "step": i,
                }
                checkpoint_path = os.path.join(config.checkpoints_path, f"checkpoint_{index + i}.npz")
                np.savez(checkpoint_path, **checkpoint)
                print(f"Saved checkpoint to {checkpoint_path}")

    return train_state, rng

def collect_data(
    expert_fn: Callable[[np.ndarray], np.ndarray],
    policy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    config: DAggerConfig,
    env: GymWrapper,
    obs_mean,
    obs_std,
    expert_prob: float,
    record_video: bool = False,
) -> Transition:
    # Collect trajectories in parallel and label visited states with expert actions.
    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []
    num_envs = getattr(env, "num_envs", 1)
    collected_episodes = 0

    with tqdm.tqdm(total=config.dagger_episodes, desc="Collecting DAgger data") as pbar:
        while collected_episodes < config.dagger_episodes:
            render_trajectory.clear()  # Keep only latest trajectory for optional video.
            obs, _ = env.reset()
            obs = np.asarray(obs)
            if obs.ndim == 1:
                obs = obs[None, ...]

            done = np.zeros(num_envs, dtype=bool)
            truncated = np.zeros(num_envs, dtype=bool)

            while not np.all(done | truncated):
                obs_normalized = (obs - obs_mean) / (obs_std + 1e-5)

                expert_action = np.asarray(expert_fn(obs))
                if expert_action.ndim == 1:
                    expert_action = expert_action[None, ...]

                policy_action = np.asarray(policy_fn(obs_normalized))
                if policy_action.ndim == 1:
                    policy_action = policy_action[None, ...]

                use_expert = np.random.rand(num_envs, 1) < expert_prob
                behavior_action = np.where(use_expert, expert_action, policy_action)

                next_obs, reward, step_done, step_truncated, _ = env.step(behavior_action)
                if record_video and collected_episodes + num_envs >= config.dagger_episodes:
                    env.render()

                next_obs = np.asarray(next_obs)
                if next_obs.ndim == 1:
                    next_obs = next_obs[None, ...]

                reward = np.asarray(reward, dtype=np.float32).reshape(num_envs)
                step_done = np.asarray(step_done).astype(bool).reshape(num_envs)
                step_truncated = np.asarray(step_truncated).astype(bool).reshape(num_envs)

                active_mask = ~(done | truncated)
                if np.any(active_mask):
                    step_terminal = (step_done | step_truncated).astype(np.float32)
                    observations.append(obs[active_mask])
                    actions.append(expert_action[active_mask])
                    rewards.append(reward[active_mask])
                    next_observations.append(next_obs[active_mask])
                    dones.append(step_terminal[active_mask])

                done |= step_done
                truncated |= step_truncated
                obs = next_obs

            finished = int(np.sum(done | truncated))
            remaining = config.dagger_episodes - collected_episodes
            counted = min(finished, remaining)
            collected_episodes += counted
            pbar.update(counted)
    
    return Transition(
        observations=jnp.array(np.concatenate(observations, axis=0), dtype=jnp.float32),
        actions=jnp.array(np.concatenate(actions, axis=0), dtype=jnp.float32),
        rewards=jnp.array(np.concatenate(rewards, axis=0), dtype=jnp.float32),
        next_observations=jnp.array(np.concatenate(next_observations, axis=0), dtype=jnp.float32),
        dones=jnp.array(np.concatenate(dones, axis=0), dtype=jnp.float32),
    )

def expand_dataset(
    dataset: Transition,
    new_data: Transition,
    obs_mean,
    obs_std,
    rng: jax.Array,
) -> Transition:
    # Normalize only observation tensors; other fields have different shapes.
    new_data = new_data._replace(
        observations=(new_data.observations - obs_mean) / (obs_std + 1e-5),
        next_observations=(new_data.next_observations - obs_mean) / (obs_std + 1e-5),
    )
    
    # Combine old dataset with new data collected from DAgger iterations
    combined_observations = jnp.concatenate([dataset.observations, new_data.observations], axis=0)
    combined_actions = jnp.concatenate([dataset.actions, new_data.actions], axis=0)
    combined_rewards = jnp.concatenate([dataset.rewards, new_data.rewards], axis=0)
    combined_next_observations = jnp.concatenate([dataset.next_observations, new_data.next_observations], axis=0)
    combined_dones = jnp.concatenate([dataset.dones, new_data.dones], axis=0)

    # Shuffle the combined dataset
    data_size = len(combined_observations)
    rng, rng_permute = jax.random.split(rng, 2)
    perm = jax.random.permutation(rng_permute, data_size)
    
    return Transition(
        observations=combined_observations[perm],
        actions=combined_actions[perm],
        rewards=combined_rewards[perm],
        next_observations=combined_next_observations[perm],
        dones=combined_dones[perm],
    )

@pyrallis.wrap()  # type: ignore
def train(config: DAggerConfig):
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
    if not Path(config.expert_checkpoint).exists():
        raise FileNotFoundError(f"Expert checkpoint not found: {config.expert_checkpoint}")

    minari_dataset = minari.load_dataset(config.dataset_id)
    dataset = qlearning_dataset(minari_dataset)
    
    render_callback = None
    if config.video_dir is not None:
        os.makedirs(config.video_dir, exist_ok=True)
        os.makedirs(os.path.join(config.video_dir, "eval"), exist_ok=True)
        os.makedirs(os.path.join(config.video_dir, "collect"), exist_ok=True)
        render_callback = _render_callback
        
    env = get_env(
        config.env, 
        config.device, 
        command_type=config.command_type, 
        render_callback=render_callback
    )
    collect_env = get_env(
        config.env,
        config.device,
        command_type=config.command_type,
        render_callback=render_callback,
        num_actors=config.collect_num_envs,
    )

    rng = jax.random.PRNGKey(config.seed)
    dataset, obs_mean, obs_std = get_dataset(dataset, config)
    # create train_state
    rng, subkey = jax.random.split(rng)
    example_batch: Transition = jax.tree_util.tree_map(lambda x: x[0], dataset)
    if config.load_model is not None:
        print(f"Loading model from checkpoint: {config.load_model}")
        train_state = load_dagger_train_state(example_batch.actions,config)
    else:
        train_state = create_dagger_train_state(
            subkey, example_batch.observations, example_batch.actions, config
        )
    algo = DAgger()
    
    update_fn = jax.jit(algo.update_n_times, static_argnums=(3,))
    act_fn = jax.jit(algo.get_action)

    # Initial training loop
    train_state, rng = training_loop(
        config,
        update_fn,
        act_fn,
        config.initial_timesteps,
        train_state,
        dataset,
        env,
        obs_mean,
        obs_std,
        rng,
    )
    
    # DAgger iterations
    print("Starting DAgger iterations...")
    expert_prob = 1.0
    expert_fn = load_expert(config)
    
    for dagger_iter in range(1, config.dagger_iterations + 1):
        expert_prob *= config.decay_factor
        policy_fn = partial(act_fn, train_state)
        
        # Collect new data with a mixed policy and label with expert actions
        collected_dataset = collect_data(
            expert_fn,
            policy_fn,
            config,
            collect_env,
            obs_mean,
            obs_std,
            expert_prob,
            record_video=config.video_dir is not None,
        )
        dataset = expand_dataset(dataset, collected_dataset, obs_mean, obs_std, rng)
        
        # Save video of the last episode of data collection for this iteration
        if config.video_dir is not None and config.dagger_episodes > 0:
            video_path = os.path.join(config.video_dir, f"collect/{dagger_iter}.mp4")
            collect_env.save_video(render_trajectory, save_path=video_path)
        
        # Train on the expanded dataset
        dagger_index = (config.initial_timesteps + (dagger_iter - 1) * config.dagger_timesteps) // config.n_jitted_updates
        train_state, rng = training_loop(
            config,
            update_fn,
            act_fn,
            config.dagger_timesteps,
            train_state,
            dataset,
            env,
            obs_mean,
            obs_std,
            rng,
            index=dagger_index,
        )

    # final evaluation
    policy_fn = partial(act_fn, train_state)
    normalized_score, raw_score = evaluate(
        policy_fn,
        env,
        num_episodes=config.n_episodes,
        obs_mean=obs_mean,
        obs_std=obs_std,
        record_video=config.video_dir is not None,
    )
    print("Final Evaluation Score:", normalized_score)
    wandb.log({
        "eval/final_score": normalized_score,
        "eval/final_raw_score": raw_score,
    })
    
    # Save final video
    if config.video_dir is not None and config.n_episodes > 0:
        video_path = os.path.join(config.video_dir, f"eval/final.mp4")
        env.save_video(render_trajectory, save_path=video_path)

    # Save final checkpoint
    if config.checkpoints_path is not None:
        num_steps = config.initial_timesteps + config.dagger_iterations * config.dagger_timesteps
        checkpoint = {
            "actor_params": flax.serialization.to_state_dict(train_state.actor.params),
            "obs_mean": np.array(obs_mean),
            "obs_std": np.array(obs_std),
            "step": num_steps,
        }
        checkpoint_path = os.path.join(config.checkpoints_path, f"checkpoint_final.npz")
        np.savez(checkpoint_path, **checkpoint)
        print(f"Saved final checkpoint to {checkpoint_path}")

    wandb.finish()


if __name__ == "__main__":
    train() # type: ignore

