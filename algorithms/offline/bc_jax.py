# Behavioral Cloning implementation in JAX
# Simple supervised learning approach for offline RL
import os
import time
import uuid
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import minari
import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyrallis
import tqdm
import wandb
from flax.training.train_state import TrainState

from algorithms.utils.wrapper_gym import get_env
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
    command_type: str = None
    # total gradient updates during training
    max_timesteps: int = int(1e6)
    # training batch size
    batch_size: int = 256
    # maximum size of the replay buffer
    buffer_size: int = 2_000_000
    # what top fraction of the dataset (sorted by return) to use
    frac: float = 0.1
    # maximum possible trajectory length
    max_traj_len: int = 1000
    # whether to normalize states
    normalize: bool = True
    # discount factor
    discount: float = 0.99
    # evaluation frequency, will evaluate eval_freq training steps
    eval_freq: int = int(5e3)
    # number of episodes to run during evaluation
    n_episodes: int = 10
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = None
    # file name for loading a model, optional
    load_model: str = ""
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
) -> Transition:
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
    rng, rng_permute, rng_select = jax.random.split(rng, 3)
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
        rng: jax.random.PRNGKey,
        config: BCConfig,
    ) -> Tuple["BCTrainState", jnp.ndarray]:
        def actor_loss_fn(actor_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
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
        rng: jax.random.PRNGKey,
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


def create_bc_train_state(
    rng: jax.random.PRNGKey,
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


def evaluate(
    policy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    env: gym.Env,
    num_episodes: int,
    obs_mean,
    obs_std,
) -> float:  # D4RL specific
    episode_returns = []
    for _ in range(num_episodes):
        episode_return = 0
        observation, _ = env.reset()
        done = truncated = False
        while not done and not truncated:
            observation = (observation - obs_mean) / obs_std
            action = policy_fn(obs=observation)
            observation, reward, done, truncated, info = env.step(action)
            episode_return += reward
        episode_returns.append(episode_return)
    
    mean_return = np.mean(episode_returns)
    # Use normalized score if available (D4RL), otherwise return raw score
    if hasattr(env, 'get_normalized_score'):
        return env.get_normalized_score(mean_return) * 100
    else:
        return mean_return

@pyrallis.wrap()
def train(config: BCConfig):
    wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=config,
        id=str(uuid.uuid4()),
    )

    minari_dataset = minari.load_dataset(config.dataset_id)
    dataset = qlearning_dataset(minari_dataset)
    env = get_env(config.env, config.device, command_type=config.command_type)

    rng = jax.random.PRNGKey(config.seed)
    dataset, obs_mean, obs_std = get_dataset(dataset, config)
    # create train_state
    rng, subkey = jax.random.split(rng)
    example_batch: Transition = jax.tree_util.tree_map(lambda x: x[0], dataset)
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
            policy_fn = partial(act_fn, train_state=train_state)
            normalized_score = evaluate(
                policy_fn,
                env,
                num_episodes=config.n_episodes,
                obs_mean=obs_mean,
                obs_std=obs_std,
            )
            eval_metrics = {f"eval/score": normalized_score}
            wandb.log(eval_metrics, step=i)

    # final evaluation
    policy_fn = partial(act_fn, train_state=train_state)
    normalized_score = evaluate(
        policy_fn,
        env,
        num_episodes=config.n_episodes,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )
    print("Final Evaluation Score:", normalized_score)
    wandb.log({f"eval/final_score": normalized_score})
    wandb.finish()


if __name__ == "__main__":
    train()

