# source https://github.com/ikostrikov/jaxrl
# https://arxiv.org/abs/2006.09359
import os
import time
import uuid
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import distrax
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

import minari

from algorithms.utils.wrapper_gym import get_env
from algorithms.utils.dataset import qlearning_dataset

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"


@dataclass
class AWACConfig:
    # wandb project name
    project: str = "CORL"
    # wandb group name
    group: str = "AWAC-D4RL"
    # wandb run name
    name: str = "AWAC"
    # training dataset and evaluation environment
    env: str = "halfcheetah-medium-expert-v2"
    dataset_id: str = "halfcheetah-medium-expert-v2"
    # actor and critic hidden dim
    hidden_dim: int = 256
    # actor and critic learning rate
    learning_rate: float = 3e-4
    # discount factor
    gamma: float = 0.99
    # coefficient for the target critic Polyak's update
    tau: float = 5e-3
    # awac actor loss temperature, controlling balance
    # between behaviour cloning and Q-value maximization
    awac_lambda: float = 1.0
    # total number of gradient updates during training
    max_timesteps: int = 1_000_000
    # training batch size
    batch_size: int = 256
    # maximum size of the replay buffer
    buffer_size: int = 2_000_000
    # whether to normalize reward (like in IQL)
    normalize_reward: bool = False
    # whether to normalize states
    normalize_state: bool = True
    # evaluation frequency, will evaluate every eval_freq training steps
    eval_freq: int = int(5e3)
    # number of episodes to run during evaluation
    n_episodes: int = 10
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = None
    # training random seed
    seed: int = 0
    # training device
    device: str = "cuda"
    # NETWORK (JAX specific)
    actor_hidden_dims: Tuple[int, ...] = (256, 256, 256, 256)
    critic_hidden_dims: Tuple[int, ...] = (256, 256)
    # JAX specific
    n_jitted_updates: int = 8

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

    def __hash__(
        self,
    ):  # make config hashable to be specified as static_argnums in jax.jit.
        return hash(self.__repr__())


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
        for i, hidden_dims in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dims, kernel_init=self.kernel_init)(x)
            if self.add_layer_norm:  # Add layer norm after activation
                if self.layer_norm_final or i + 1 < len(self.hidden_dims):
                    x = nn.LayerNorm()(x)
            if (
                i + 1 < len(self.hidden_dims) or self.activate_final
            ):  # Add activation after layer norm
                x = self.activations(x)
        return x


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(
        self, observation: jnp.ndarray, action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.concatenate([observation, action], axis=-1)
        q1 = MLP((*self.hidden_dims, 1), add_layer_norm=True)(x)
        q2 = MLP((*self.hidden_dims, 1), add_layer_norm=True)(x)
        return q1, q2


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
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
        )(observations)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)

        log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        return distribution


class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray


def get_dataset(
    dataset: Dict, config: AWACConfig, clip_to_eps: bool = True, eps: float = 1e-5
) -> Transition:
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
    if config.normalize_state:
        obs_mean = dataset.observations.mean(0)
        obs_std = dataset.observations.std(0)
        dataset = dataset._replace(
            observations=(dataset.observations - obs_mean) / (obs_std + 1e-5),
            next_observations=(dataset.next_observations - obs_mean) / (obs_std + 1e-5),
        )
    return dataset, obs_mean, obs_std


def target_update(
    model: TrainState, target_model: TrainState, tau: float
) -> Tuple[TrainState, jnp.ndarray]:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)


def update_by_loss_grad(
    train_state: TrainState, loss_fn: Callable
) -> Tuple[float, Any]:
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(train_state.params)
    new_train_state = train_state.apply_gradients(grads=grad)
    return new_train_state, loss


class AWACTrainState(NamedTuple):
    rng: jax.random.PRNGKey
    critic: TrainState
    target_critic: TrainState
    actor: TrainState


class AWAC(object):

    @classmethod
    def update_actor(
        self,
        train_state: AWACTrainState,
        batch: Transition,
        rng: jax.random.PRNGKey,
        config: AWACConfig,
    ) -> Tuple["AWACTrainState", jnp.ndarray]:
        def get_actor_loss(actor_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
            dist = train_state.actor.apply_fn(actor_params, batch.observations)
            pi_actions = dist.sample(seed=rng)
            q_1, q_2 = train_state.critic.apply_fn(
                train_state.critic.params, batch.observations, pi_actions
            )
            v = jnp.minimum(q_1, q_2)

            lim = 1 - 1e-5
            actions = jnp.clip(batch.actions, -lim, lim)
            q_1, q_2 = train_state.critic.apply_fn(
                train_state.critic.params, batch.observations, actions
            )
            q = jnp.minimum(q_1, q_2)
            adv = q - v
            weights = jnp.exp(adv / config.awac_lambda)

            weights = jax.lax.stop_gradient(weights)

            log_prob = dist.log_prob(batch.actions)
            loss = -jnp.mean(log_prob * weights).mean()
            return loss

        new_actor, actor_loss = update_by_loss_grad(train_state.actor, get_actor_loss)
        return train_state._replace(actor=new_actor), actor_loss

    @classmethod
    def update_critic(
        self,
        train_state: AWACTrainState,
        batch: Transition,
        rng: jax.random.PRNGKey,
        config: AWACConfig,
    ) -> Tuple["AWACTrainState", jnp.ndarray]:
        def get_critic_loss(
            critic_params: flax.core.FrozenDict[str, Any]
        ) -> jnp.ndarray:
            dist = train_state.actor.apply_fn(
                train_state.actor.params, batch.observations
            )
            next_actions = dist.sample(seed=rng)
            n_q_1, n_q_2 = train_state.target_critic.apply_fn(
                train_state.target_critic.params, batch.next_observations, next_actions
            )
            next_q = jnp.minimum(n_q_1, n_q_2)
            q_target = batch.rewards + config.gamma * (1 - batch.dones) * next_q
            q_target = jax.lax.stop_gradient(q_target)

            q_1, q_2 = train_state.critic.apply_fn(
                critic_params, batch.observations, batch.actions
            )

            loss = jnp.mean((q_1 - q_target) ** 2 + (q_2 - q_target) ** 2)
            return loss

        new_critic, critic_loss = update_by_loss_grad(
            train_state.critic, get_critic_loss
        )
        return train_state._replace(critic=new_critic), critic_loss

    @classmethod
    def update_n_times(
        self,
        train_state: AWACTrainState,
        dataset: Transition,
        rng: jax.random.PRNGKey,
        config: AWACConfig,
    ) -> Tuple["AWACTrainState", Dict]:
        for _ in range(config.n_jitted_updates):
            rng, batch_rng, critic_rng, actor_rng = jax.random.split(rng, 4)
            batch_indices = jax.random.randint(
                batch_rng, (config.batch_size,), 0, len(dataset.observations)
            )
            batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)

            train_state, critic_loss = self.update_critic(
                train_state, batch, critic_rng, config
            )
            new_target_critic = target_update(
                train_state.critic,
                train_state.target_critic,
                config.tau,
            )
            train_state, actor_loss = self.update_actor(
                train_state, batch, actor_rng, config
            )
        return train_state._replace(target_critic=new_target_critic), {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
        }

    @classmethod
    def get_action(
        self,
        train_state: AWACTrainState,
        observations: np.ndarray,
        seed: jax.random.PRNGKey,
        temperature: float = 1.0,
        max_action: float = 1.0,  # In D4RL envs, the action space is [-1, 1]
    ) -> jnp.ndarray:
        actions = train_state.actor.apply_fn(
            train_state.actor.params, observations=observations, temperature=temperature
        ).sample(seed=seed)
        actions = jnp.clip(actions, -max_action, max_action)
        return actions


def create_awac_train_state(
    rng: jax.random.PRNGKey,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    config: AWACConfig,
) -> AWACTrainState:
    rng, actor_rng, critic_rng, value_rng = jax.random.split(rng, 4)
    # initialize actor
    action_dim = actions.shape[-1]
    actor_model = GaussianPolicy(
        config.actor_hidden_dims,
        action_dim=action_dim,
    )
    actor = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, observations),
        tx=optax.adam(learning_rate=config.learning_rate),
    )
    # initialize critic
    critic_model = DoubleCritic(config.critic_hidden_dims)
    critic = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions),
        tx=optax.adam(learning_rate=config.learning_rate),
    )
    # initialize target critic
    target_critic = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions),
        tx=optax.adam(learning_rate=config.learning_rate),
    )
    return AWACTrainState(
        rng,
        critic=critic,
        target_critic=target_critic,
        actor=actor,
    )


def evaluate(
    policy_fn: Callable,
    env: gym.Env,
    num_episodes: int,
    obs_mean: float,
    obs_std: float,
) -> float:
    episode_returns = []
    for _ in range(num_episodes):
        episode_return = 0
        observation, _ = env.reset()
        done = truncated = False
        while not done and not truncated:
            observation = (observation - obs_mean) / (obs_std + 1e-5)
            action = policy_fn(observations=observation)
            observation, reward, done, truncated, info = env.step(np.array(action))
            episode_return += reward
        episode_returns.append(episode_return)
    mean_return = np.mean(episode_returns)
    # Use normalized score if available, otherwise return raw score
    if hasattr(env, 'get_normalized_score'):
        return env.get_normalized_score(mean_return) * 100
    else:
        return mean_return


@pyrallis.wrap()
def train(config: AWACConfig):
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

    rng = jax.random.PRNGKey(config.seed)
    
    minari_dataset = minari.load_dataset(config.dataset_id)
    qdataset = qlearning_dataset(minari_dataset)
    env = get_env(config.env, config.device)
    
    dataset, obs_mean, obs_std = get_dataset(qdataset, config)
    
    # create train_state
    rng, subkey = jax.random.split(rng)
    example_batch: Transition = jax.tree_util.tree_map(lambda x: x[0], dataset)
    train_state: AWACTrainState = create_awac_train_state(
        subkey,
        example_batch.observations,
        example_batch.actions,
        config,
    )
    algo = AWAC()
    update_fn = jax.jit(algo.update_n_times, static_argnums=(3,))
    act_fn = jax.jit(algo.get_action)

    num_steps = config.max_timesteps // config.n_jitted_updates
    eval_interval = config.eval_freq // config.n_jitted_updates
    for i in tqdm.tqdm(range(1, num_steps + 1), smoothing=0.1, dynamic_ncols=True):
        rng, subkey = jax.random.split(rng)
        train_state, update_info = update_fn(
            train_state,
            dataset,
            subkey,
            config,
        )
        train_metrics = {f"training/{k}": v for k, v in update_info.items()}
        wandb.log(train_metrics, step=i)

        if i % eval_interval == 0:
            policy_fn = partial(
                act_fn,
                temperature=0.0,
                seed=jax.random.PRNGKey(0),
                train_state=train_state,
            )
            normalized_score = evaluate(
                policy_fn, env, config.n_episodes, obs_mean, obs_std
            )
            print(f"Step {i}: {normalized_score}")
            eval_metrics = {"eval/score": normalized_score}
            wandb.log(eval_metrics, step=i)

            if config.checkpoints_path is not None:
                checkpoint = {
                    "actor_params": flax.serialization.to_state_dict(train_state.actor.params),
                    "critic_params": flax.serialization.to_state_dict(train_state.critic.params),
                    "target_critic_params": flax.serialization.to_state_dict(train_state.target_critic.params),
                    "obs_mean": np.array(obs_mean),
                    "obs_std": np.array(obs_std),
                    "step": i,
                }
                checkpoint_path = os.path.join(config.checkpoints_path, f"checkpoint_{i}.npz")
                np.savez(checkpoint_path, **checkpoint)
                print(f"Saved checkpoint to {checkpoint_path}")

    # final evaluation
    policy_fn = partial(
        act_fn,
        temperature=0.0,
        seed=jax.random.PRNGKey(0),
        train_state=train_state,
    )
    normalized_score = evaluate(policy_fn, env, config.n_episodes, obs_mean, obs_std)
    print("Final Evaluation Score:", normalized_score)
    wandb.log({"eval/final_score": normalized_score})

    # Save final checkpoint
    if config.checkpoints_path is not None:
        checkpoint = {
            "actor_params": flax.serialization.to_state_dict(train_state.actor.params),
            "critic_params": flax.serialization.to_state_dict(train_state.critic.params),
            "target_critic_params": flax.serialization.to_state_dict(train_state.target_critic.params),
            "obs_mean": np.array(obs_mean),
            "obs_std": np.array(obs_std),
            "step": num_steps,
        }
        checkpoint_path = os.path.join(config.checkpoints_path, f"checkpoint_final.npz")
        np.savez(checkpoint_path, **checkpoint)
        print(f"Saved final checkpoint to {checkpoint_path}")

    wandb.finish()


if __name__ == "__main__":
    train()
