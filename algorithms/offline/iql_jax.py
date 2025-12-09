# source https://github.com/ikostrikov/implicit_q_learning
# https://arxiv.org/abs/2110.06169
import os
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
import minari
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
class IQLConfig:
    # wandb project name
    project: str = "CORL"
    # wandb group name
    group: str = "IQL-JAX"
    # wandb run name
    name: str = "IQL"
    # training dataset and evaluation environment
    env: str = "halfcheetah-medium-expert-v2"
    dataset_id: str = "halfcheetah-medium-expert-v2"
    # discount factor
    discount: float = 0.99
    # coefficient for the target critic Polyak's update
    tau: float = 0.005
    # actor update inverse temperature, similar to AWAC
    beta: float = 3.0
    # coefficient for asymmetric critic loss (expectile)
    iql_tau: float = 0.7
    # total gradient updates during training
    max_timesteps: int = int(1e6)
    # maximum size of the replay buffer
    buffer_size: int = 2_000_000
    # training batch size
    batch_size: int = 256
    # whether to normalize states
    normalize: bool = True
    # whether to normalize reward
    normalize_reward: bool = False
    # V-critic function learning rate
    vf_lr: float = 3e-4
    # Q-critic learning rate
    qf_lr: float = 3e-4
    # actor learning rate
    actor_lr: float = 3e-4
    # evaluation frequency, will evaluate every eval_freq training steps
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
    # JAX-specific: number of jitted updates per step
    n_jitted_updates: int = 8
    # JAX-specific: hidden dimensions for networks
    hidden_dims: Tuple[int, ...] = (256, 256)
    # JAX-specific: whether to use layer norm
    layer_norm: bool = True
    # JAX-specific: whether to use cosine decay schedule for actor
    opt_decay_schedule: bool = True

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

    def __hash__(self):
        # make config hashable to be specified as static_argnums in jax.jit.
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
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1), activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    split_rngs = kwargs.pop("split_rngs", {})
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={**split_rngs, "params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1), layer_norm=self.layer_norm)(observations)
        return jnp.squeeze(critic, -1)


class GaussianPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -5.0
    log_std_max: Optional[float] = 2

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
        )(observations)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init()
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
    dones_float: jnp.ndarray


def get_normalization(dataset: Transition) -> float:
    # into numpy.ndarray
    dataset = jax.tree_util.tree_map(lambda x: np.array(x), dataset)
    returns = []
    ret = 0
    for r, term in zip(dataset.rewards, dataset.dones_float):
        ret += r
        if term:
            returns.append(ret)
            ret = 0
    return (max(returns) - min(returns)) / 1000


def get_dataset(
    config: IQLConfig, clip_to_eps: bool = True, eps: float = 1e-5
) -> Transition:
    dataset = minari.load_dataset(config.dataset_id)
    dataset = qlearning_dataset(dataset)

    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

    dones_float = np.zeros_like(dataset['rewards'])

    for i in range(len(dones_float) - 1):
        if np.linalg.norm(dataset['observations'][i + 1] -
                            dataset['next_observations'][i]
                            ) > 1e-6 or dataset['terminals'][i] == 1.0:
            dones_float[i] = 1
        else:
            dones_float[i] = 0
    dones_float[-1] = 1

    dataset = Transition(
        observations=jnp.array(dataset["observations"], dtype=jnp.float32),
        actions=jnp.array(dataset["actions"], dtype=jnp.float32),
        rewards=jnp.array(dataset["rewards"], dtype=jnp.float32),
        next_observations=jnp.array(dataset["next_observations"], dtype=jnp.float32),
        dones=jnp.array(dataset["terminals"], dtype=jnp.float32),
        dones_float=jnp.array(dones_float, dtype=jnp.float32),
    )
    if "antmaze" in config.env:
        dataset = dataset._replace(
            rewards=dataset.rewards - 1.0
        )
    # normalize states
    obs_mean, obs_std = 0, 1
    if config.normalize:
        obs_mean = dataset.observations.mean(0)
        obs_std = dataset.observations.std(0)
        dataset = dataset._replace(
            observations=(dataset.observations - obs_mean) / (obs_std + 1e-5),
            next_observations=(dataset.next_observations - obs_mean) / (obs_std + 1e-5),
        )
    # normalize rewards
    if config.normalize_reward:    
        normalizing_factor = get_normalization(dataset)
        dataset = dataset._replace(rewards=dataset.rewards / normalizing_factor)
    
    # shuffle data and select the first buffer_size samples
    data_size = min(config.buffer_size, len(dataset.observations))
    rng = jax.random.PRNGKey(config.seed)
    rng, rng_permute, rng_select = jax.random.split(rng, 3)
    perm = jax.random.permutation(rng_permute, len(dataset.observations))
    dataset = jax.tree_util.tree_map(lambda x: x[perm], dataset)
    assert len(dataset.observations) >= data_size
    dataset = jax.tree_util.tree_map(lambda x: x[:data_size], dataset)
    return dataset, obs_mean, obs_std


def expectile_loss(diff, expectile=0.8) -> jnp.ndarray:
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def target_update(
    model: TrainState, target_model: TrainState, tau: float
) -> TrainState:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)


def update_by_loss_grad(
    train_state: TrainState, loss_fn: Callable
) -> Tuple[TrainState, jnp.ndarray]:
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(train_state.params)
    new_train_state = train_state.apply_gradients(grads=grad)
    return new_train_state, loss


class IQLTrainState(NamedTuple):
    rng: jax.random.PRNGKey
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    actor: TrainState


class IQL(object):

    @classmethod
    def update_critic(
        self, train_state: IQLTrainState, batch: Transition, config: IQLConfig
    ) -> Tuple["IQLTrainState", Dict]:
        next_v = train_state.value.apply_fn(
            train_state.value.params, batch.next_observations
        )
        target_q = batch.rewards + config.discount * (1 - batch.dones) * next_v
        
        def critic_loss_fn(
            critic_params: flax.core.FrozenDict[str, Any]
        ) -> jnp.ndarray:
            q1, q2 = train_state.critic.apply_fn(
                critic_params, batch.observations, batch.actions
            )
            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
            return critic_loss

        new_critic, critic_loss = update_by_loss_grad(
            train_state.critic, critic_loss_fn
        )
        return train_state._replace(critic=new_critic), critic_loss

    @classmethod
    def update_value(
        self, train_state: IQLTrainState, batch: Transition, config: IQLConfig
    ) -> Tuple["IQLTrainState", Dict]:
        q1, q2 = train_state.target_critic.apply_fn(
            train_state.target_critic.params, batch.observations, batch.actions
        )
        q = jax.lax.stop_gradient(jnp.minimum(q1, q2))
        def value_loss_fn(value_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
            v = train_state.value.apply_fn(value_params, batch.observations)
            value_loss = expectile_loss(q - v, config.iql_tau).mean()
            return value_loss

        new_value, value_loss = update_by_loss_grad(train_state.value, value_loss_fn)
        return train_state._replace(value=new_value), value_loss

    @classmethod
    def update_actor(
        self, train_state: IQLTrainState, batch: Transition, config: IQLConfig
    ) -> Tuple["IQLTrainState", Dict]:
        v = train_state.value.apply_fn(train_state.value.params, batch.observations)
        q1, q2 = train_state.critic.apply_fn(
            train_state.target_critic.params, batch.observations, batch.actions
        )
        q = jnp.minimum(q1, q2)
        exp_a = jnp.exp((q - v) * config.beta)
        exp_a = jnp.minimum(exp_a, 100.0)
        def actor_loss_fn(actor_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
            dist = train_state.actor.apply_fn(actor_params, batch.observations)
            log_probs = dist.log_prob(batch.actions)
            actor_loss = -(exp_a * log_probs).mean()
            return actor_loss

        new_actor, actor_loss = update_by_loss_grad(train_state.actor, actor_loss_fn)
        return train_state._replace(actor=new_actor), actor_loss

    @classmethod
    def update_n_times(
        self,
        train_state: IQLTrainState,
        dataset: Transition,
        rng: jax.random.PRNGKey,
        config: IQLConfig,
    ) -> Tuple["IQLTrainState", Dict]:
        for _ in range(config.n_jitted_updates):
            rng, subkey = jax.random.split(rng)
            batch_indices = jax.random.randint(
                subkey, (config.batch_size,), 0, len(dataset.observations)
            )
            batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)

            train_state, value_loss = self.update_value(train_state, batch, config)
            train_state, actor_loss = self.update_actor(train_state, batch, config)
            train_state, critic_loss = self.update_critic(train_state, batch, config)
            new_target_critic = target_update(
                train_state.critic, train_state.target_critic, config.tau
            )
            train_state = train_state._replace(target_critic=new_target_critic)
        return train_state, {
            "value_loss": value_loss,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
        }

    @classmethod
    def get_action(
        self,
        train_state: IQLTrainState,
        observations: np.ndarray,
        seed: jax.random.PRNGKey,
        temperature: float = 1.0,
        max_action: float = 1.0,  # In D4RL, the action space is [-1, 1]
    ) -> jnp.ndarray:
        actions = train_state.actor.apply_fn(
            train_state.actor.params, observations, temperature=temperature
        ).sample(seed=seed)
        actions = jnp.clip(actions, -max_action, max_action)
        return actions


def create_iql_train_state(
    rng: jax.random.PRNGKey,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    config: IQLConfig,
) -> IQLTrainState:
    rng, actor_rng, critic_rng, value_rng = jax.random.split(rng, 4)
    # initialize actor
    action_dim = actions.shape[-1]
    actor_model = GaussianPolicy(
        config.hidden_dims,
        action_dim=action_dim,
        log_std_min=-5.0,
    )
    if config.opt_decay_schedule:
        schedule_fn = optax.cosine_decay_schedule(-config.actor_lr, config.max_timesteps)
        actor_tx = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn))
    else:
        actor_tx = optax.adam(learning_rate=config.actor_lr)
    actor = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, observations),
        tx=actor_tx,
    )
    # initialize critic
    critic_model = ensemblize(Critic, num_qs=2)(config.hidden_dims)
    critic = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions),
        tx=optax.adam(learning_rate=config.qf_lr),
    )
    target_critic = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions),
        tx=optax.adam(learning_rate=config.qf_lr),
    )
    # initialize value
    value_model = ValueCritic(config.hidden_dims, layer_norm=config.layer_norm)
    value = TrainState.create(
        apply_fn=value_model.apply,
        params=value_model.init(value_rng, observations),
        tx=optax.adam(learning_rate=config.vf_lr),
    )
    return IQLTrainState(
        rng,
        critic=critic,
        target_critic=target_critic,
        value=value,
        actor=actor,
    )


def evaluate(
    policy_fn, env: gym.Env, num_episodes: int, obs_mean: float, obs_std: float
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
    return np.mean(episode_returns)


@pyrallis.wrap()
def train(config: IQLConfig):
    wandb.init(
        config=config,
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
    )
    wandb.run.save()

    # Setup checkpoints
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    rng = jax.random.PRNGKey(config.seed)
    env = get_env(config.env, config.device)
    dataset, obs_mean, obs_std = get_dataset(config)
    
    # create train_state
    rng, subkey = jax.random.split(rng)
    example_batch: Transition = jax.tree_util.tree_map(lambda x: x[0], dataset)
    train_state: IQLTrainState = create_iql_train_state(
        subkey,
        example_batch.observations,
        example_batch.actions,
        config,
    )

    algo = IQL()
    update_fn = jax.jit(algo.update_n_times, static_argnums=(3,))
    act_fn = jax.jit(algo.get_action)
    num_steps = config.max_timesteps // config.n_jitted_updates
    eval_interval = config.eval_freq // config.n_jitted_updates

    for i in tqdm.tqdm(range(1, num_steps + 1), smoothing=0.1, dynamic_ncols=True):
        rng, subkey = jax.random.split(rng)
        train_state, update_info = update_fn(train_state, dataset, subkey, config)

        current_step = i * config.n_jitted_updates
        train_metrics = {f"training/{k}": v for k, v in update_info.items()}
        wandb.log(train_metrics, step=current_step)

        if i % eval_interval == 0:
            policy_fn = partial(
                act_fn,
                temperature=0.0,
                seed=jax.random.PRNGKey(0),
                train_state=train_state,
            )
            eval_score = evaluate(
                policy_fn,
                env,
                num_episodes=config.n_episodes,
                obs_mean=obs_mean,
                obs_std=obs_std,
            )
            print(f"Step: {current_step}, Eval Score: {eval_score}")
            eval_metrics = {f"{config.env}/eval_score": eval_score}
            wandb.log(eval_metrics, step=current_step)

            # Save checkpoint
            if config.checkpoints_path is not None:
                checkpoint = {
                    "actor_params": flax.serialization.to_state_dict(train_state.actor.params),
                    "critic_params": flax.serialization.to_state_dict(train_state.critic.params),
                    "target_critic_params": flax.serialization.to_state_dict(train_state.target_critic.params),
                    "value_params": flax.serialization.to_state_dict(train_state.value.params),
                    "obs_mean": np.array(obs_mean),
                    "obs_std": np.array(obs_std),
                    "step": current_step,
                }
                checkpoint_path = os.path.join(config.checkpoints_path, f"checkpoint_{current_step}.npz")
                np.savez(checkpoint_path, **checkpoint)
                print(f"Saved checkpoint to {checkpoint_path}")

    # final evaluation
    policy_fn = partial(
        act_fn,
        temperature=0.0,
        seed=jax.random.PRNGKey(0),
        train_state=train_state,
    )
    final_score = evaluate(
        policy_fn,
        env,
        num_episodes=config.n_episodes,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )
    print("Final Evaluation", final_score)
    wandb.log({f"{config.env}/final_eval_score": final_score})

    # Save final checkpoint
    if config.checkpoints_path is not None:
        checkpoint = {
            "actor_params": flax.serialization.to_state_dict(train_state.actor.params),
            "critic_params": flax.serialization.to_state_dict(train_state.critic.params),
            "target_critic_params": flax.serialization.to_state_dict(train_state.target_critic.params),
            "value_params": flax.serialization.to_state_dict(train_state.value.params),
            "obs_mean": np.array(obs_mean),
            "obs_std": np.array(obs_std),
            "step": config.max_timesteps,
        }
        checkpoint_path = os.path.join(config.checkpoints_path, "checkpoint_final.npz")
        np.savez(checkpoint_path, **checkpoint)
        print(f"Saved final checkpoint to {checkpoint_path}")

    wandb.finish()


if __name__ == "__main__":
    train()
