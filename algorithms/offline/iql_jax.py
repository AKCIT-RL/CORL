# source https://github.com/ikostrikov/implicit_q_learning
# https://arxiv.org/abs/2110.06169
import os
import uuid
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import distrax
import flax.linen as nn
from pyrallis.argparsing import wrap as pyrallis_wrap
from pyrallis.cfgparsing import dump as pyrallis_dump
import jax
import jax.numpy as jnp
import minari
import numpy as np
import optax
import tqdm
import wandb
import yaml
from flax import serialization as flax_serialization
from flax import core as flax_core
from flax.training.train_state import TrainState

from algorithms.utils import proxy
from algorithms.utils.randomize_gym import GymWrapper, get_env, maybe_get_shifted_env, record_policy_video
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
    n_eval_actors: int = 10
    # number of episodes for the final evaluation (larger -> lower variance)
    n_eval_episodes_final: int = 50
    # fixed seed for evaluation rollouts (reproducible / comparable)
    eval_seed: int = 0
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = None
    # save an intermediate checkpoint every N evaluations (final always saved)
    checkpoints_every: int = 10
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

    command_type: Optional[str] = None
    # Optional Tier-5 shifted evaluation: JSON string of flattened env-config
    # overrides (e.g. stronger push-recovery kicks) applied only to a second eval
    # env. Normalization reuses the in-distribution dataset refs, so the shifted
    # score is comparable to eval/final_score (their gap = robustness gap).
    eval_shift: Optional[str] = None
    # JAX-specific: whether to use deterministic actor
    iql_deterministic: bool = False

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

    def __hash__(self):
        # make config hashable to be specified as static_argnums in jax.jit.
        return hash(self.__repr__())


def default_init(scale: Optional[float] = float(jnp.sqrt(2))):
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
        in_axes=None, # type: ignore
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
) -> Tuple[Transition, Any, Any]:
    dataset = minari.load_dataset(config.dataset_id)
    dataset = qlearning_dataset(dataset)

    if clip_to_eps:
        lim = 1 - eps
        dataset.actions = np.clip(dataset.actions, -lim, lim)

    dones_float = np.zeros_like(dataset.rewards)

    for i in range(len(dones_float) - 1):
        if np.linalg.norm(dataset.observations[i + 1] -
                            dataset.next_observations[i]
                            ) > 1e-6 or dataset.terminals[i] == 1.0:
            dones_float[i] = 1
        else:
            dones_float[i] = 0
    dones_float[-1] = 1

    transition = Transition(
        observations=jnp.array(dataset.observations, dtype=jnp.float32),
        actions=jnp.array(dataset.actions, dtype=jnp.float32),
        rewards=jnp.array(dataset.rewards, dtype=jnp.float32),
        next_observations=jnp.array(dataset.next_observations, dtype=jnp.float32),
        dones=jnp.array(dataset.terminals, dtype=jnp.float32),
        dones_float=jnp.array(dones_float, dtype=jnp.float32),
    )
    if "antmaze" in config.env:
        transition = transition._replace(
            rewards=transition.rewards - 1.0
        )
    obs_mean, obs_std = 0, 1
    if config.normalize:
        obs_mean = transition.observations.mean(0)
        obs_std = transition.observations.std(0)
        transition = transition._replace(
            observations=(transition.observations - obs_mean) / (obs_std + 1e-5),
            next_observations=(transition.next_observations - obs_mean) / (obs_std + 1e-5),
        )
    # normalize rewards
    if config.normalize_reward:    
        normalizing_factor = get_normalization(transition)
        transition = transition._replace(rewards=transition.rewards / normalizing_factor)
    
    # shuffle data and select the first buffer_size samples
    data_size = min(config.buffer_size, len(transition.observations))
    rng = jax.random.PRNGKey(config.seed)
    rng, rng_permute, rng_select = jax.random.split(rng, 3)
    perm = jax.random.permutation(rng_permute, len(transition.observations))
    transition = jax.tree_util.tree_map(lambda x: x[perm], transition)
    assert len(transition.observations) >= data_size
    transition = jax.tree_util.tree_map(lambda x: x[:data_size], transition)
    return transition, obs_mean, obs_std


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
    rng: jax.Array
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    actor: TrainState


class IQL(object):

    @classmethod
    def update_critic(
        cls, train_state: IQLTrainState, batch: Transition, config: IQLConfig
    ) -> Tuple["IQLTrainState", jnp.ndarray]:
        next_v = train_state.value.apply_fn(
            train_state.value.params, batch.next_observations
        )
        target_q = batch.rewards + config.discount * (1 - batch.dones) * next_v
        
        def critic_loss_fn(
            critic_params: flax_core.FrozenDict[str, Any]
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
        cls, train_state: IQLTrainState, batch: Transition, config: IQLConfig
    ) -> Tuple["IQLTrainState", jnp.ndarray]:
        q1, q2 = train_state.target_critic.apply_fn(
            train_state.target_critic.params, batch.observations, batch.actions
        )
        q = jax.lax.stop_gradient(jnp.minimum(q1, q2))
        def value_loss_fn(value_params: flax_core.FrozenDict[str, Any]) -> jnp.ndarray:
            v = train_state.value.apply_fn(value_params, batch.observations)
            value_loss = expectile_loss(q - v, config.iql_tau).mean()
            return value_loss

        new_value, value_loss = update_by_loss_grad(train_state.value, value_loss_fn)
        return train_state._replace(value=new_value), value_loss

    @classmethod
    def update_actor(
        cls, train_state: IQLTrainState, batch: Transition, config: IQLConfig
    ) -> Tuple["IQLTrainState", jnp.ndarray]:
        v = train_state.value.apply_fn(train_state.value.params, batch.observations)
        q1, q2 = train_state.critic.apply_fn(
            train_state.target_critic.params, batch.observations, batch.actions
        )
        q = jnp.minimum(q1, q2)
        exp_a = jnp.exp((q - v) * config.beta)
        exp_a = jnp.minimum(exp_a, 100.0)
        def actor_loss_fn(actor_params: flax_core.FrozenDict[str, Any]) -> jnp.ndarray:
            dist = train_state.actor.apply_fn(actor_params, batch.observations)
            log_probs = dist.log_prob(batch.actions)
            actor_loss = -(exp_a * log_probs).mean()
            return actor_loss

        new_actor, actor_loss = update_by_loss_grad(train_state.actor, actor_loss_fn)
        return train_state._replace(actor=new_actor), actor_loss

    @classmethod
    def update_n_times(
        cls,
        train_state: IQLTrainState,
        dataset: Transition,
        rng: jax.Array,
        config: IQLConfig,
    ) -> Tuple["IQLTrainState", Dict]:
        for _ in range(config.n_jitted_updates):
            rng, subkey = jax.random.split(rng)
            batch_indices = jax.random.randint(
                subkey, (config.batch_size,), 0, len(dataset.observations)
            )
            batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)

            train_state, value_loss = cls.update_value(train_state, batch, config)
            train_state, actor_loss = cls.update_actor(train_state, batch, config)
            train_state, critic_loss = cls.update_critic(train_state, batch, config)
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
        cls,
        train_state: IQLTrainState,
        observations: np.ndarray,
        seed: jax.Array,
        temperature: float = 1.0,
        max_action: float = 1.0,  # In D4RL, the action space is [-1, 1]
    ) -> jnp.ndarray:
        actions = train_state.actor.apply_fn(
            train_state.actor.params, observations, temperature=temperature
        ).sample(seed=seed)
        actions = jnp.clip(actions, -max_action, max_action)
        return actions


def create_iql_train_state(
    rng: jax.Array,
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
    policy_fn: Callable, 
    env: GymWrapper, 
    num_episodes: int, 
    obs_mean: float, 
    obs_std: float, 
    seed: int = 0
) -> Tuple[float, float]:
    env.reset_rng(seed)
    num_envs = getattr(env, "num_envs", 1)
    episode_returns = []

    while len(episode_returns) < num_episodes:
        episode_return = np.zeros(num_envs, dtype=np.float32)
        finished = np.zeros(num_envs, dtype=bool)
        observation, _ = env.reset()
        while not np.all(finished):
            observation = (observation - obs_mean) / (obs_std + 1e-5)
            action = policy_fn(observations=observation)
            observation, reward, done, truncated, _ = env.step(np.array(action))
            active_mask = ~finished
            episode_return += np.asarray(reward, dtype=np.float32) * active_mask
            finished |= np.asarray(done, dtype=bool) | np.asarray(truncated, dtype=bool)
        completed = min(num_envs, num_episodes - len(episode_returns))
        episode_returns.extend(episode_return[:completed].tolist())

    mean_return = float(np.mean(episode_returns))
    # Normalize using the env's D4RL-style reference scores (loaded from the
    # dataset metadata). Falls back to the raw return when refs are absent.
    normalized = env.get_normalized_score(mean_return)
    normalized_score = normalized * 100 if normalized is not None else mean_return
    return normalized_score, mean_return


def get_actor_from_checkpoint(
    checkpoint_path: str,
    state_dim: int,
    action_dim: int,
    max_action: float = 1.0,
) -> Dict[str, Any]:
    """
    Load an IQL actor from a JAX checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory
        state_dim: Dimension of the state/observation space
        action_dim: Dimension of the action space
        max_action: Maximum action value (default: 1.0)

    Returns:
        Dictionary containing:
            - 'actor_fn': JIT-compiled function that takes (obs, seed, temperature) and
                          returns a sampled action
            - 'get_action': Convenience wrapper that applies obs normalisation and returns
                            a numpy action. Uses temperature=0.0 (deterministic) by default.
            - 'actor_params': The loaded model parameters
            - 'obs_mean': Observation mean for normalisation
            - 'obs_std': Observation std for normalisation
            - 'config': The loaded config dictionary
    """
    ckpt_path = Path(checkpoint_path).resolve()

    # Load config if available
    hidden_dims = (256, 256)
    config = None
    config_file = ckpt_path / "config.yaml"
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
        if config and "hidden_dims" in config:
            hidden_dims = tuple(config["hidden_dims"])

    # Find checkpoint files
    ckpt_files = list(ckpt_path.glob("checkpoint_*.npz"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_path}")

    def get_step(p):
        step_str = p.stem.split("_")[-1]
        return float("inf") if step_str == "final" else int(step_str)

    ckpt_files.sort(key=get_step)

    final_ckpt = ckpt_path / "checkpoint_final.npz"
    selected_ckpt = final_ckpt if final_ckpt.exists() else ckpt_files[-1]

    print(f"[get_actor_from_checkpoint] Loading checkpoint: {selected_ckpt}")

    # Load checkpoint
    checkpoint = np.load(selected_ckpt, allow_pickle=True)
    actor_params_dict = checkpoint["actor_params"].item()
    obs_mean = checkpoint["obs_mean"]
    obs_std = checkpoint["obs_std"]

    # Reconstruct actor model
    actor_model = GaussianPolicy(
        hidden_dims=hidden_dims,
        action_dim=action_dim,
    )

    # Restore params into a proper FrozenDict
    dummy_obs = jnp.zeros((1, state_dim))
    actor_params = flax_serialization.from_state_dict(
        actor_model.init(jax.random.PRNGKey(0), dummy_obs),
        actor_params_dict,
    )

    # JIT-compiled sampling function
    @jax.jit
    def actor_fn(
        obs: jnp.ndarray,
        seed: jax.Array,
        temperature: float = 1.0,
    ) -> jnp.ndarray:
        dist, _ = actor_model.apply(actor_params, obs, temperature=temperature)
        actions = dist.sample(seed=seed)
        return jnp.clip(actions, -max_action, max_action)

    # Convenience wrapper: normalises obs, uses temperature=0.0 for deterministic actions
    def get_action(
        obs: np.ndarray,
        seed: Optional[jax.Array] = None,
        temperature: float = 0.0,
    ) -> np.ndarray:
        if seed is None:
            seed = jax.random.PRNGKey(0)
        obs_normalized = (obs - obs_mean) / (obs_std + 1e-5)
        action = actor_fn(jnp.array(obs_normalized), seed=seed, temperature=temperature)
        return np.array(action)

    print(
        f"[get_actor_from_checkpoint] Loaded IQL actor with "
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


def train():
   wrapped_train = pyrallis_wrap()(_train)
   wrapped_train()

def _train(config: IQLConfig):
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
    )

    # Setup checkpoints
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis_dump(config, f)

    rng = jax.random.PRNGKey(config.seed)
    minari_dataset = minari.load_dataset(config.dataset_id)
    env = get_env(
        config.device, 
        command_type=config.command_type, 
        dataset=minari_dataset,
        num_actors=config.n_eval_actors
    )
    shifted_env = maybe_get_shifted_env(
        config.device, command_type=config.command_type,
        dataset=minari_dataset, eval_shift=config.eval_shift,
        num_actors=config.n_eval_actors
    )
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
            normalized_score, raw_score = evaluate(
                policy_fn,
                env,
                num_episodes=config.n_episodes,
                obs_mean=obs_mean,
                obs_std=obs_std,
                seed=config.eval_seed,
            )
            print(f"Step: {current_step}, Eval Score: {normalized_score}")
            eval_metrics = {
                "eval/score": normalized_score,
                "eval/raw_score": raw_score,
            }
            wandb.log(eval_metrics, step=current_step)

            # Save checkpoint
            if config.checkpoints_path is not None and (i // eval_interval) % config.checkpoints_every == 0:
                checkpoint = {
                    "actor_params": flax_serialization.to_state_dict(train_state.actor.params),
                    "critic_params": flax_serialization.to_state_dict(train_state.critic.params),
                    "target_critic_params": flax_serialization.to_state_dict(train_state.target_critic.params),
                    "value_params": flax_serialization.to_state_dict(train_state.value.params),
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

    print("Running proxy evaluation of the final policy...")
    fn = policy_fn
    policy_fn = lambda obs: fn(observations=obs)
    proxyResult = proxy.evaluate(
        policy_fn,
        env,
        config.n_eval_episodes_final, 
        obs_mean, 
        obs_std,
        render=True,
        algorithm_name=config.name,
        dict_prefix="eval/proxy_results"
    )

    # Log proxy results
    wandb.log(proxyResult)

    # Save final checkpoint
    if config.checkpoints_path is not None:
        checkpoint = {
            "actor_params": flax_serialization.to_state_dict(train_state.actor.params),
            "critic_params": flax_serialization.to_state_dict(train_state.critic.params),
            "target_critic_params": flax_serialization.to_state_dict(train_state.target_critic.params),
            "value_params": flax_serialization.to_state_dict(train_state.value.params),
            "obs_mean": np.array(obs_mean),
            "obs_std": np.array(obs_std),
            "step": config.max_timesteps,
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
            act=policy_fn,
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
    train()
