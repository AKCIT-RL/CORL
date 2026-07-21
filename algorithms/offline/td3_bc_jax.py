# source https://github.com/sfujim/TD3_BC
# https://arxiv.org/abs/2106.06860
import os
import time
import uuid
from dataclasses import dataclass
from functools import partial
from pathlib import Path
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
import yaml
from flax.training.train_state import TrainState

from algorithms.utils.wrapper_gym import get_env, maybe_get_shifted_env, record_policy_video
from algorithms.utils.dataset import qlearning_dataset

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"


@dataclass
class TD3BCConfig:
    # wandb project name
    project: str = "CORL"
    # wandb group name
    group: str = "TD3_BC-D4RL"
    # wandb run name
    name: str = "TD3_BC"
    # training dataset and evaluation environment
    env: str = "halfcheetah-medium-expert-v2"  # OpenAI gym environment name
    dataset_id: str = "halfcheetah-medium-expert-v2"
    command_type: str = None
    # Optional Tier-5 shifted evaluation: JSON string of flattened env-config
    # overrides (e.g. stronger push-recovery kicks) applied only to a second eval
    # env. Normalization reuses the in-distribution dataset refs, so the shifted
    # score is comparable to eval/final_score (their gap = robustness gap).
    eval_shift: Optional[str] = None
    # coefficient for the Q-function in actor loss
    alpha: float = 2.5
    # discount factor
    discount: float = 0.99
    # standard deviation for the gaussian exploration noise
    expl_noise: float = 0.1
    # coefficient for the target critic Polyak's update
    tau: float = 0.005
    # scaling coefficient for the noise added to target actor during critic update
    policy_noise: float = 0.2
    # range for the target actor noise clipping
    noise_clip: float = 0.5
    # actor update delay
    policy_freq: int = 2
    # total gradient updates during training
    max_timesteps: int = int(1e6)
    # maximum size of the replay buffer
    buffer_size: int = 2_000_000
    # training batch size
    batch_size: int = 256
    # whether to normalize states
    normalize: bool = True
    # whether to normalize reward (like in IQL)
    normalize_reward: bool = False
    # evaluation frequency, will evaluate every eval_freq training steps
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
    load_model: str = ""
    # training random seed
    seed: int = 0
    # training device
    device: str = "cuda"
    # NETWORK
    hidden_dims: Tuple[int, ...] = (256, 256)
    critic_lr: float = 1e-3
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


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(
        self, observation: jnp.ndarray, action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.concatenate([observation, action], axis=-1)
        q1 = MLP((*self.hidden_dims, 1), layer_norm=True)(x)
        q2 = MLP((*self.hidden_dims, 1), layer_norm=True)(x)
        return q1, q2


class TD3Actor(nn.Module):
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
    dataset, config: TD3BCConfig, clip_to_eps: bool = True, eps: float = 1e-5
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
    if config.normalize:
        obs_mean = dataset.observations.mean(0)
        obs_std = dataset.observations.std(0)
        dataset = dataset._replace(
            observations=(dataset.observations - obs_mean) / (obs_std + 1e-5),
            next_observations=(dataset.next_observations - obs_mean) / (obs_std + 1e-5),
        )
    return dataset, obs_mean, obs_std


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


class TD3BCTrainState(NamedTuple):
    actor: TrainState
    critic: TrainState
    target_actor: TrainState
    target_critic: TrainState
    max_action: float = 1.0


class TD3BC(object):
    @classmethod
    def update_actor(
        self,
        train_state: TD3BCTrainState,
        batch: Transition,
        rng: jax.random.PRNGKey,
        config: TD3BCConfig,
    ) -> Tuple["TD3BCTrainState", jnp.ndarray]:
        def actor_loss_fn(actor_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
            predicted_action = train_state.actor.apply_fn(
                actor_params, batch.observations
            )
            critic_params = jax.lax.stop_gradient(train_state.critic.params)
            q_value, _ = train_state.critic.apply_fn(
                critic_params, batch.observations, predicted_action
            )

            mean_abs_q = jax.lax.stop_gradient(jnp.abs(q_value).mean())
            loss_lambda = config.alpha / mean_abs_q

            bc_loss = jnp.square(predicted_action - batch.actions).mean()
            loss_actor = -1.0 * q_value.mean() * loss_lambda + bc_loss
            return loss_actor

        new_actor, actor_loss = update_by_loss_grad(train_state.actor, actor_loss_fn)
        return train_state._replace(actor=new_actor), actor_loss

    @classmethod
    def update_critic(
        self,
        train_state: TD3BCTrainState,
        batch: Transition,
        rng: jax.random.PRNGKey,
        config: TD3BCConfig,
    ) -> Tuple["TD3BCTrainState", jnp.ndarray]:
        def critic_loss_fn(
            critic_params: flax.core.FrozenDict[str, Any]
        ) -> jnp.ndarray:
            q_pred_1, q_pred_2 = train_state.critic.apply_fn(
                critic_params, batch.observations, batch.actions
            )
            target_next_action = train_state.target_actor.apply_fn(
                train_state.target_actor.params, batch.next_observations
            )
            noise = (
                config.policy_noise
                * train_state.max_action
                * jax.random.normal(rng, batch.actions.shape)
            )
            target_next_action = target_next_action + noise.clip(
                -config.noise_clip, config.noise_clip
            )
            target_next_action = target_next_action.clip(
                -train_state.max_action, train_state.max_action
            )
            q_next_1, q_next_2 = train_state.target_critic.apply_fn(
                train_state.target_critic.params,
                batch.next_observations,
                target_next_action,
            )
            target = batch.rewards[..., None] + config.discount * jnp.minimum(
                q_next_1, q_next_2
            ) * (1 - batch.dones[..., None])
            target = jax.lax.stop_gradient(target)  # stop gradient for target
            value_loss_1 = jnp.square(q_pred_1 - target)
            value_loss_2 = jnp.square(q_pred_2 - target)
            value_loss = (value_loss_1 + value_loss_2).mean()
            return value_loss

        new_critic, critic_loss = update_by_loss_grad(
            train_state.critic, critic_loss_fn
        )
        return train_state._replace(critic=new_critic), critic_loss

    @classmethod
    def update_n_times(
        self,
        train_state: TD3BCTrainState,
        data: Transition,
        rng: jax.random.PRNGKey,
        config: TD3BCConfig,
    ) -> Tuple["TD3BCTrainState", Dict]:
        for _ in range(
            config.n_jitted_updates
        ):  # we can jit for roop for static unroll
            rng, batch_rng = jax.random.split(rng, 2)
            batch_idx = jax.random.randint(
                batch_rng, (config.batch_size,), 0, len(data.observations)
            )
            batch: Transition = jax.tree_util.tree_map(lambda x: x[batch_idx], data)
            rng, critic_rng, actor_rng = jax.random.split(rng, 3)
            train_state, critic_loss = self.update_critic(
                train_state, batch, critic_rng, config
            )
            if _ % config.policy_freq == 0:
                train_state, actor_loss = self.update_actor(
                    train_state, batch, actor_rng, config
                )
                new_target_critic = target_update(
                    train_state.critic, train_state.target_critic, config.tau
                )
                new_target_actor = target_update(
                    train_state.actor, train_state.target_actor, config.tau
                )
                train_state = train_state._replace(
                    target_critic=new_target_critic,
                    target_actor=new_target_actor,
                )
        return train_state, {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
        }

    @classmethod
    def get_action(
        self,
        train_state: TD3BCTrainState,
        obs: jnp.ndarray,
        max_action: float = 1.0,  # In D4RL, action is scaled to [-1, 1]
    ) -> jnp.ndarray:
        action = train_state.actor.apply_fn(train_state.actor.params, obs)
        action = action.clip(-max_action, max_action)
        return action


def create_td3bc_train_state(
    rng: jax.random.PRNGKey,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    config: TD3BCConfig,
) -> TD3BCTrainState:
    critic_model = DoubleCritic(
        hidden_dims=config.hidden_dims,
    )
    action_dim = actions.shape[-1]
    actor_model = TD3Actor(
        action_dim=action_dim,
        hidden_dims=config.hidden_dims,
    )
    rng, critic_rng, actor_rng = jax.random.split(rng, 3)
    # initialize critic
    critic_train_state: TrainState = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions),
        tx=optax.adam(config.critic_lr),
    )
    target_critic_train_state: TrainState = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions),
        tx=optax.adam(config.critic_lr),
    )
    # initialize actor
    actor_train_state: TrainState = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, observations),
        tx=optax.adam(config.actor_lr),
    )
    target_actor_train_state: TrainState = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, observations),
        tx=optax.adam(config.actor_lr),
    )
    return TD3BCTrainState(
        actor=actor_train_state,
        critic=critic_train_state,
        target_actor=target_actor_train_state,
        target_critic=target_critic_train_state,
    )


def evaluate(
    policy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    env: gym.Env,
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
            observation = (observation - obs_mean) / obs_std
            action = policy_fn(obs=observation)
            observation, reward, done, truncated, info = env.step(action)
            episode_return += reward
        episode_returns.append(episode_return)
    
    mean_return = np.mean(episode_returns)
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
    Load a TD3+BC actor from a JAX checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        state_dim: Dimension of the state/observation space.
        action_dim: Dimension of the action space.
        max_action: Maximum action magnitude (default: 1.0).

    Returns:
        Dictionary containing:
            - 'actor_fn': JIT-compiled function that takes observations and returns actions.
            - 'get_action': Convenience wrapper that applies obs normalisation and returns
                            a numpy action.
            - 'actor_params': The loaded model parameters.
            - 'obs_mean': Observation mean for normalisation.
            - 'obs_std': Observation std for normalisation.
            - 'config': The loaded config dictionary.
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
    checkpoint = np.load(selected_ckpt, allow_pickle=True)
    actor_params_dict = checkpoint["actor_params"].item()
    obs_mean = checkpoint["obs_mean"]
    obs_std = checkpoint["obs_std"]

    # Build model and restore params
    actor_model = TD3Actor(
        hidden_dims=hidden_dims,
        action_dim=action_dim,
        max_action=max_action,
    )
    actor_params = flax.serialization.from_state_dict(
        actor_model.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim))),
        actor_params_dict,
    )

    @jax.jit
    def actor_fn(obs: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(actor_model.apply(actor_params, obs), -max_action, max_action)

    def get_action(obs: np.ndarray) -> np.ndarray:
        obs_normalized = (obs - obs_mean) / (obs_std + 1e-5)
        return np.array(actor_fn(jnp.array(obs_normalized)))

    print(
        f"[get_actor_from_checkpoint] Loaded TD3+BC actor with "
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


@pyrallis.wrap()
def train(config: TD3BCConfig):
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
    train_state = create_td3bc_train_state(
        subkey, example_batch.observations, example_batch.actions, config
    )
    algo = TD3BC()
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
                    "critic_params": flax.serialization.to_state_dict(train_state.critic.params),
                    "target_actor_params": flax.serialization.to_state_dict(train_state.target_actor.params),
                    "target_critic_params": flax.serialization.to_state_dict(train_state.target_critic.params),
                    "obs_mean": np.array(obs_mean),
                    "obs_std": np.array(obs_std),
                    "step": i,
                }
                checkpoint_path = os.path.join(config.checkpoints_path, f"checkpoint_{i}.npz")
                np.savez(checkpoint_path, **checkpoint)
                print(f"Saved checkpoint to {checkpoint_path}")

    # final evaluation
    policy_fn = partial(act_fn, train_state=train_state)
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
            "critic_params": flax.serialization.to_state_dict(train_state.critic.params),
            "target_actor_params": flax.serialization.to_state_dict(train_state.target_actor.params),
            "target_critic_params": flax.serialization.to_state_dict(train_state.target_critic.params),
            "obs_mean": np.array(obs_mean),
            "obs_std": np.array(obs_std),
            "step": num_steps,
        }
        checkpoint_path = os.path.join(config.checkpoints_path, f"checkpoint_final.npz")
        np.savez(checkpoint_path, **checkpoint)
        print(f"Saved final checkpoint to {checkpoint_path}")

    # Record a rollout video of the final policy
    try:
        video_dir = config.checkpoints_path if config.checkpoints_path is not None else "videos"
        video_path = os.path.join(video_dir, f"{config.name}.mp4")
        record_policy_video(
            env_name=config.env,
            act=lambda o: policy_fn(obs=o),
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
