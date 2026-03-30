from dataclasses import dataclass
from datetime import datetime
from functools import partial
import os
from typing import Callable, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
import pyrallis
import mujoco.egl
import tqdm
os.environ["MUJOCO_GL"] = "egl"
gl_context = mujoco.egl.GLContext(1024, 1024)
gl_context.make_current()

from algorithms.offline.bc_jax import BCActor
from algorithms.utils.wrapper_gym import GymWrapper, get_env

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

@dataclass
class TestBCConfig:
   env: str = "halfcheetah-medium-expert-v2"
   load_model: Optional[str] = None
   command_type: Optional[str] = None
   n_episodes: int = 10
   num_actors: int = 4
   hidden_dims: Tuple[int, ...] = (256, 256)
   seed: int = 0
   device: str = "cuda"

def load_model(
   config: TestBCConfig, 
   action_shape: tuple, 
   obs_shape: tuple, 
   rng: jax.Array
) -> Tuple[BCActor, dict | FrozenDict, jnp.ndarray, jnp.ndarray]:
   if config.load_model is not None:
      checkpoint = np.load(config.load_model, allow_pickle=True)
      model = BCActor(hidden_dims=config.hidden_dims, action_dim=action_shape[0])
      params = checkpoint["actor_params"].item()
      obs_mean = checkpoint["obs_mean"]
      obs_std = checkpoint["obs_std"]
   else:
      model = BCActor(hidden_dims=config.hidden_dims, action_dim=action_shape[0])
      obs_dummy = jnp.zeros((1, *obs_shape))
      params = model.init(rng, obs_dummy)
      obs_mean = jnp.zeros(obs_shape)
      obs_std = jnp.ones(obs_shape)
   return model, params, obs_mean, obs_std

def _first_env_state(state):
   def _select_first(x):
      shape = getattr(x, "shape", None)
      if shape is not None and len(shape) > 0:
         return x[0]
      return x

   return jax.tree_util.tree_map(_select_first, state)

render_trajectory = []
def _render_callback(_, state):
   render_trajectory.append(_first_env_state(state))
   
def evaluate(
   policy_fn: Callable[[jnp.ndarray], jnp.ndarray],
   env: GymWrapper,
   num_episodes: int,
   obs_mean,
   obs_std,
   render=False
):
   episode_returns = []
   num_envs = getattr(env, "num_envs", 1)

   with tqdm.tqdm(total=num_episodes, desc="Evaluating") as pbar:
      while len(episode_returns) < num_episodes:
         episode_return = np.zeros(num_envs, dtype=np.float32)
         finished = np.zeros(num_envs, dtype=bool)
         observation, _ = env.reset()

         while not np.all(finished):
            observation = (observation - obs_mean) / (obs_std + 1e-5)
            action = policy_fn(observation)
            observation, reward, done, truncated, _ = env.step(action)

            done = np.asarray(done, dtype=bool)
            truncated = np.asarray(truncated, dtype=bool)
            active_mask = ~finished
            episode_return += np.asarray(reward, dtype=np.float32) * active_mask
            finished |= done | truncated

            if render:
               env.render()

         completed = min(num_envs, num_episodes - len(episode_returns))
         episode_returns.extend(episode_return[:completed].tolist())
         pbar.update(completed)
      
      mean_return = np.mean(episode_returns)
      # Use normalized score if available (D4RL), otherwise return raw score
      if hasattr(env, 'get_normalized_score'):
         return env.get_normalized_score(mean_return) * 100 # type: ignore
      else:
         return mean_return.item()

@pyrallis.wrap()  # type: ignore
def test(config: TestBCConfig):
   # Get environment
   env = get_env(
      config.env, 
      device=config.device, 
      render_callback=_render_callback,
      num_actors=max(1, min(config.num_actors, config.n_episodes)),
      command_type=config.command_type
   )
   
   # Get action and observation shapes
   action_shape: tuple[int, ...] = env.action_space.shape  # type: ignore
   obs_shape: tuple[int, ...] = env.observation_space.shape  # type: ignore
   
   # Get rng key
   rng = jax.random.PRNGKey(config.seed)
   rng, model_rng = jax.random.split(rng)
   
   # Load model and params
   model, params, obs_mean, obs_std = load_model(config, action_shape, obs_shape, model_rng)
   
   # Create partial actor function
   policy_fn = partial(model.apply, params)
   
   # Evaluate policy
   mean_return = evaluate(policy_fn, env, config.n_episodes, obs_mean, obs_std, render=True) # type: ignore
   print(f"Mean Return over {config.n_episodes} episodes: {mean_return}")
   
   timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
   env.save_video(render_trajectory, save_path=f"{config.env}-{timestamp}.mp4")

if __name__ == "__main__":
   test()  # type: ignore
   