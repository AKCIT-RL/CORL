from dataclasses import dataclass, fields
import os
import sys
from typing import Callable, List, Optional, Tuple, cast
import numpy as np
import jax
import jax.numpy as jnp
from pyrallis.argparsing import wrap
import mujoco.egl
import tqdm
import yaml

os.environ["MUJOCO_GL"] = "egl"
gl_context = mujoco.egl.GLContext(1024, 1024)
gl_context.make_current()

from algorithms.offline.bc_jax import BCActor
from algorithms.utils.wrapper_gym import GymWrapper, get_env

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

@dataclass
class CompareRandomizeAttributes:
   checkpoint_path: str
   checkpoint_config: Optional[str] = None
   n_actors: int = 4
   n_episodes: int = 20
   seed: int = 0
   device: str = "cuda"
   
@dataclass
class CompareRandomizeConfig:
   env: str
   hidden_dims: list[int]
   command_type: str

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
      
def load_config(config: CompareRandomizeAttributes) -> CompareRandomizeConfig:
   valid_fields = {
      field.name for field in fields(CompareRandomizeConfig)
   }
   
   if config.checkpoint_config is not None:
      if os.path.isfile(config.checkpoint_config):
         with open(config.checkpoint_config, "r") as f:
            data = yaml.safe_load(f)
            
         filtered_data = {
            k: v for k, v in data.items() 
            if k in valid_fields
         }
         return CompareRandomizeConfig(**filtered_data)
   
   checkpoint_dir = config.checkpoint_path
   if os.path.isfile(checkpoint_dir):
      checkpoint_dir = os.path.dirname(checkpoint_dir)
   
   config_path = os.path.join(checkpoint_dir, "config.yaml")
   
   if os.path.exists(config_path):
      config.checkpoint_config = config_path
      
      with open(config_path, "r") as f:
         data = yaml.safe_load(f)
      
      filtered_data = {
         k: v for k, v in data.items() 
         if k in valid_fields
      }
      return CompareRandomizeConfig(**filtered_data)
   
   raise FileNotFoundError("Could not find config.yaml")

def get_action_shape(env: GymWrapper) -> tuple[int, ...]:
   if env.action_space.shape is None:
      raise ValueError("Action space must have a defined shape.")
   
   action_shape: tuple[int, ...] = env.action_space.shape
   return action_shape

def parse_checkpoint_path(checkpoint_path: str) -> str:
   if os.path.isfile(checkpoint_path) and checkpoint_path.endswith(".npz"):
      return checkpoint_path
   elif os.path.isdir(checkpoint_path):
      npz_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".npz")]
      
      if not npz_files:
         raise FileNotFoundError(f"No .npz files found in directory: {checkpoint_path}")

      checkpoint_final_file = "checkpoint_final.npz"
      if checkpoint_final_file in npz_files:
         return os.path.join(checkpoint_path, checkpoint_final_file)

      def get_checkpoint_number(filename: str) -> int:
         base_name = os.path.basename(filename)
         number_str = base_name.split("_")[-1].replace(".npz", "")
         return int(number_str) if number_str.isdigit() else -1
      
      sorted_checkpoints = sorted(
         npz_files, 
         key=get_checkpoint_number,
         reverse=True
      )
      
      return os.path.join(checkpoint_path, sorted_checkpoints[0])
   
   raise FileNotFoundError(
      f"Checkpoint path does not exist: {checkpoint_path}"
   )
   
def load_checkpoint(
   attrs: CompareRandomizeAttributes,
   config: CompareRandomizeConfig,
   env: GymWrapper
) -> Tuple[Callable[[jnp.ndarray], jnp.ndarray], np.ndarray, np.ndarray]:
   # Parse checkpoint path
   checkpoint_path = parse_checkpoint_path(attrs.checkpoint_path)
   print("Carregando checkpoint de:", checkpoint_path)
   
   # Get action shape
   action_shape = get_action_shape(env)
   
   # Load model and params
   checkpoint = np.load(checkpoint_path, allow_pickle=True)
   model = BCActor(
      hidden_dims=config.hidden_dims, 
      action_dim=action_shape[0]
   )
   params = checkpoint["actor_params"].item()
   obs_mean = checkpoint["obs_mean"]
   obs_std = checkpoint["obs_std"]
   
   # Create partial actor function
   def policy_fn(obs: jnp.ndarray) -> jnp.ndarray:
      return cast(jnp.ndarray, model.apply(params, obs))
   return policy_fn, obs_mean, obs_std
   
def evaluate(
   policy_fn: Callable[[jnp.ndarray], jnp.ndarray],
   env: GymWrapper,
   num_episodes: int,
   obs_mean,
   obs_std,
   render=False
) -> List[float]:
   episode_returns = []
   num_envs = getattr(env, "num_envs", 1)
   max_steps = 2000

   with tqdm.tqdm(total=num_episodes, desc="Evaluating") as pbar:
      while len(episode_returns) < num_episodes:
         episode_return = np.zeros(num_envs, dtype=np.float32)
         finished = np.zeros(num_envs, dtype=bool)
         observation, _ = env.reset()
         steps = 0

         while not np.all(finished):
            steps += 1
            
            if steps >= max_steps:
               finished[:] = True
               break
            
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
      
      if hasattr(env, 'get_normalized_score'):
         normalized_returns = []

         for ret in episode_returns:
            normalized = env.get_normalized_score(ret)

            if normalized is not None:
               normalized_returns.append(float(normalized))
            else:
               normalized_returns.append(ret)

         return normalized_returns

      return episode_returns

from typing import List
import numpy as np

def print_results(base_returns: List[float], randomize_returns: List[float]):
   print(f"{'='*10} RESULTADOS {'='*10}")
   
   base_arr = np.array(base_returns, dtype=np.float64)
   rand_arr = np.array(randomize_returns, dtype=np.float64)
   
   base_mean = np.mean(base_arr)
   base_std = np.std(base_arr)
   print(f"Baseline:")
   print(f" - Média: {base_mean:.2f}")
   print(f" - Std: {base_std:.2f}")
   
   randomize_mean = np.mean(rand_arr)
   randomize_std = np.std(rand_arr)
   print(f"Randomizado:")
   print(f" - Média: {randomize_mean:.2f}")
   print(f" - Std: {randomize_std:.2f}")
   
   envs_relation = randomize_mean / base_mean if base_mean != 0 else float('inf')
   print(f"Relação Randomizado / Baseline: {envs_relation:.2f}")
   
   evaluate_robustness(base_arr, rand_arr)

def evaluate_robustness(base_arr: np.ndarray, rand_arr: np.ndarray):
   print("\nAnálise de Robustez (Trajetórias Pareadas)...")
   
   deltas = rand_arr - base_arr
   mean_delta = np.mean(deltas)
   p5_delta = np.percentile(deltas, 5)  # 5% piores quedas de desempenho
   
   print(f" - Variação Média (Delta): {mean_delta:+.2f}")
   print(f" - Pior Caso (5º Percentil): {p5_delta:+.2f}")
   
   with np.errstate(divide='ignore', invalid='ignore'):
      rel_drops = np.where(
         base_arr != 0,
         (base_arr - rand_arr) / np.abs(base_arr), 
         0.0
      )
   
   critical_rate = np.mean(rel_drops > 0.10) * 100
   print(f" - Taxa de Queda Crítica (>10% perda): {critical_rate:.1f}%")
   
   if np.allclose(base_arr, rand_arr):
      print(" - IC 95% (Relação): [1.00, 1.00] (Trajetórias idênticas)")
      return

   try:
      from scipy.stats import bootstrap
      
      def ratio_stat(b, r):
         m_b = np.mean(b, axis=-1)
         m_r = np.mean(r, axis=-1)
         return np.where(m_b != 0, m_r / m_b, np.nan)
      
      res = bootstrap(
         (base_arr, rand_arr),
         statistic=ratio_stat,
         paired=True,
         n_resamples=1000,
         method='basic',
      )
      ci_low = res.confidence_interval.low
      ci_high = res.confidence_interval.high
      print(f" - IC 95% (Relação): [{ci_low:.2f}, {ci_high:.2f}]")
   except (ImportError, ValueError):
      pass
   
def _main(attrs: CompareRandomizeAttributes):
   # Get config
   print(f"Carregando configuração do checkpoint...")
   config = load_config(attrs)
   
   print("Configuração carregada:")
   for field in fields(CompareRandomizeConfig):
      value = getattr(config, field.name)
      print(f" - {field.name}: {value}")
   
   # Get base environment
   print(f"\nCarregando ambiente base...")
   num_actors = max(1, min(attrs.n_actors, attrs.n_episodes))
   base_env = get_env(
      env_name=config.env,
      device=attrs.device, 
      render_callback=_render_callback,
      num_actors=num_actors,
      command_type=config.command_type,
      randomize=False,
      config_overrides={"impl": "jax"}
   )
      
   # Get checkpoint
   policy, obs_mean, obs_std = load_checkpoint(
      attrs, config, base_env
   )
   
   # Evaluate policy in base environment
   print(f"Executando política no ambiente base...")
   base_returns = evaluate(
      policy, base_env, attrs.n_episodes, obs_mean, obs_std, render=True
   )
   
   # Get randomize environment
   print(f"\nCarregando ambiente randomizado...")
   randomize_env = get_env(
      env_name=config.env,
      device=attrs.device, 
      render_callback=_render_callback,
      num_actors=num_actors,
      command_type=config.command_type,
      randomize=True,
      config_overrides={"impl": "jax"}
   )
   
   # Evaluate policy in randomize environment
   print(f"Executando política no ambiente randomizado...")
   randomize_returns = evaluate(
      policy, randomize_env, attrs.n_episodes, obs_mean, obs_std, render=True
   )
   
   # Show results
   print_results(base_returns, randomize_returns)

def main():
   wrapped_main = wrap()(_main)
   wrapped_main()

if __name__ == "__main__":
   main()