from dataclasses import asdict, dataclass
import os
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional
import numpy as np
import jax
import jax.numpy as jnp
import mujoco.egl
import tqdm
import math

os.environ["MUJOCO_GL"] = "egl"
gl_context = mujoco.egl.GLContext(1024, 1024)
gl_context.make_current()

from algorithms.utils.randomize_gym import GymWrapper, get_predefined_randomize_configs

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def clean_dict(d):
   if isinstance(d, dict):
      return {k: clean_dict(v) for k, v in d.items()}
   elif isinstance(d, list):
      return [clean_dict(x) for x in d]
   elif isinstance(d, float):
      if np.isnan(d) or math.isnan(d):
         return 0.0
      if np.isinf(d) or math.isinf(d):
         return 999999.0 if d > 0 else -999999.0
      return d
   else:
      return d

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

@dataclass
class ConfigResult:
   mean: float
   std: float
   ratio_randomized_baseline: float
   robustness_analysis: Dict[str, float]

ProxyResult = Dict[str, ConfigResult]

def evaluate(
   policy_fn: Callable[[jnp.ndarray], jnp.ndarray],
   env: GymWrapper,
   num_episodes: int,
   obs_mean,
   obs_std,
   render: bool = False,
   algorithm_name: Optional[str] = None
) -> Dict:
   default_render_callback = env.render_callback
   if render:
      env.render_callback = _render_callback

   randomize_configs = ["default", "custom"]
   custom_configs = {
      "humanoid_gym": "configs/randomize/humanoid_gym.yaml"
   }

   data = {}

   def run_test(cfg_name, cfg_file=None, display_name=None):
      nonlocal data, policy_fn, obs_mean, obs_std

      display_name = display_name or cfg_name

      print(f"\n {'='*10} Random config: {display_name} {'='*10}")

      print("\nLoading environment...")
      configs = get_predefined_randomize_configs(cfg_name, cfg_file)
      env.update_randomize_functions(configs.get_functions())
      env.warmup_jit_reset()

      print("Evaluating policy...")
      base_returns = _evaluate_individual(
         policy_fn, env, num_episodes, obs_mean, obs_std, render=render
      )

      data[display_name] = base_returns

      if render and render_trajectory:
         timestamp = time.strftime("%Y%m%d-%H%M%S")

         directory = Path(f"videos/{algorithm_name}") if algorithm_name else Path(f"videos/{display_name}")
         directory.mkdir(parents=True, exist_ok=True)

         filepath = f"{directory}/{display_name}-{timestamp}.mp4" if algorithm_name else f"{directory}/{timestamp}.mp4"

         print(f"Saving video to: {filepath}")
         env.save_video(render_trajectory, save_path=filepath)
         render_trajectory.clear()

   for cfg_name in randomize_configs:
      if cfg_name == "custom":
         for display_name, cfg_file in custom_configs.items():
            run_test(cfg_name, cfg_file, display_name)
      else:
         run_test(cfg_name)
   
   # Show results
   results = _calculate_results(data)
   env.render_callback = default_render_callback

   return {k: clean_dict(asdict(v)) for k, v in results.items()}
   
def _evaluate_individual(
   policy_fn: Callable[[jnp.ndarray], jnp.ndarray],
   env: GymWrapper,
   num_episodes: int,
   obs_mean,
   obs_std,
   render=False
) -> np.ndarray:
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

      episode_returns = np.array(episode_returns, dtype=np.float32)
      
      if hasattr(env, 'get_normalized_score'):
         normalized_returns = []

         for ret in episode_returns:
            normalized = env.get_normalized_score(ret)

            if normalized is not None:
               normalized_returns.append(100 * float(normalized))
            else:
               normalized_returns.append(ret)

         return np.array(normalized_returns, dtype=np.float32)

      return episode_returns

def _calculate_results(
   data: Dict[str, np.ndarray]
) -> ProxyResult:
   print(f"\n{'='*10} RESULTS {'='*10}")

   results: ProxyResult = {}

   base_returns = data.get("default", np.array([], dtype=np.float32))
   if base_returns.size > 0:
      base_mean = np.mean(base_returns)
   
   for cfg_name, returns in data.items():
      mean = np.mean(returns)
      std = np.std(returns)
      srr = mean / base_mean if base_mean != 0 else float('inf')
      print(f"\nRandomization {cfg_name}:")
      print(f" - Mean: {mean:.2f}")
      print(f" - Std: {std:.2f}")
      print(f" - Ratio Randomized / Baseline: {srr:.2f}")

      robustness_analysis={}
      if base_returns.size > 0:
         if cfg_name != "default":
            robustness_analysis = _evaluate_robustness(base_returns, returns)

      results[cfg_name] = ConfigResult(
         mean=float(mean),
         std=float(std),
         ratio_randomized_baseline=float(srr),
         robustness_analysis=robustness_analysis
      )

   return results

def _evaluate_robustness(
   base_arr: np.ndarray, 
   rand_arr: np.ndarray
) -> Dict[str, float]:
   print("\nRobustness Analysis (Paired Trajectories)...")

   result = {}
   
   deltas = rand_arr - base_arr
   mean_delta = np.mean(deltas)
   p5_delta = np.percentile(deltas, 5)  # 5% piores quedas de desempenho
   
   print(f" - Mean Delta: {mean_delta:+.2f}")
   print(f" - Worst Case (5th Percentile): {p5_delta:+.2f}")
   result["mean_delta"] = float(mean_delta)
   result["worst_case_5th_percentile"] = float(p5_delta)
   
   with np.errstate(divide='ignore', invalid='ignore'):
      rel_drops = np.where(
         base_arr != 0,
         (base_arr - rand_arr) / np.abs(base_arr), 
         0.0
      )
   
   critical_rate = np.mean(rel_drops > 0.10) * 100
   print(f" - Critical Drop Rate (>10% loss): {critical_rate:.1f}%")
   result["critical_drop_rate"] = float(critical_rate)
   
   if np.allclose(base_arr, rand_arr):
      print(" - No significant difference between baseline and randomized returns.")

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
      print(f" - 95% CI (Ratio): [{ci_low:.2f}, {ci_high:.2f}]")
      result["95_ci_ratio_low"] = float(ci_low)
      result["95_ci_ratio_high"] = float(ci_high)

   except (ImportError, ValueError):
      pass

   return result