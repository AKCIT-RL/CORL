from dataclasses import dataclass, fields
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import minari
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

from algorithms.utils.randomize_gym import GymWrapper, get_env

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

@dataclass
class CompareRandomizeAttributes:
   checkpoint_path: str
   checkpoint_config: Optional[str] = None
   n_actors: int = 4
   n_episodes: int = 20
   seed: int = 0
   render: bool = False
   dataset_id: Optional[str] = None
   device: str = "cuda"
   # Comma-separated suite names to run; "default" is always included as the SRR baseline.
   configs: Optional[str] = None
   metrics_dir: str = "logs/compare/metrics"
   
@dataclass
class CompareRandomizeConfig:
   env: str
   # The actor architecture is read from the checkpoint weights, so no layer sizes
   # are needed here -- AWAC configs do not even carry `hidden_dims`.
   command_type: Optional[str] = None
   # Needed for D4RL normalization: carries return_min/return_expert in its metadata.
   dataset_id: Optional[str] = None
   seed: Optional[int] = None

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
   
_MAX_ACTION = 1.0

def _ordered_dense(tree: Dict[str, Any], prefix: str = "Dense") -> List[Dict[str, Any]]:
   keys = [k for k in tree if k.startswith(prefix + "_")]
   return [tree[k] for k in sorted(keys, key=lambda k: int(k.split("_")[-1]))]

def _mlp(layers: List[Dict[str, Any]], x: jnp.ndarray, activate_final: bool) -> jnp.ndarray:
   for i, layer in enumerate(layers):
      x = x @ layer["kernel"] + layer["bias"]
      if activate_final or i + 1 < len(layers):
         x = jax.nn.relu(x)
   return x

def build_policy(checkpoint) -> Callable[[jnp.ndarray], jnp.ndarray]:
   """Rebuild a deterministic actor from the weight tree alone.

   The five algorithms ship three different actor shapes, and three of them store
   the weights under the same `actor_params` key, so loading blindly yields a
   silently wrong policy instead of an error. Dispatch on structure, not on name:
     * `policy_params`      -> CQL TanhGaussian, base_network emits [mean, log_std]
     * `log_stds` present   -> IQL / AWAC Gaussian, MLP(activate_final) + mean head
     * otherwise            -> BC / TD3+BC, MLP straight to max_action * tanh
   """
   if "policy_params" in checkpoint.files:
      params = checkpoint["policy_params"].item()["params"]
      layers = _ordered_dense(params["base_network"])

      def policy_fn(obs):
         out = _mlp(layers, obs, activate_final=False)
         mean, _ = jnp.split(out, 2, axis=-1)
         return jnp.tanh(mean)

      return policy_fn

   params = checkpoint["actor_params"].item()["params"]

   if "log_stds" in params:
      hidden = _ordered_dense(params["MLP_0"])
      head = params["Dense_0"]

      def policy_fn(obs):
         h = _mlp(hidden, obs, activate_final=True)
         mean = h @ head["kernel"] + head["bias"]
         return jnp.clip(mean, -_MAX_ACTION, _MAX_ACTION)

      return policy_fn

   layers = _ordered_dense(params["MLP_0"])

   def policy_fn(obs):
      out = _mlp(layers, obs, activate_final=False)
      return jnp.clip(_MAX_ACTION * jnp.tanh(out), -_MAX_ACTION, _MAX_ACTION)

   return policy_fn

def load_checkpoint(
   attrs: CompareRandomizeAttributes,
   config: CompareRandomizeConfig,
   env: GymWrapper
) -> Tuple[Callable[[jnp.ndarray], jnp.ndarray], np.ndarray, np.ndarray]:
   checkpoint_path = parse_checkpoint_path(attrs.checkpoint_path)
   print("Carregando checkpoint de:", checkpoint_path)

   checkpoint = np.load(checkpoint_path, allow_pickle=True)
   policy_fn = build_policy(checkpoint)
   return policy_fn, checkpoint["obs_mean"], checkpoint["obs_std"]
   
def evaluate(
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
               # 0-1 scale, NOT the x100 used in the training logs: the published
               # table merges these JSONs with older .txt logs and publish_metrics
               # calibrates DEGENERATE_NOMINAL against this scale.
               normalized_returns.append(float(normalized))
            else:
               normalized_returns.append(ret)
         return np.array(normalized_returns, dtype=np.float32)

      return np.array(episode_returns, dtype=np.float32)

# `only_domain` swaps the env's sensor noise for the clean signal, so dividing it by
# `default` folds that swap into the ratio; its baseline is the clean-obs run.
_BASELINE_OVERRIDE = {"only_domain": "disabled"}

def print_results(data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
   print(f"\n{'='*10} RESULTADOS {'='*10}")

   metrics: Dict[str, Dict[str, float]] = {}
   for cfg_name, returns in data.items():
      arr = np.asarray(returns, dtype=np.float64)

      base_name = _BASELINE_OVERRIDE.get(cfg_name, "default")
      if base_name not in data:
         base_name = "default"
      base_arr = np.asarray(data.get(base_name, []), dtype=np.float64)

      stats: Dict[str, float] = {
         "score": float(np.mean(arr)),
         "score_std": float(np.std(arr)),
         "score_median": float(np.median(arr)),
         "n_episodes": float(arr.size),
      }
      metrics[cfg_name] = stats

      print(f"\nRandomização {cfg_name}:")
      print(f" - Média: {stats['score']:.2f}")
      print(f" - Std: {stats['score_std']:.2f}")

      if base_arr.size == 0:
         continue

      base_mean = float(np.mean(base_arr))
      srr = stats["score"] / base_mean if base_mean != 0 else float('inf')
      stats["srr"] = srr
      stats["gap"] = base_mean - stats["score"]
      print(f" - Baseline: {base_name}")
      print(f" - Relação Randomizado / Baseline: {srr:.2f}")

      if cfg_name != base_name:
         stats.update(evaluate_robustness(base_arr, arr))
         if base_mean != 0:
            stats["p5_retention"] = (base_mean + stats["p5_delta"]) / base_mean

   return metrics

def evaluate_robustness(base_arr: np.ndarray, rand_arr: np.ndarray) -> Dict[str, float]:
   print("\nAnálise de Robustez (Trajetórias Pareadas)...")
   
   deltas = rand_arr - base_arr
   mean_delta = float(np.mean(deltas))
   p5_delta = float(np.percentile(deltas, 5))  # 5% piores quedas de desempenho
   
   print(f" - Variação Média (Delta): {mean_delta:+.2f}")
   print(f" - Pior Caso (5º Percentil): {p5_delta:+.2f}")

   metrics: Dict[str, float] = {"delta_mean": mean_delta, "p5_delta": p5_delta}
   
   with np.errstate(divide='ignore', invalid='ignore'):
      rel_drops = np.where(
         base_arr != 0,
         (base_arr - rand_arr) / np.abs(base_arr), 
         0.0
      )
   
   # >10% catches any noticeable degradation; >50% is the one that tracks actual failure.
   for threshold in (0.10, 0.50):
      rate = float(np.mean(rel_drops > threshold) * 100)
      metrics[f"critical_rate_{int(threshold * 100)}"] = rate
      print(f" - Taxa de Queda Crítica (>{threshold:.0%} perda): {rate:.1f}%")
   
   if np.allclose(base_arr, rand_arr):
      print(" - IC 95% (Relação): [1.00, 1.00] (Trajetórias idênticas)")
      metrics["srr_ci_low"] = 1.0
      metrics["srr_ci_high"] = 1.0
      return metrics

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
      ci_low = float(res.confidence_interval.low)
      ci_high = float(res.confidence_interval.high)
      metrics["srr_ci_low"] = ci_low
      metrics["srr_ci_high"] = ci_high
      print(f" - IC 95% (Relação): [{ci_low:.2f}, {ci_high:.2f}]")
   except (ImportError, ValueError):
      pass

   return metrics
   
def _run_identity(checkpoint_path: str) -> Tuple[str, Optional[int]]:
   """Run name and training step for a checkpoint that may be a dir or a single .npz."""
   step = None
   if os.path.isfile(checkpoint_path) and checkpoint_path.endswith(".npz"):
      stem = os.path.basename(checkpoint_path)[: -len(".npz")]
      tail = stem.split("_")[-1]
      step = int(tail) if tail.isdigit() else None
      name = os.path.basename(os.path.dirname(os.path.normpath(checkpoint_path)))
   else:
      name = os.path.basename(os.path.normpath(checkpoint_path))
   return (f"{name}@{step}" if step is not None else name), step

def _main(attrs: CompareRandomizeAttributes):
   # Get config
   print("Carregando configuração do checkpoint...")
   config = load_config(attrs)
   
   print("Configuração carregada:")
   for field in fields(CompareRandomizeConfig):
      value = getattr(config, field.name)
      print(f" - {field.name}: {value}")

   num_actors = max(1, min(attrs.n_actors, attrs.n_episodes))

   # The checkpoint config carries the dataset it was trained on; --dataset_id
   # overrides it to score against a different reference.
   dataset_id = attrs.dataset_id or config.dataset_id
   dataset = minari.load_dataset(dataset_id) if dataset_id else None
   if dataset is None:
      print("AVISO: sem dataset_id, os scores serão retornos brutos (não normalizados).")

   run_name, ckpt_step = _run_identity(attrs.checkpoint_path)

   randomize_configs = ["default", "full", "only_domain", "custom", "disabled"]
   custom_configs = {
      "example": "configs/randomize/example.yaml",
      "humanoid_gym": "configs/randomize/humanoid_gym.yaml",
      "humanoid_gym_medium": "configs/randomize/humanoid_gym_medium.yaml"
   }

   selected = None
   if attrs.configs:
      selected = {name.strip() for name in attrs.configs.split(",")} | {"default"}
      unknown = selected - set(randomize_configs) - set(custom_configs)
      if unknown:
         raise ValueError(f"Unknown configs: {sorted(unknown)}")

   data = {}
   policy = None
   obs_mean = None
   obs_std = None

   def run_test(cfg_name, cfg_file=None, display_name=None):
      nonlocal data, attrs, policy, obs_mean, obs_std

      display_name = display_name or cfg_name

      print(f"\n {'='*10} Randomize Config: {display_name} {'='*10}")
      print("\nCarregando ambiente...")

      env = get_env(
         device=attrs.device, 
         render_callback=_render_callback,
         num_actors=num_actors,
         command_type=config.command_type,
         config_overrides={"impl": "jax"},
         randomize_configs=cfg_name,
         randomize_options=cfg_file,
         dataset=dataset,
         env_name=config.env
      )

      # Get checkpoint
      if not policy:
         print("Carregando checkpoint...")
         policy, obs_mean, obs_std = load_checkpoint(
            attrs, config, env
         )
      else:
         print("Checkpoint já carregado, reutilizando política...")

      print("Executando política...")
      base_returns = evaluate(
         policy, env, attrs.n_episodes, obs_mean, obs_std, render=attrs.render
      )

      data[display_name] = base_returns

      if attrs.render and render_trajectory:
         ck = os.path.basename(os.path.normpath(attrs.checkpoint_path)).replace(".npz", "")

         directory = Path(f"videos/{ck}")
         directory.mkdir(parents=True, exist_ok=True)

         timestamp = time.strftime("%Y%m%d-%H%M%S")
         filepath = f"{directory}/{display_name}-{timestamp}.mp4"

         print(f"Salvando vídeo em: {filepath}")
         env.save_video(render_trajectory, save_path=filepath)
         render_trajectory.clear()

   for cfg_name in randomize_configs:
      if cfg_name == "custom":
         for display_name, cfg_file in custom_configs.items():
            if selected is None or display_name in selected:
               run_test(cfg_name, cfg_file, display_name)
      elif selected is None or cfg_name in selected:
         run_test(cfg_name)
   
   # Show results
   metrics = print_results(data)

   record = {
      "run": run_name,
      "checkpoint": os.path.basename(os.path.normpath(attrs.checkpoint_path)),
      "checkpoint_path": attrs.checkpoint_path,
      "checkpoint_step": ckpt_step,
      "env": config.env,
      "dataset_id": dataset_id,
      "command_type": config.command_type,
      "train_seed": config.seed,
      "n_episodes": attrs.n_episodes,
      "n_actors": num_actors,
      "normalized": dataset is not None,
      "metrics": metrics,
      # kept so the published table can be recomputed without re-simulating
      "episode_returns": {k: np.asarray(v).tolist() for k, v in data.items()},
   }

   out_dir = Path(attrs.metrics_dir)
   out_dir.mkdir(parents=True, exist_ok=True)
   out_path = out_dir / f"{run_name}.json"
   out_path.write_text(json.dumps(record, indent=2))
   print(f"\nMétricas salvas em: {out_path}")

def main():
   wrapped_main = wrap()(_main)
   wrapped_main()

if __name__ == "__main__":
   main()