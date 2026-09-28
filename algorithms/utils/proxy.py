"""Sim2real proxy evaluation of a finished training run.

Runs exactly the evaluation of scripts/run_srr_eval.sh / run_srr_matrix.sh
(scripts/compare_randomize.py: same suite, rollout budget, normalization and
metrics) on the run's final checkpoint, so the numbers logged to W&B at the end
of training are the same ones the SRR table is built from.
"""
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from mujoco_playground import registry

# Imported first: it builds the EGL context before jax is initialized.
from scripts.compare_randomize import CompareRandomizeAttributes, _main as compare_main
from algorithms.utils.randomize_gym import (
   _assert_supported_obs_layout,
   get_predefined_randomize_configs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Same defaults as scripts/run_srr_eval.sh and scripts/run_srr_matrix.sh.
SRR_SUITE = "humanoid_gym_medium"
SRR_EPISODES = 100
SRR_ACTORS = 50
METRICS_DIR = REPO_ROOT / "logs/compare/metrics"


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

def _flatten_proxy_results(proxy_result: Dict, prefix: str) -> Dict:
   """Convert proxy metrics to top-level W&B keys with slash separators."""
   flattened = {}

   def visit(value, path):
      if isinstance(value, dict):
         for key, nested_value in value.items():
               visit(nested_value, f"{path}/{key}")
      else:
         flattened[path] = value

   for config_name, config_result in proxy_result.items():
      visit(config_result, f"{prefix}/{config_name}")

   return flattened

def suite_supported(env_name: str, suite: str = SRR_SUITE) -> bool:
   """Whether ``env_name`` has the observation layout the perturbation suite assumes."""
   cfg_file = REPO_ROOT / f"configs/randomize/{suite}.yaml"
   configs = get_predefined_randomize_configs("custom", str(cfg_file), env_name)
   env = registry.load(env_name, config_overrides={"impl": "jax"})
   try:
      _assert_supported_obs_layout(env, configs, env_name)
   except ValueError as exc:
      print(f"Skipping suite '{suite}': {exc}")
      return False
   return True

def evaluate(
   checkpoint_path: str,
   env_name: str,
   device: str = "cuda",
   dict_prefix: Optional[str] = None,
) -> Dict:
   """Score the run in ``checkpoint_path`` on `default` + the SRR suite.

   The checkpoint is picked the same way as the SRR scripts do (checkpoint_final.npz,
   else the highest step). Envs outside the supported observation layout get only
   the `default` baseline; that partial record is written next to the checkpoint
   instead of logs/compare/metrics, so the SRR matrix still treats the run as pending.
   """
   supported = suite_supported(env_name)
   record = compare_main(CompareRandomizeAttributes(
      checkpoint_path=str(checkpoint_path),
      n_actors=SRR_ACTORS,
      n_episodes=SRR_EPISODES,
      device=device,
      configs=SRR_SUITE if supported else "default",
      metrics_dir=str(METRICS_DIR if supported else checkpoint_path),
   ))

   return format_metrics(record, dict_prefix)

def format_metrics(record: Dict, dict_prefix: Optional[str] = None) -> Dict:
   """W&B-ready metrics from a compare_randomize record (in memory or its JSON)."""
   metrics = clean_dict(record["metrics"])
   return metrics if dict_prefix is None else _flatten_proxy_results(metrics, prefix=dict_prefix)
