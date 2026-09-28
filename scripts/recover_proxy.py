"""Finish training runs that died in the end-of-training proxy evaluation.

The matrix runs of job 33282 trained to completion, then crashed while saving the
proxy video (no ffmpeg on the node), before logging the proxy metrics and saving
checkpoint_final.npz. The last periodic checkpoint is already the final policy,
so nothing needs retraining. For each failed run's matrix log this:

  1. promotes the last periodic checkpoint to checkpoint_final.npz (only when its
     step is the final training step);
  2. runs proxy.evaluate -- the scripts/run_srr_eval.sh evaluation -- unless the
     SRR matrix already wrote a JSON for the final policy, which is reused;
  3. resumes the same W&B run, logs eval/proxy_results/* and the rollout video,
     and finishes it, so the run shows up as finished;
  4. touches the .done_runs marker, so run_offline_matrix.sh skips it.

Usage:
  python -m scripts.recover_proxy logs/matrix/bc-go2-footstand-medium-seed2.log [...]
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from algorithms.utils import proxy
from scripts.compare_randomize import build_policy

import wandb
from algorithms.utils.randomize_gym import record_policy_video

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_URL = re.compile(r"wandb\.ai/([^/\s]+)/([^/\s]+)/runs/([0-9a-zA-Z-]+)")
CKPT_PATH = re.compile(r"Checkpoints path: (\S+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _final_step(cfg: dict) -> int:
   # DT counts plain updates; the MLP trainers' `num_steps` counts jitted updates.
   if "update_steps" in cfg:
      return cfg["update_steps"]
   return cfg["max_timesteps"] // cfg.get("n_jitted_updates", 1)


def _promote_final_checkpoint(ckpt_dir: Path, cfg: dict) -> Path:
   final = ckpt_dir / "checkpoint_final.npz"
   if final.exists():
      return final

   steps = sorted(
      int(p.stem.split("_")[-1])
      for p in ckpt_dir.glob("checkpoint_*.npz")
      if p.stem.split("_")[-1].isdigit()
   )
   if not steps or steps[-1] != _final_step(cfg):
      raise RuntimeError(
         f"{ckpt_dir}: last checkpoint step {steps[-1] if steps else None} is not the "
         f"final step {_final_step(cfg)}; the run did not finish training"
      )
   shutil.copyfile(ckpt_dir / f"checkpoint_{steps[-1]}.npz", final)
   print(f"Promoted checkpoint_{steps[-1]}.npz -> {final}")
   return final


def _reusable_record(ckpt_dir: Path, cfg: dict) -> Optional[dict]:
   """The SRR matrix's JSON for this run, if it already scored the final policy.

   run_srr_matrix.sh points compare_randomize at the run directory, which picks the
   highest checkpoint, so a JSON written after checkpoint_<final step>.npz scored
   exactly the policy we would evaluate now, with the same suite and budget.
   """
   path = proxy.METRICS_DIR / f"{cfg['name']}.json"
   final_ckpt = ckpt_dir / f"checkpoint_{_final_step(cfg)}.npz"
   if not path.is_file() or not final_ckpt.is_file():
      return None
   if path.stat().st_mtime <= final_ckpt.stat().st_mtime:
      return None

   record = json.loads(path.read_text())
   if (
      record.get("n_episodes") != proxy.SRR_EPISODES
      or record.get("n_actors") != proxy.SRR_ACTORS
      or record.get("checkpoint_step") not in (None, _final_step(cfg))
      or set(record.get("metrics", {})) != {"default", proxy.SRR_SUITE}
   ):
      return None
   print(f"Reusing SRR metrics from {path}")
   return record


def _record_video(final: Path, cfg: dict) -> Path:
   checkpoint = np.load(final, allow_pickle=True)
   if "transformer_params" in checkpoint.files:
      raise NotImplementedError("DT needs dt_jax.record_dt_video and its train_state")
   policy_fn = build_policy(checkpoint)
   video_path = final.parent / f"{cfg['name']}.mp4"
   record_policy_video(
      env_name=cfg["env"],
      act=policy_fn,
      obs_mean=checkpoint["obs_mean"],
      obs_std=checkpoint["obs_std"],
      device=cfg.get("device", "cuda"),
      save_path=str(video_path),
      command_type=cfg.get("command_type"),
   )
   return video_path


def recover(log_path: Path, device: str) -> None:
   text = ANSI.sub("", log_path.read_text(errors="replace"))
   url = RUN_URL.findall(text)
   ckpt = CKPT_PATH.findall(text)
   if not url or not ckpt:
      raise RuntimeError(f"{log_path}: W&B run URL or checkpoints path not found")
   entity, project, run_id = url[-1]

   ckpt_dir = REPO_ROOT / ckpt[-1]
   cfg = yaml.safe_load((ckpt_dir / "config.yaml").read_text())
   cfg.setdefault("env", cfg.get("env_name"))  # DT names it `env_name`
   print(f"=== {cfg['name']} (wandb {entity}/{project}/{run_id})")

   final = _promote_final_checkpoint(ckpt_dir, cfg)
   record = _reusable_record(ckpt_dir, cfg)
   if record is not None:
      metrics = proxy.format_metrics(record, dict_prefix="eval/proxy_results")
   else:
      metrics = proxy.evaluate(
         str(ckpt_dir), env_name=cfg["env"], device=device, dict_prefix="eval/proxy_results"
      )

   run = wandb.init(entity=entity, project=project, id=run_id, resume="must")
   run.log(metrics)
   try:
      video_path = _record_video(final, cfg)
      run.log({"eval/video": wandb.Video(str(video_path))})
      print(f"Saved rollout video to {video_path}")
   except Exception as e:
      print(f"[video] failed to record rollout video: {e}")
   run.finish()

   # <group>-seed<N>.log -> .done_runs/<group>-seed<N>.done, as run_offline_matrix.sh does.
   marker = REPO_ROOT / ".done_runs" / f"{log_path.stem}.done"
   marker.parent.mkdir(parents=True, exist_ok=True)
   marker.touch()
   print(f"Marked done: {marker}")


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
   parser.add_argument("logs", nargs="+", type=Path, help="logs/matrix/<group>-seed<N>.log")
   parser.add_argument("--device", default="cuda")
   args = parser.parse_args()
   logs = [p.resolve() for p in args.logs]

   os.chdir(REPO_ROOT)  # compare_randomize resolves the suite YAMLs relative to the repo
   failed = 0
   for log_path in logs:
      try:
         recover(log_path, args.device)
      except Exception as e:
         failed += 1
         print(f"FAILED {log_path}: {e}", file=sys.stderr)
   return 1 if failed else 0


if __name__ == "__main__":
   sys.exit(main())
