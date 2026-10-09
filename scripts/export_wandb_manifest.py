"""Export a manifest of the 720 benchmark runs from W&B.

One row per run: run name, W&B run id and group, state, algorithm, task,
dataset, training seed, creation time, wall-clock hours and GPU. The dataset and
seed come from the config.yaml that the run uploads, or from its command line
when it uploaded none; the Decision Transformer runs trained with seed 10 are
the benchmark's seed 0 and are written as 0.

The run selection is the one of scripts/export_wandb_scores.py: runs created on
or after --exclude_from (default 2026-10-07) are later CQL experiments and are
left out, and exactly 120 runs must remain per algorithm.

Usage (from the CORL root):
  python -m scripts.export_wandb_manifest --out analysis/paper_stats/wandb_runs.csv
"""

import argparse
import datetime as dt
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import wandb
import yaml

from scripts.export_wandb_scores import ALGOS, algo_of


def _arg(run, name):
    args = (run.metadata or {}).get("args", []) or []
    for i, a in enumerate(args):
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
        if a == f"--{name}" and i + 1 < len(args):
            return args[i + 1]
    return None


def _config(run) -> dict:
    with tempfile.TemporaryDirectory() as d:
        run.file("config.yaml").download(root=d, replace=True)
        cfg = yaml.safe_load(open(os.path.join(d, "config.yaml")))
    return {k: (v.get("value") if isinstance(v, dict) else v) for k, v in cfg.items()}


def row(run) -> dict:
    algo = algo_of(run.name)
    # config.yaml holds the seed the run actually used. Some DT runs were launched
    # with --seed 1 or 2 before that flag overrode the base config's seed of 10,
    # so the command line is only a fallback for runs that uploaded no config.
    try:
        cfg = _config(run)
    except wandb.errors.CommError:
        cfg = {}
    dataset_id = cfg.get("dataset_id") or _arg(run, "dataset_id")
    seed = cfg.get("seed") if cfg.get("seed") is not None else _arg(run, "seed")
    seed = int(seed)
    _, task, diff = dataset_id.split("/")
    if algo == "DT" and seed == 10:
        seed = 0  # the DT runs trained with seed 10 are the benchmark's seed 0
    return {
        "run": run.name,
        "wandb_id": run.id,
        "wandb_group": run.group,
        "state": run.state,
        "algorithm": algo,
        "task": task,
        "dataset": diff.removesuffix("-v0"),
        "train_seed": seed,
        "created_at": run.created_at,
        "runtime_h": (run.summary._json_dict.get("_runtime") or 0) / 3600,
        "gpu": (run.metadata or {}).get("gpu"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="akcit-offlinerl/Offline-Benchmark")
    parser.add_argument("--exclude_from", default="2026-10-07")
    parser.add_argument("--out", default="analysis/paper_stats/wandb_runs.csv")
    args = parser.parse_args()

    cutoff = dt.datetime.fromisoformat(args.exclude_from).replace(tzinfo=dt.timezone.utc)
    api = wandb.Api(timeout=120)
    runs = [
        r for r in api.runs(args.project, per_page=500)
        if algo_of(r.name) in ALGOS
        and dt.datetime.fromisoformat(r.created_at.replace("Z", "+00:00")) < cutoff
    ]
    with ThreadPoolExecutor(24) as ex:
        df = pd.DataFrame(list(ex.map(row, runs)))
    counts = df.groupby("algorithm").size()
    assert (counts == 120).all() and len(counts) == 6, f"unexpected run counts:\n{counts}"
    assert df.groupby(["algorithm", "task", "dataset", "train_seed"]).size().eq(1).all(), \
        "each algorithm x task x dataset x seed must have exactly one run"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.sort_values(["algorithm", "task", "dataset", "train_seed"]).to_csv(args.out, index=False)
    print(f"{len(df)} runs -> {args.out}")


if __name__ == "__main__":
    main()
