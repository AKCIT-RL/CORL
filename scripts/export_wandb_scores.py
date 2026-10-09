"""Export the final nominal and held-out scores of the 720 benchmark runs from W&B.

One row per run: algorithm, task, dataset, final nominal score (eval/final_score,
normalized x100) and, for the robustness tier, the held-out score
(eval/shifted_final_score). The Decision Transformer logs one score per target
return; we keep the first (highest) target, the one used everywhere in the paper.

Runs created on or after --exclude_from (default 2026-10-07) are left out: they are
CQL comparison experiments, not part of the benchmark grid. The script checks that
exactly 120 runs remain per algorithm.

Usage (from the CORL root):
  python -m scripts.export_wandb_scores --out analysis/paper_stats/inputs/runs.csv
"""

import argparse
import datetime as dt
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import wandb
import yaml

ALGOS = ("BC", "TD3-BC", "AWAC", "IQL", "CQL", "DT")


def algo_of(name: str) -> str:
    return "TD3-BC" if name.startswith("TD3-BC") else name.split("-")[0]


def dataset_id_of(run) -> str:
    args = (run.metadata or {}).get("args", []) or []
    for i, a in enumerate(args):
        if a.startswith("--dataset_id="):
            return a.split("=", 1)[1]
        if a == "--dataset_id" and i + 1 < len(args):
            return args[i + 1]
    # Some runs did not pass the flag on the command line; read it from config.yaml.
    with tempfile.TemporaryDirectory() as d:
        run.file("config.yaml").download(root=d, replace=True)
        cfg = yaml.safe_load(open(os.path.join(d, "config.yaml")))
    return (cfg.get("dataset_id") or {}).get("value")


def row(run) -> dict:
    algo = algo_of(run.name)
    s = run.summary._json_dict
    if algo == "DT":
        fin = {float(m.group(1)): v for k, v in s.items() if (m := re.match(r"eval/([\d.]+)_final_score$", k))}
        sh = {float(m.group(1)): v for k, v in s.items() if (m := re.match(r"eval/shifted_([\d.]+)_final_score$", k))}
        score = fin[max(fin)] if fin else None
        shifted = sh[max(sh)] if sh else None
    else:
        score = s.get("eval/final_score")
        shifted = s.get("eval/shifted_final_score")
    _, task, diff = dataset_id_of(run).split("/")
    return {
        "run": run.name,
        "algorithm": algo,
        "task": task,
        "dataset": diff.removesuffix("-v0"),
        "score": score,
        "shifted": shifted,
        "created_at": run.created_at,
        "runtime_h": (s.get("_runtime") or 0) / 3600,
        "host": (run.metadata or {}).get("host"),
        "gpu": (run.metadata or {}).get("gpu"),
        "cql_alpha_prime": s.get("training/cql/alpha_prime"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="anonymous/Offline-Benchmark")
    parser.add_argument("--exclude_from", default="2026-10-07")
    parser.add_argument("--out", default="analysis/paper_stats/inputs/runs.csv")
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
    # The AWAC run that finished without a score counts as 0, as in the paper.
    df["score"] = df["score"].fillna(0.0)
    counts = df.groupby("algorithm").size()
    assert (counts == 120).all() and len(counts) == 6, f"unexpected run counts:\n{counts}"
    assert df.groupby(["algorithm", "task", "dataset"]).size().eq(3).all(), "each cell must have 3 runs"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.sort_values(["algorithm", "task", "dataset", "run"]).to_csv(args.out, index=False)
    print(f"{len(df)} runs -> {args.out}")


if __name__ == "__main__":
    main()
