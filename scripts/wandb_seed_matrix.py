#!/usr/bin/env python
"""Seed-coverage matrix (algorithm x dataset) from the wandb project.

Counts how many distinct seeds already finished for every
(algorithm, task, difficulty) triple and reports what is still missing.

Usage:
    export WANDB_API_KEY=...
    python scripts/wandb_seed_matrix.py [--target 5] [--markdown] [--cache /tmp/wandb_runs.json]

Runs are keyed by --dataset_id / --seed taken from each run's wandb metadata
(the training scripts do not push their config to wandb, and the early July
runs were logged with a broken --group), so metadata is the only reliable
source for the (task, difficulty, seed) triple.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = REPO / "configs" / "offline" / "_datasets.yaml"
PROJECT = "akcit-offlinerl/Offline-Benchmark"
ALGOS = ["bc", "awac", "td3_bc", "cql", "iql", "dt"]
PROG2ALGO = {f"-m algorithms.offline.{a}_jax": a for a in ALGOS}


def fetch_runs(project: str) -> list[dict]:
    import wandb

    api = wandb.Api(timeout=60)
    runs = list(api.runs(project, per_page=500))

    def grab(run):
        rec = {
            "id": run.id,
            "name": run.name,
            "group": run.group,
            "state": run.state,
            "created": str(run.created_at),
            "seed": None,
            "dataset_id": None,
            "program": None,
            "final_score": run.summary.get("eval/final_score"),
        }
        meta = run.metadata or {}
        rec["program"] = meta.get("program")
        args = meta.get("args") or []
        # scripts.recover_proxy resumes a finished run and overwrites its metadata
        # with its own command, whose only argument is <algo>-<task>-<diff>-seed<N>.log.
        if rec["program"] == "-m scripts.recover_proxy" and args:
            m = re.search(r"/([a-z0-9_]+)-[^/]*-seed(\d+)\.log$", args[0])
            if m:
                rec["program"] = f"-m algorithms.offline.{m.group(1)}_jax"
                rec["seed"] = m.group(2)
                rec["dataset_id"] = run.config.get("dataset_id")
            return rec
        for i, arg in enumerate(args[:-1]):
            if arg == "--seed":
                rec["seed"] = args[i + 1]
            elif arg == "--dataset_id":
                rec["dataset_id"] = args[i + 1]
        return rec

    def grab_seed(run):
        rec = grab(run)
        # DT's seed-10 runs (trained before --seed reached dt_jax) are relabeled
        # as matrix seed 0 through config.matrix_seed; their metadata still says 1/2.
        if "matrix_seed" in run.config:
            rec["seed"] = str(run.config["matrix_seed"])
        return rec

    with ThreadPoolExecutor(max_workers=16) as pool:
        return list(pool.map(grab_seed, runs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=5, help="seeds wanted per cell")
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--cache", default="/tmp/wandb_runs.json")
    ap.add_argument("--offline", action="store_true", help="reuse --cache, do not hit the API")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    if args.offline:
        runs = json.load(open(args.cache))
    else:
        runs = fetch_runs(args.project)
        json.dump(runs, open(args.cache, "w"), indent=1)

    registry = yaml.safe_load(open(REGISTRY))
    tasks = list(registry["tasks"])
    diffs = list(registry["difficulties"])

    seen: dict[tuple[str, str, str], set[int]] = collections.defaultdict(set)
    skipped = []
    for run in runs:
        algo = PROG2ALGO.get(run["program"])
        dataset_id = run["dataset_id"]
        if not algo or not dataset_id or run["state"] != "finished" or run["seed"] is None:
            skipped.append(run)
            continue
        _, task, difficulty = dataset_id.split("/")
        seen[(algo, task, difficulty.removesuffix("-v0"))].add(int(run["seed"]))

    done = sum(len(v) for v in seen.values())
    print(f"{args.project}: {len(runs)} runs, {done} usable (algo, dataset, seed) cells, "
          f"{len(skipped)} ignored (unfinished / no metadata)")

    if args.markdown:
        print("\n| dataset | " + " | ".join(ALGOS) + " |")
        print("|---" * (len(ALGOS) + 1) + "|")
    else:
        header = f"{'dataset':<38}" + "".join(f"{a:>8}" for a in ALGOS)
        print("\n" + header + "\n" + "-" * len(header))

    missing = 0
    for task in tasks:
        for diff in diffs:
            cells = [len(seen[(a, task, diff)]) for a in ALGOS]
            missing += sum(max(0, args.target - c) for c in cells)
            label = f"{task}/{diff}"
            if args.markdown:
                print(f"| {label} | " + " | ".join(str(c) for c in cells) + " |")
            else:
                print(f"{label:<38}" + "".join(f"{c:>8}" for c in cells))

    totals = [sum(len(seen[(a, t, d)]) for t in tasks for d in diffs) for a in ALGOS]
    if args.markdown:
        print("| **total** | " + " | ".join(f"**{t}**" for t in totals) + " |")
    else:
        print("-" * (38 + 8 * len(ALGOS)))
        print(f"{'TOTAL':<38}" + "".join(f"{t:>8}" for t in totals))

    full = len(tasks) * len(diffs) * len(ALGOS) * args.target
    print(f"\ntarget = {args.target} seeds x {len(tasks) * len(diffs)} datasets x {len(ALGOS)} algos = {full} runs")
    print(f"done   = {full - missing}   missing = {missing}")

    seeds_used = sorted({s for v in seen.values() for s in v})
    print(f"seeds seen in wandb: {seeds_used}")
    return 0


if __name__ == "__main__":
    if not os.environ.get("WANDB_API_KEY") and "--offline" not in sys.argv:
        print("warning: WANDB_API_KEY not set; relying on ~/.netrc", file=sys.stderr)
    raise SystemExit(main())
