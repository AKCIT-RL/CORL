"""Recompute the episode-distribution SRR metrics without pairing episodes.

The nominal ("default") and perturbed arms of scripts/compare_randomize.py do not
share reset keys, so the i-th nominal and i-th perturbed episodes do not start from
the same state. The paired metrics it writes (p5_delta, p5_retention,
critical_rate_*) therefore mix the effect of the perturbation with the variance of
initial states. This script recomputes them from the per-episode returns stored in
each metrics JSON, using only the distribution of perturbed episodes:

  worst_case_retention = q05(P_episodes) / N        (5th percentile of perturbed scores)
  failure_rate_50      = mean(P_episodes < 0.5 * N)  (share of perturbed episodes below N/2)

where N is the mean nominal score. Episode scores vary a lot even without any
perturbation, so each measure is also computed on the nominal episodes themselves
(worst_case_retention_nominal, failure_rate_50_nominal). That is the baseline the
perturbed value must be read against. SRR = mean(P_episodes) / N is unchanged. As in the
published CSV, the ratios are only defined when N >= 0.05 (ratio_valid).

Usage (from the CORL root):
  python -m scripts.unpaired_srr_metrics                         # logs/compare/metrics -> CSV
  python -m scripts.unpaired_srr_metrics --metrics_dir DIR --out FILE
  python -m scripts.unpaired_srr_metrics --runs_csv metrics.csv  # keep only the
      checkpoints listed in the published CSV (drops stale or replaced runs)
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

NOMINAL = "default"
VALID_FLOOR = 0.05


def row_from_json(path: str, suite: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    episodes = d.get("episode_returns") or {}
    if NOMINAL not in episodes or suite not in episodes:
        return {}
    nominal = np.asarray(episodes[NOMINAL], dtype=np.float64)
    perturbed = np.asarray(episodes[suite], dtype=np.float64)
    n_mean = float(nominal.mean())
    p_mean = float(perturbed.mean())
    valid = n_mean >= VALID_FLOOR
    checkpoint = d.get("run") or os.path.basename(path).removesuffix(".json")
    seed = d.get("train_seed")
    if checkpoint.startswith("DT-") and seed == 10:
        seed = 0  # the DT runs trained with seed 10 are the benchmark's seed 0
    return {
        "checkpoint": checkpoint,
        "env": d.get("env"),
        "dataset_id": d.get("dataset_id"),
        "train_seed": seed,
        "suite": suite,
        "n_nominal": nominal.size,
        "n_perturbed": perturbed.size,
        "nominal_score": n_mean,
        "perturbed_score": p_mean,
        "nominal_sd": float(nominal.std(ddof=1)),
        "perturbed_sd": float(perturbed.std(ddof=1)),
        "ratio_valid": valid,
        "srr": p_mean / n_mean if valid else np.nan,
        "worst_case_retention": float(np.percentile(perturbed, 5)) / n_mean if valid else np.nan,
        "failure_rate_50": float(np.mean(perturbed < 0.5 * n_mean)) * 100 if valid else np.nan,
        "worst_case_retention_nominal": float(np.percentile(nominal, 5)) / n_mean if valid else np.nan,
        "failure_rate_50_nominal": float(np.mean(nominal < 0.5 * n_mean)) * 100 if valid else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics_dir", default="logs/compare/metrics")
    parser.add_argument("--suite", default="humanoid_gym_relative")
    parser.add_argument("--out", default="logs/compare/unpaired_srr_metrics.csv")
    parser.add_argument("--runs_csv", default=None,
                        help="Published metrics.csv; only its checkpoints are kept.")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.metrics_dir, "*.json")))
    rows = [r for r in (row_from_json(p, args.suite) for p in paths) if r]
    df = pd.DataFrame(rows)
    df["algorithm"] = df["checkpoint"].str.extract(r"^(TD3-BC|[A-Z]+)-")
    if args.runs_csv:
        keep = set(pd.read_csv(args.runs_csv)["checkpoint"])
        missing = keep - set(df["checkpoint"])
        df = df[df["checkpoint"].isin(keep)]
        print(f"kept {len(df)} checkpoints listed in {args.runs_csv}; "
              f"{len(missing)} listed there have no local JSON")
    df.to_csv(args.out, index=False)

    print(f"{len(paths)} JSON files, {len(df)} with both arms -> {args.out}")
    valid = df[df["ratio_valid"]]
    cols = ["srr", "worst_case_retention_nominal", "worst_case_retention",
            "failure_rate_50_nominal", "failure_rate_50"]
    summary = valid.groupby("algorithm")[cols].mean()
    summary["n_valid"] = valid.groupby("algorithm").size()
    summary["n_total"] = df.groupby("algorithm").size()
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
