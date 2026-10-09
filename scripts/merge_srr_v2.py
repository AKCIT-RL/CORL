"""Merge the humanoid_gym_relative_v2 re-evaluation into the paper's analysis inputs.

humanoid_gym_relative only scaled geom_friction[0, 0]. On G1 (explicit foot-floor
pairs, mu = 0.6) the friction term never reached the feet, and on H1 (feet with the
floor's priority, mu = 1.0) it could only raise friction. humanoid_gym_relative_v2
scales every floor contact (see randomize_gym.floor_friction_targets) and is
identical to v1 on every Go2 env, so only the perturbed arm of the 145 G1 and H1
checkpoints was re-run; their nominal arm is reused unchanged.

This script takes the published v1 inputs, replaces the perturbed arm of every
G1/H1 checkpoint with its v2 evaluation, relabels every perturbed row as
humanoid_gym_relative_v2 (the Go2 rows are valid v2 evaluations as they stand),
and writes sim2real_metrics.csv and unpaired_srr_metrics.csv. Before writing, it
checks that every v2 JSON matches a checkpoint of the CSV, that its nominal
episodes are identical to the v1 JSON, and that no G1/H1 checkpoint is left on v1.

Usage (from the CORL root):
  python -m scripts.merge_srr_v2 --v1_inputs analysis/paper_stats/inputs_v1 \
      --v1_json analysis/paper_stats/metrics_json \
      --v2_json analysis/paper_stats/metrics_json_v2 --out analysis/paper_stats/inputs
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from scripts.unpaired_srr_metrics import row_from_json

V1, V2 = "humanoid_gym_relative", "humanoid_gym_relative_v2"
FIXED_ENVS = ("G1JoystickFlatTerrain", "H1JoystickGaitTracking")
# Columns of sim2real_metrics.csv that come from metrics[suite] of the JSON.
METRIC_COLS = ["n_episodes", "score", "score_std", "score_median", "srr", "gap", "delta_mean", "p5_delta",
               "p5_retention", "critical_rate_10", "critical_rate_50", "srr_ci_low", "srr_ci_high"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v1_inputs", default="analysis/paper_stats/inputs_v1")
    parser.add_argument("--v1_json", default="analysis/paper_stats/metrics_json")
    parser.add_argument("--v2_json", default="analysis/paper_stats/metrics_json_v2")
    parser.add_argument("--out", default="analysis/paper_stats/inputs")
    args = parser.parse_args()

    m = pd.read_csv(os.path.join(args.v1_inputs, "sim2real_metrics.csv"))
    u = pd.read_csv(os.path.join(args.v1_inputs, "unpaired_srr_metrics.csv"))
    pert = m.suite == V1
    fixed = set(m.loc[pert & m.env.isin(FIXED_ENVS), "checkpoint"])

    seen, unpaired = set(), {}
    for path in sorted(glob.glob(os.path.join(args.v2_json, "*.json"))):
        with open(path) as f:
            d = json.load(f)
        ck = d["run"]
        assert ck in fixed, f"{ck}: not a G1/H1 checkpoint of the published CSV"
        with open(os.path.join(args.v1_json, f"{ck}.json")) as f:
            nominal_v1 = json.load(f)["episode_returns"]["default"]
        assert np.array_equal(d["episode_returns"]["default"], nominal_v1), f"{ck}: nominal arm differs from v1"
        row = (m.checkpoint == ck) & pert
        assert row.sum() == 1
        m.loc[row, METRIC_COLS] = [d["metrics"][V2][c] for c in METRIC_COLS]
        unpaired[ck] = row_from_json(path, V2)
        seen.add(ck)
    assert seen == fixed, f"G1/H1 checkpoints without a v2 evaluation: {sorted(fixed - seen)}"

    m.loc[pert, "suite"] = V2
    u = u.set_index("checkpoint")
    for ck, r in unpaired.items():
        for k, v in r.items():
            if k != "checkpoint":
                u.loc[ck, k] = v
    u["suite"] = V2
    u = u.reset_index()

    os.makedirs(args.out, exist_ok=True)
    m.to_csv(os.path.join(args.out, "sim2real_metrics.csv"), index=False)
    u.to_csv(os.path.join(args.out, "unpaired_srr_metrics.csv"), index=False)
    print(f"replaced the perturbed arm of {len(seen)} G1/H1 checkpoints; wrote {args.out}")


if __name__ == "__main__":
    main()
