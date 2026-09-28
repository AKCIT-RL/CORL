"""Publish sim2real evaluation metrics to the akcit-rl/offline-benchmark dataset repo.

Reads two sources and merges them into one flat table:
  * logs/compare/metrics/*.json  -- written by compare_randomize.py (full precision)
  * logs/compare/**/*.txt        -- older runs that predate the JSON output (2 decimals)

Run with --push to upload; without it, the CSV is only written locally.
"""
import argparse
import json
import pathlib
import re

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ID = "akcit-rl/offline-benchmark"
OUT = ROOT / "logs/compare/sim2real_metrics.csv"

# Older text logs, kept so the published table covers everything already simulated.
LEGACY = {
    "after": "logs/compare/after",
    "medium": "logs/compare/medium",
    "phase2": "logs/compare/phase2",
    "sweep": "logs/compare/sweep",
}

SUITE_BLOCK = re.compile(r"Randomiza\S+ (\w+):\n - M\S+dia: ([-\d.]+)\n - Std: ([-\d.]+)")

METRIC_COLS = [
    "score", "score_std", "score_median", "srr", "gap", "delta_mean", "p5_delta",
    "p5_retention", "critical_rate_10", "critical_rate_50", "srr_ci_low", "srr_ci_high",
]

# Below this nominal score the policy never learned the task, and every ratio-based
# metric degenerates: a policy that fails identically in both conditions scores SRR 1.0.
DEGENERATE_NOMINAL = 0.05


def matrix_seed(algorithm, train_seed):
    """Seed as labeled in the benchmark.

    dt_jax ignored --seed until 2026-09-28 and trained every run with train_seed 10;
    those runs are the benchmark's DT seed 0 (wandb config.matrix_seed = 0).
    """
    if algorithm.split("-")[0] == "DT" and train_seed == 10:
        return 0
    return train_seed


def task_and_split(dataset_id):
    """'playground/go2-flat-forward/expert-v0' -> ('go2-flat-forward', 'expert')."""
    if not dataset_id:
        return None, None
    m = re.search(r"/([\w-]+)/([\w-]+)-v\d+$", dataset_id)
    return (m.group(1), m.group(2)) if m else (None, None)


def rows_from_json(path):
    rec = json.loads(path.read_text())
    task, split = task_and_split(rec.get("dataset_id"))
    nominal = rec["metrics"].get("default", {}).get("score")
    for suite, stats in rec["metrics"].items():
        row = {
            "algorithm": rec["checkpoint"].split("-Go2")[0],
            "env": rec.get("env"),
            "task": task,
            "dataset": split,
            "train_seed": matrix_seed(rec["checkpoint"].split("-Go2")[0], rec.get("train_seed")),
            "checkpoint": rec["checkpoint"],
            "checkpoint_step": rec.get("checkpoint_step"),
            "suite": suite,
            "n_episodes": rec.get("n_episodes"),
            "nominal_score": nominal,
            "ratio_valid": nominal is not None and nominal >= DEGENERATE_NOMINAL,
            "precision": "full",
            "source": "json",
        }
        row.update({c: stats.get(c) for c in METRIC_COLS})
        yield row


def checkpoint_dir_for(source, stem):
    if source in ("after", "medium"):
        return f"BC-Go2JoystickFlatTerrain-{stem}", None
    if source == "sweep":
        run, _, ckpt = stem.partition("__")
        m = re.search(r"checkpoint_(\d+)", ckpt)
        return run, int(m.group(1)) if m else None
    return stem, None


def config_of(ckpt_dir):
    path = ROOT / "checkpoints/BC" / ckpt_dir / "config.yaml"
    if not path.exists():
        return None
    y = path.read_text()

    def grab(pat, cast=str):
        m = re.search(pat, y, re.M)
        return cast(m.group(1)) if m else None

    return {
        "env": grab(r"^env: (\S+)"),
        "dataset_id": grab(r"^dataset_id: (\S+)"),
        "train_seed": grab(r"^seed: (\d+)", int),
    }


def rows_from_legacy(source, path):
    text = path.read_text()
    ckpt_dir, step = checkpoint_dir_for(source, path.stem)
    cfg = config_of(ckpt_dir)
    if cfg is None:
        return
    task, split = task_and_split(cfg["dataset_id"])

    blocks = list(SUITE_BLOCK.finditer(text))
    baseline = None
    parsed = {}
    for i, m in enumerate(blocks):
        end = blocks[i + 1].start() if i + 1 < len(blocks) else len(text)
        chunk = text[m.end():end]
        stats = {"score": float(m.group(2)), "score_std": float(m.group(3))}
        for key, pat in (
            ("srr", r"Rela\S+ Randomizado / Baseline: ([-\d.]+)"),
            ("delta_mean", r"Delta\): ([-+\d.]+)"),
            ("p5_delta", r"Percentil\): ([-+\d.]+)"),
            ("critical_rate_10", r"Cr\S+tica \(>10% perda\): ([\d.]+)%"),
            ("critical_rate_50", r"Cr\S+tica \(>50% perda\): ([\d.]+)%"),
        ):
            found = re.search(pat, chunk)
            if found:
                stats[key] = float(found.group(1))
        ci = re.search(r"IC 95% \(Rela\S+\): \[([-\d.]+), ([-\d.]+)\]", chunk)
        if ci:
            stats["srr_ci_low"] = float(ci.group(1))
            stats["srr_ci_high"] = float(ci.group(2))
        parsed[m.group(1)] = stats
        if m.group(1) == "default":
            baseline = stats["score"]

    for suite, stats in parsed.items():
        if baseline:
            stats.setdefault("gap", baseline - stats["score"])
            if "p5_delta" in stats:
                stats["p5_retention"] = (baseline + stats["p5_delta"]) / baseline
        row = {
            "algorithm": ckpt_dir.split("-Go2")[0],
            "env": cfg["env"],
            "task": task,
            "dataset": split,
            "train_seed": matrix_seed(ckpt_dir.split("-Go2")[0], cfg["train_seed"]),
            "checkpoint": ckpt_dir,
            "checkpoint_step": step,
            "suite": suite,
            "n_episodes": None,
            "nominal_score": baseline,
            "ratio_valid": baseline is not None and baseline >= DEGENERATE_NOMINAL,
            "precision": "2dp",
            "source": source,
        }
        row.update({c: stats.get(c) for c in METRIC_COLS})
        yield row


def main(push):
    rows = []
    for f in sorted((ROOT / "logs/compare/metrics").glob("*.json")):
        rows.extend(rows_from_json(f))
    n_json = len(rows)

    have = {(r["checkpoint"], r["checkpoint_step"], r["suite"]) for r in rows}
    for source, folder in LEGACY.items():
        for f in sorted((ROOT / folder).glob("*.txt")):
            for row in rows_from_legacy(source, f):
                key = (row["checkpoint"], row["checkpoint_step"], row["suite"])
                if key not in have:  # full-precision JSON always wins
                    rows.append(row)
                    have.add(key)

    df = pd.DataFrame(rows).sort_values(
        ["env", "task", "dataset", "train_seed", "checkpoint_step", "suite"],
        na_position="first")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"{len(df)} linhas ({n_json} de JSON, {len(df) - n_json} de logs antigos)")
    print(f"{df['checkpoint'].nunique()} checkpoints | suites: {sorted(df['suite'].unique())}")
    print(f"escrito em {OUT}")

    if not push:
        print("\n(dry-run: use --push para enviar ao Hugging Face)")
        return

    from huggingface_hub import HfApi
    HfApi().upload_file(
        path_or_fileobj=str(OUT),
        path_in_repo="sim2real/metrics.csv",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"sim2real metrics: {len(df)} rows, {df['checkpoint'].nunique()} checkpoints",
    )
    print(f"enviado para {REPO_ID}:sim2real/metrics.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--push", action="store_true")
    main(p.parse_args().push)
