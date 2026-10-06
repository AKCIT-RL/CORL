#!/usr/bin/env bash
# Train and evaluate an offline algorithm on every dataset in the offline benchmark.
#
# Config layout (base + registry):
#   configs/offline/<algo>/base.yaml -> shared hyperparameters for <algo>
#   configs/offline/_datasets.yaml   -> per-task env / command_type / dt targets + difficulties
# For every task x difficulty this loads base.yaml and injects the dataset
# fields (env, dataset_id, command_type, group[, target_returns]) on the CLI.
#
# Each run is trained and then scored by the SRR evaluation (default +
# humanoid_gym_relative, 100 episodes / 50 actors; see scripts/README.md):
#   1. training ends with proxy.evaluate, which writes
#      logs/compare/metrics/<run>.json and logs eval/proxy_results/* to wandb;
#   2. if that JSON is missing afterwards (the proxy failed, or the process died
#      after training), scripts.recover_proxy finishes the evaluation from the
#      training log and logs it to the same wandb run;
#   3. only then is the run marked done under .done_runs.
# Training logs go to logs/matrix/<group>-seed<SEED>.log.
#
# EVAL_ONLY=1 skips training: every finished checkpoint of the selected datasets
# and seed that has no metrics JSON yet is scored with scripts.compare_randomize
# (results go to the JSON only, not to wandb).
#
# wandb: runs log to the project in <algo>/base.yaml ("Offline-Benchmark") under
# your default entity; export WANDB_ENTITY to log to a team instead. Without a
# wandb account, export WANDB_MODE=offline (or disabled).
#
# Usage:
#   ./run_offline_all.sh <algo> [task_id ...]
#   ./run_offline_all.sh bc                              # BC on all tasks x difficulties
#   ./run_offline_all.sh iql go2-getup h1-gait-tracking  # subset of tasks
#   SEED=1 ./run_offline_all.sh td3_bc                   # override the training seed
#   FORCE=1 ./run_offline_all.sh bc                      # retrain even already-finished datasets
#   EVAL_ONLY=1 ./run_offline_all.sh cql                 # only evaluate existing checkpoints
#
# A failed dataset does not stop the sweep; the failures are listed at the end
# and the script exits with status 1.
#
# <algo> is one of: bc awac td3_bc cql iql dt
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 <algo> [task_id ...]   (algo: bc awac td3_bc cql iql dt)" >&2
  exit 2
fi

ALGO="$1"; shift

# Repository root: the directory holding this script. Override by exporting REPO
# or passing it on the command line, e.g. REPO=/path/to/CORL ./run_offline_all.sh bc
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO"

# Refresh the mujoco_playground fork to the latest commit of its pinned branch
# (uv.lock always records a fixed commit, so re-resolve it before training to
# pick up new envs/fixes). Set SKIP_PLAYGROUND_UPGRADE=1 to skip this.
if [[ "${SKIP_PLAYGROUND_UPGRADE:-0}" != "1" ]]; then
  echo "Updating 'playground' to the latest branch commit (SKIP_PLAYGROUND_UPGRADE=1 to skip)..."
  uv sync --upgrade-package playground
fi

# The uv environment: .venv by default, UV_PROJECT_ENVIRONMENT when set (the
# Docker image uses /opt/venv).
PY="${UV_PROJECT_ENVIRONMENT:-$REPO/.venv}/bin/python"
DATA_ROOT="$REPO/datasets/playground"
REGISTRY="configs/offline/_datasets.yaml"

# Per-algo module, checkpoint folder and which field carries the eval env name.
# dt uses "env_name" and also takes per-task "target_returns"; the rest use "env".
case "$ALGO" in
  bc)     MODULE="algorithms.offline.bc_jax";     CKPT_ALGO="BC";     ENV_FLAG="--env" ;;
  awac)   MODULE="algorithms.offline.awac_jax";   CKPT_ALGO="AWAC";   ENV_FLAG="--env" ;;
  td3_bc) MODULE="algorithms.offline.td3_bc_jax"; CKPT_ALGO="TD3-BC"; ENV_FLAG="--env" ;;
  cql)    MODULE="algorithms.offline.cql_jax";    CKPT_ALGO="CQL";    ENV_FLAG="--env" ;;
  iql)    MODULE="algorithms.offline.iql_jax";    CKPT_ALGO="IQL";    ENV_FLAG="--env" ;;
  dt)     MODULE="algorithms.offline.dt_jax";     CKPT_ALGO="DT";     ENV_FLAG="--env_name" ;;
  *) echo "unknown algo '$ALGO' (expected: bc awac td3_bc cql iql dt)" >&2; exit 2 ;;
esac

BASE_CONFIG="configs/offline/${ALGO}/base.yaml"
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "missing base config: $BASE_CONFIG" >&2
  exit 1
fi

EVAL_ONLY="${EVAL_ONLY:-0}"
DEVICE="${DEVICE:-cuda}"

# minari resolves "playground/..." dataset ids from this path
export MINARI_DATASETS_PATH="$REPO/datasets"
# wandb auth: only training (and the recovery that resumes its wandb run) needs
# it, and only when logging online.
WANDB_MODE="${WANDB_MODE:-online}"
if [[ "$EVAL_ONLY" != "1" && "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]] \
    && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
  echo "wandb is not logged in: export WANDB_API_KEY=... (or run 'wandb login')," >&2
  echo "or export WANDB_MODE=offline to log locally only." >&2
  exit 1
fi
# Live training logs through tee.
export PYTHONUNBUFFERED=1
SEED="${SEED:-0}"
# Skip datasets already trained so a resubmission resumes instead of redoing the
# whole sweep (each finished dataset drops a marker under .done_runs). Without
# this, every resubmission creates a fresh wandb run in the same group -> duplicates.
# Set FORCE=1 to ignore the markers and retrain everything.
FORCE="${FORCE:-0}"
DONE_DIR="$REPO/.done_runs"
mkdir -p "$DONE_DIR"

# SRR evaluation: same suite and budget as algorithms/utils/proxy.py and
# scripts/run_srr_eval.sh, so every path writes comparable numbers.
SRR_SUITE="humanoid_gym_relative"
SRR_EPISODES=100
SRR_ACTORS=50
METRICS_DIR="logs/compare/metrics"
EVAL_LOG_DIR="logs/compare/algorithms"
TRAIN_LOG_DIR="logs/matrix"
mkdir -p "$METRICS_DIR" "$EVAL_LOG_DIR" "$TRAIN_LOG_DIR"

# Manifest columns: dataset_id \x1f env \x1f command_type \x1f group \x1f target_returns \x1f eval_shift
# Optional args filter by task_id. Datasets missing on disk are downloaded from
# the Hugging Face hub (akcit-rl/playground) into DATA_ROOT before being emitted.
manifest="$("$PY" - "$REGISTRY" "$DATA_ROOT" "$ALGO" "$@" <<'PY'
import os, sys, yaml, json

registry = yaml.safe_load(open(sys.argv[1]))
data_root = sys.argv[2]
algo = sys.argv[3]
filters = set(sys.argv[4:])

# datasets/playground mirrors this Hugging Face dataset repo one-to-one.
HF_REPO = "akcit-rl/playground"


def ensure_dataset(task, diff):
    """Return True if datasets/playground/<task>/<diff>-v0 is available locally,
    downloading it from Hugging Face when it is missing."""
    local_dir = os.path.join(data_root, task, f"{diff}-v0")
    if os.path.isdir(local_dir):
        return True
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        print(f"[download] huggingface_hub unavailable, skipping {task}/{diff}-v0: {e}", file=sys.stderr)
        return False
    print(f"[download] {task}/{diff}-v0 missing locally; fetching from {HF_REPO}", file=sys.stderr)
    try:
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            local_dir=data_root,
            allow_patterns=[
                f"{task}/{diff}-v0/*",
                f"{task}/namespace_metadata.json",
                "namespace_metadata.json",
            ],
        )
    except Exception as e:
        print(f"[download] failed for {task}/{diff}-v0: {e}", file=sys.stderr)
        return False
    return os.path.isdir(local_dir)


for task, info in registry["tasks"].items():
    if filters and task not in filters:
        continue
    env = info["env"]
    ct = info.get("command_type")
    ct = "" if ct is None else str(ct)
    tr = info.get("dt_target_returns") or []
    tr = "[%s]" % ", ".join(str(x) for x in tr) if tr else ""
    # Optional Tier-5 shifted-eval overrides -> compact single-line JSON (no
    # separators/newlines) so it survives the \x1f-delimited manifest.
    es = info.get("eval_shift")
    es = json.dumps(es, separators=(",", ":")) if es else ""
    for diff in registry["difficulties"]:
        if not ensure_dataset(task, diff):
            continue
        dataset_id = f"playground/{task}/{diff}-v0"
        group = f"{algo}-{task}-{diff}"
        print("\x1f".join([dataset_id, env, ct, group, tr, es]))
PY
)"

if [[ -z "$manifest" ]]; then
  echo "No datasets found under $DATA_ROOT and none could be downloaded from Hugging Face (filters: $*)" >&2
  exit 1
fi

# Whether the run in checkpoint dir $1 already has its SRR metrics JSON. Same
# resume rule as scripts/run_srr_eval.sh: the JSON is written only at the end.
has_metrics() {
  [[ -s "$METRICS_DIR/$(basename "$1").json" ]]
}

# Score checkpoint dir $1 with the SRR suite (JSON only, no wandb).
srr_eval() {
  local ckpt_dir="$1" name
  name="$(basename "$ckpt_dir")"
  echo "--- SRR eval ${name} (default + ${SRR_SUITE}, ${SRR_EPISODES} episodes)"
  "$PY" -m scripts.compare_randomize \
    --checkpoint_path "$ckpt_dir" --device "$DEVICE" \
    --n_actors "$SRR_ACTORS" --n_episodes "$SRR_EPISODES" \
    --configs "$SRR_SUITE" < /dev/null > "$EVAL_LOG_DIR/$name.txt" 2>&1
}

# Finished checkpoint dirs (with checkpoint_final.npz) of dataset $1 at the
# benchmark seed $SEED. DT runs trained before --seed reached dt_jax all have
# seed 10 and are the benchmark's seed 0.
finished_checkpoints() {
  "$PY" - "checkpoints/$CKPT_ALGO" "$1" "$SEED" "$CKPT_ALGO" <<'PY'
import sys
from pathlib import Path

import yaml

root, dataset_id, seed, algo = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
for cfg_file in sorted(root.glob(f"{algo}-*/config.yaml")):
    if not (cfg_file.parent / "checkpoint_final.npz").is_file():
        continue
    cfg = yaml.safe_load(cfg_file.read_text()) or {}
    run_seed = cfg.get("seed")
    if algo == "DT" and run_seed == 10:
        run_seed = 0
    if cfg.get("dataset_id") == dataset_id and run_seed == seed:
        print(cfg_file.parent)
PY
}

# Last "Checkpoints path: ..." printed by the trainer in log $1.
checkpoint_dir_from_log() {
  sed 's/\x1b\[[0-9;]*m//g' "$1" | sed -n 's/^.*Checkpoints path: \(\S*\).*$/\1/p' | tail -n 1
}

n_runs="$(printf '%s\n' "$manifest" | wc -l)"
echo "Algorithm: ${ALGO}  (${MODULE})"
if [[ "$EVAL_ONLY" == "1" ]]; then
  echo "EVAL_ONLY: scoring existing checkpoints of ${n_runs} dataset(s) with seed=${SEED}"
else
  echo "Training + evaluating ${n_runs} dataset(s) with seed=${SEED}"
fi
echo "config -> ${BASE_CONFIG}"
echo "wandb  -> ${WANDB_ENTITY:-<default entity>}/Offline-Benchmark (mode: ${WANDB_MODE})"
echo "minari -> ${MINARI_DATASETS_PATH}"
echo "SRR    -> default + ${SRR_SUITE}, ${SRR_EPISODES} episodes / ${SRR_ACTORS} actors -> ${METRICS_DIR}"
echo

trained=0
evaluated=0
failed=()

while IFS=$'\x1f' read -r dataset_id env command_type group target_returns eval_shift; do
  [[ -z "$dataset_id" ]] && continue

  if [[ "$EVAL_ONLY" == "1" ]]; then
    mapfile -t ckpts < <(finished_checkpoints "$dataset_id")
    if [[ "${#ckpts[@]}" -eq 0 ]]; then
      echo "=== ${ALGO} | ${dataset_id}: no finished checkpoint with seed=${SEED} ==="
      continue
    fi
    for ckpt_dir in "${ckpts[@]}"; do
      if has_metrics "$ckpt_dir"; then
        echo "=== skip $(basename "$ckpt_dir") (already evaluated) ==="
      elif srr_eval "$ckpt_dir"; then
        evaluated=$((evaluated + 1))
      else
        echo "    FAILED: see $EVAL_LOG_DIR/$(basename "$ckpt_dir").txt" >&2
        failed+=("eval $(basename "$ckpt_dir")")
      fi
    done
    continue
  fi

  marker="$DONE_DIR/${group}-seed${SEED}.done"
  if [[ "$FORCE" != "1" && -f "$marker" ]]; then
    echo "=== skip ${ALGO} | ${dataset_id} (already done: ${group}, seed=${SEED}; FORCE=1 to redo) ==="
    continue
  fi
  echo "=== ${ALGO} | ${dataset_id} (env=${env}, command_type=${command_type:-None}) ==="
  args=(
    -m "$MODULE"
    --config_path "$BASE_CONFIG"
    "$ENV_FLAG" "$env"
    --dataset_id "$dataset_id"
    --group "$group"
    --seed "$SEED"
  )
  if [[ -n "$command_type" ]]; then
    args+=(--command_type "$command_type")
  fi
  if [[ -n "$eval_shift" ]]; then
    args+=(--eval_shift "$eval_shift")
  fi
  if [[ "$ALGO" == "dt" && -n "$target_returns" ]]; then
    args+=(--target_returns "$target_returns")
  fi

  # recover_proxy reads the wandb run and the checkpoint dir from this log and
  # derives the .done_runs marker from its name. Children get /dev/null as stdin:
  # the loop's stdin is the manifest.
  train_log="$TRAIN_LOG_DIR/${group}-seed${SEED}.log"
  train_ok=1
  "$PY" "${args[@]}" < /dev/null 2>&1 | tee "$train_log" || train_ok=0
  ckpt_dir="$(checkpoint_dir_from_log "$train_log")"

  if [[ "$train_ok" == "1" && -n "$ckpt_dir" ]] && has_metrics "$ckpt_dir"; then
    trained=$((trained + 1))
    touch "$marker"
    continue
  fi

  # Training failed or ended without the SRR metrics. recover_proxy finishes the
  # evaluation when training itself completed (it refuses otherwise), logs it to
  # the same wandb run and touches the marker.
  if [[ -z "$ckpt_dir" ]]; then
    echo "    FAILED: no checkpoint dir in $train_log" >&2
    failed+=("train ${group}")
    continue
  fi
  echo "--- SRR metrics missing for $(basename "$ckpt_dir"); recovering from $train_log"
  if "$PY" -m scripts.recover_proxy "$train_log" --device "$DEVICE" < /dev/null; then
    trained=$((trained + 1))
  else
    failed+=("$([[ "$train_ok" == "1" ]] && echo eval || echo train) ${group}")
  fi
done <<< "$manifest"

echo
if [[ "$EVAL_ONLY" == "1" ]]; then
  echo "${ALGO}: ${evaluated} checkpoint(s) evaluated."
else
  echo "${ALGO}: ${trained} run(s) trained and evaluated."
fi
echo "SRR metrics in ${METRICS_DIR}"
if [[ "${#failed[@]}" -gt 0 ]]; then
  echo "${#failed[@]} failure(s):" >&2
  printf '  %s\n' "${failed[@]}" >&2
  exit 1
fi
