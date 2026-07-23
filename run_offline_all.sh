#!/usr/bin/env bash
# Train an offline algorithm on every dataset in the offline benchmark.
#
# Config layout (base + registry):
#   configs/offline/<algo>/base.yaml -> shared hyperparameters for <algo>
#   configs/offline/_datasets.yaml   -> per-task env / command_type / dt targets + difficulties
# For every task x difficulty this loads base.yaml and injects the dataset
# fields (env, dataset_id, command_type, group[, target_returns]) on the CLI.
#
# wandb logs are stored under: akcit-offlinerl/Offline-Benchmark
#   - project "Offline-Benchmark" comes from <algo>/base.yaml
#   - entity  "akcit-offlinerl"   is exported below
#
# Usage:
#   ./run_offline_all.sh <algo> [task_id ...]
#   ./run_offline_all.sh bc                              # BC on all tasks x difficulties
#   ./run_offline_all.sh iql go2-getup h1-gait-tracking  # subset of tasks
#   SEED=1 ./run_offline_all.sh td3_bc                   # override the training seed
#   FORCE=1 ./run_offline_all.sh bc                      # retrain even already-finished datasets
#
# <algo> is one of: bc awac td3_bc cql iql dt
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 <algo> [task_id ...]   (algo: bc awac td3_bc cql iql dt)" >&2
  exit 2
fi

ALGO="$1"; shift

# Repository root. Override by exporting REPO or passing it on the command line,
# e.g. REPO=/path/to/CORL ./run_offline_all.sh bc
REPO="${REPO:-/home/luana/Documents/OfflineRL/Benchmark/CORL}"
cd "$REPO"

# Refresh the mujoco_playground fork to the latest commit of its pinned branch
# (uv.lock always records a fixed commit, so re-resolve it before training to
# pick up new envs/fixes). Set SKIP_PLAYGROUND_UPGRADE=1 to skip this.
if [[ "${SKIP_PLAYGROUND_UPGRADE:-0}" != "1" ]]; then
  echo "Updating 'playground' to the latest branch commit (SKIP_PLAYGROUND_UPGRADE=1 to skip)..."
  uv sync --upgrade-package playground
fi

PY="$REPO/.venv/bin/python"
DATA_ROOT="$REPO/datasets/playground"
REGISTRY="configs/offline/_datasets.yaml"

# Per-algo module + which field carries the eval env name.
# dt uses "env_name" and also takes per-task "target_returns"; the rest use "env".
case "$ALGO" in
  bc)     MODULE="algorithms.offline.bc_jax";     ENV_FLAG="--env" ;;
  awac)   MODULE="algorithms.offline.awac_jax";   ENV_FLAG="--env" ;;
  td3_bc) MODULE="algorithms.offline.td3_bc_jax"; ENV_FLAG="--env" ;;
  cql)    MODULE="algorithms.offline.cql_jax";    ENV_FLAG="--env" ;;
  iql)    MODULE="algorithms.offline.iql_jax";    ENV_FLAG="--env" ;;
  dt)     MODULE="algorithms.offline.dt_jax";     ENV_FLAG="--env_name" ;;
  *) echo "unknown algo '$ALGO' (expected: bc awac td3_bc cql iql dt)" >&2; exit 2 ;;
esac

BASE_CONFIG="configs/offline/${ALGO}/base.yaml"
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "missing base config: $BASE_CONFIG" >&2
  exit 1
fi

# minari resolves "playground/..." dataset ids from this path
export MINARI_DATASETS_PATH="$REPO/datasets"
# wandb destination: akcit-offlinerl/Offline-Benchmark
export WANDB_ENTITY="akcit-offlinerl"
# wandb auth: export WANDB_API_KEY before submitting to override the default.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
if [[ -z "$WANDB_API_KEY" ]]; then
  echo "WANDB_API_KEY is not set; export it before running (export WANDB_API_KEY=...)." >&2
  exit 1
fi
SEED="${SEED:-0}"
# Skip datasets already trained so a resubmission resumes instead of redoing the
# whole sweep (each finished dataset drops a marker under .done_runs). Without
# this, every resubmission creates a fresh wandb run in the same group -> duplicates.
# Set FORCE=1 to ignore the markers and retrain everything.
FORCE="${FORCE:-0}"
DONE_DIR="$REPO/.done_runs"
mkdir -p "$DONE_DIR"

# Manifest columns: dataset_id \t env \t command_type \t group \t target_returns
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

n_runs="$(printf '%s\n' "$manifest" | wc -l)"
echo "Algorithm: ${ALGO}  (${MODULE})"
echo "Running on ${n_runs} dataset(s) with seed=${SEED}"
echo "config -> ${BASE_CONFIG}"
echo "wandb  -> ${WANDB_ENTITY}/Offline-Benchmark"
echo "minari -> ${MINARI_DATASETS_PATH}"
echo

while IFS=$'\x1f' read -r dataset_id env command_type group target_returns eval_shift; do
  [[ -z "$dataset_id" ]] && continue
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
  "$PY" "${args[@]}"
  touch "$marker"
done <<< "$manifest"

echo "All ${ALGO} runs finished."
