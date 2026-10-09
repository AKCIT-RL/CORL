#!/usr/bin/env bash
# Held-out regime of the disturbed-locomotion tier for the 65 checkpoints still
# missing (the other 79 were scored locally). Same settings as the SRR evaluation:
# 100 episodes per arm, 50 parallel envs, suite "heldout" of compare_randomize.py.
#
# Needs: the CORL code with the "heldout" suite, the datasets under
# $MINARI_DATASETS_PATH and the checkpoints under checkpoints/<ALGO>/<run>/.
# Resumes from the metrics JSONs, so it can be restarted at any time.
#
#   P=2 ./scripts/run_heldout_remaining.sh
#
# Bring back the whole $METRICS_DIR folder (one JSON per checkpoint).
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
export MINARI_DATASETS_PATH="${MINARI_DATASETS_PATH:-$REPO_ROOT/datasets}"
PY="${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}/bin/python"
P="${P:-2}"
METRICS_DIR="${METRICS_DIR:-logs/compare/metrics_heldout}"
LOG_DIR="${LOG_DIR:-${METRICS_DIR%/}_logs}"
mkdir -p "$METRICS_DIR" "$LOG_DIR"

RUNS=(
  DT-Go2PushRecovery-4638fae0
  DT-Go2PushRecovery-4ad3990d
  DT-Go2PushRecovery-71001b3c
  DT-Go2PushRecovery-78c8e608
  DT-Go2PushRecovery-81b81bd5
  DT-Go2PushRecovery-9774e389
  DT-Go2PushRecovery-a53c936a
  DT-Go2PushRecovery-bd46d75e
  DT-Go2PushRecovery-bd7b417c
  DT-Go2PushRecovery-d86520e7
  DT-Go2PushRecovery-ddf9cf45
  DT-Go2RoughCurriculum-10488c9b
  DT-Go2RoughCurriculum-141709a4
  DT-Go2RoughCurriculum-2658cebb
  DT-Go2RoughCurriculum-4c9eed3a
  DT-Go2RoughCurriculum-56f8b525
  DT-Go2RoughCurriculum-8f4d8d5c
  DT-Go2RoughCurriculum-93eab48d
  DT-Go2RoughCurriculum-99d26741
  DT-Go2RoughCurriculum-9ee7a560
  DT-Go2RoughCurriculum-bc2ebb4d
  DT-Go2RoughCurriculum-bff3b624
  DT-Go2RoughCurriculum-e6020149
  IQL-Go2PushRecovery-777b565f
  IQL-Go2PushRecovery-77bf34f4
  IQL-Go2PushRecovery-78d95485
  IQL-Go2PushRecovery-a0d601f6
  IQL-Go2PushRecovery-b8c6f953
  IQL-Go2PushRecovery-dea75e73
  IQL-Go2PushRecovery-df3cc89c
  IQL-Go2PushRecovery-e43c3873
  IQL-Go2RoughCurriculum-1c55b821
  IQL-Go2RoughCurriculum-1febf357
  IQL-Go2RoughCurriculum-379a841a
  IQL-Go2RoughCurriculum-45c03089
  IQL-Go2RoughCurriculum-52be6a74
  IQL-Go2RoughCurriculum-5fb91021
  IQL-Go2RoughCurriculum-9997123b
  IQL-Go2RoughCurriculum-af04999a
  IQL-Go2RoughCurriculum-c3bab948
  IQL-Go2RoughCurriculum-caac6091
  IQL-Go2RoughCurriculum-d3c94fed
  IQL-Go2RoughCurriculum-e24930c3
  TD3-BC-Go2PushRecovery-70e0f4e8
  TD3-BC-Go2PushRecovery-7f3b0c93
  TD3-BC-Go2PushRecovery-88ae02e3
  TD3-BC-Go2PushRecovery-8a6035ab
  TD3-BC-Go2PushRecovery-8c7d1c2e
  TD3-BC-Go2PushRecovery-8ffd55f9
  TD3-BC-Go2PushRecovery-922d64d0
  TD3-BC-Go2PushRecovery-941a6926
  TD3-BC-Go2PushRecovery-9bff9ba1
  TD3-BC-Go2PushRecovery-a91d08bf
  TD3-BC-Go2RoughCurriculum-2f198241
  TD3-BC-Go2RoughCurriculum-341f0c17
  TD3-BC-Go2RoughCurriculum-67948cd1
  TD3-BC-Go2RoughCurriculum-6cf3cb80
  TD3-BC-Go2RoughCurriculum-7a29d6a4
  TD3-BC-Go2RoughCurriculum-a1bc7972
  TD3-BC-Go2RoughCurriculum-aefc0896
  TD3-BC-Go2RoughCurriculum-bd8c34e5
  TD3-BC-Go2RoughCurriculum-c2141932
  TD3-BC-Go2RoughCurriculum-daedf78b
  TD3-BC-Go2RoughCurriculum-db22f4cf
  TD3-BC-Go2RoughCurriculum-dd85804d
)

missing=0
for run in "${RUNS[@]}"; do
  algo="${run%%-Go2*}"
  [[ -s "checkpoints/$algo/$run/checkpoint_final.npz" && -s "checkpoints/$algo/$run/config.yaml" ]] \
    || { echo "missing checkpoint: checkpoints/$algo/$run"; missing=$((missing + 1)); }
done
(( missing )) && { echo "$missing checkpoints missing, aborting"; exit 1; }

export PY METRICS_DIR LOG_DIR
evaluate() {
  run="$1"; algo="${run%%-Go2*}"
  [[ -s "$METRICS_DIR/$run.json" ]] && { echo "skip   $run"; return 0; }
  if "$PY" -m scripts.compare_randomize --checkpoint_path "checkpoints/$algo/$run/" --device cuda \
       --n_actors 50 --n_episodes 100 --configs heldout --metrics_dir "$METRICS_DIR" \
       > "$LOG_DIR/$run.txt" 2>&1; then
    echo "$(date +%H:%M:%S) ok     $run"
  else
    echo "$(date +%H:%M:%S) FAILED $run (see $LOG_DIR/$run.txt)"
  fi
}
export -f evaluate

echo "queue: ${#RUNS[@]} checkpoints | $P in parallel | metrics: $METRICS_DIR"
started=$(date +%s)
printf '%s\n' "${RUNS[@]}" | xargs -P "$P" -I{} bash -c 'evaluate "$@"' _ {}
done_n=$(for run in "${RUNS[@]}"; do [[ -s "$METRICS_DIR/$run.json" ]] && echo; done | wc -l)
echo "done in $(( ($(date +%s) - started) / 60 )) min | $done_n/${#RUNS[@]} JSONs"
