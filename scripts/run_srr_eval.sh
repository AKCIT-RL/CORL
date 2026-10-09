#!/usr/bin/env bash
# Sim2Real proxy evaluation across algorithms.
#
# Every env with an entry in randomize_gym.OBS_LAYOUTS (all nine benchmark envs)
# can be scored; the custom suite perturbs each one through its own layout.
#
# Resume is keyed on the metrics JSON, which compare_randomize.py writes only after
# a run finishes, so an interrupted run is retried rather than silently skipped.
#
#   ALGOS="IQL CQL" EPISODES=50 ./scripts/run_srr_eval.sh
#
# Tier-5 held-out regime, kept apart from the SRR metrics:
#   SUITE=heldout ENVS="Go2PushRecovery Go2RoughCurriculum" \
#     METRICS_DIR=logs/compare/metrics_heldout ./scripts/run_srr_eval.sh
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
export MINARI_DATASETS_PATH="${MINARI_DATASETS_PATH:-$REPO_ROOT/datasets}"
# The uv environment: .venv by default, UV_PROJECT_ENVIRONMENT when set.
PY="${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}/bin/python"

ALGOS="${ALGOS:-BC AWAC CQL IQL TD3-BC DT}"
ENVS="${ENVS:-Go2JoystickFlatTerrain Go2PushRecovery Go2RoughCurriculum Go2Getup Go2GetupWalk Go2Footstand Go2Handstand G1JoystickFlatTerrain H1JoystickGaitTracking}"
SUITE="${SUITE:-humanoid_gym_relative}"
EPISODES="${EPISODES:-100}"
ACTORS="${ACTORS:-50}"

if [[ -n "${METRICS_DIR:-}" ]]; then
  LOG_DIR="${LOG_DIR:-${METRICS_DIR%/}_logs}"
else
  METRICS_DIR="logs/compare/metrics"
  LOG_DIR="${LOG_DIR:-logs/compare/algorithms}"
fi
mkdir -p "$METRICS_DIR" "$LOG_DIR"

todo=()
for algo in $ALGOS; do
  for env in $ENVS; do
    for d in "checkpoints/$algo/$algo-$env-"*/; do
      [[ -d "$d" ]] && todo+=("$d")
    done
  done
done

total=${#todo[@]}
echo "queue: $total checkpoints | suites: default + $SUITE | $EPISODES episodes"
started=$(date +%s)
failed=0

for i in "${!todo[@]}"; do
  d="${todo[$i]}"
  name="$(basename "${d%/}")"
  pos="[$((i + 1))/$total]"

  if [[ -s "$METRICS_DIR/$name.json" ]]; then
    echo "$pos skip  $name"
    continue
  fi

  echo "$pos run   $name"
  if ! "$PY" -m scripts.compare_randomize \
      --checkpoint_path "$d" --device cuda \
      --n_actors "$ACTORS" --n_episodes "$EPISODES" \
      --configs "$SUITE" --metrics_dir "$METRICS_DIR" > "$LOG_DIR/$name.txt" 2>&1; then
    failed=$((failed + 1))
    echo "         FAILED: $(tail -3 "$LOG_DIR/$name.txt" | tr '\n' ' ' | cut -c1-150)"
  fi
done

echo "done in $(((($(date +%s) - started)) / 60)) min | failures: $failed"
