#!/usr/bin/env bash
# Sim2Real proxy evaluation: cross-tier coverage (phase 2) + checkpoint sweep (phase 3).
#
# Both phases evaluate `default` (nominal) against `humanoid_gym_medium` (the calibrated
# rung). Logs are skipped when already present, so the script is safe to re-run after
# an interruption.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

export MINARI_DATASETS_PATH="${MINARI_DATASETS_PATH:-$REPO_ROOT/datasets}"
PY=.venv/bin/python
SUITE=humanoid_gym_medium

run_one() {
  local ckpt="$1" out="$2" actors="$3" episodes="$4"
  if [[ -s "$out" ]] && grep -q "RESULTADOS" "$out"; then
    echo "skip  $(basename "$out")"
    return
  fi
  echo "run   $(basename "$out")"
  if ! $PY -m scripts.compare_randomize \
      --checkpoint_path "$ckpt" --device cuda \
      --n_actors "$actors" --n_episodes "$episodes" \
      --configs "$SUITE" > "$out" 2>&1; then
    echo "FALHOU $(basename "$out") -- $(tail -3 "$out" | tr '\n' ' ')"
  fi
}

# ---------------------------------------------------------------- phase 2
# Go2PushRecovery (tier 5) and Go2RoughCurriculum (tier 5), 4 datasets x 2 seeds.
# Both inherit the Go2 joystick observation layout, so the component split applies.
echo "===================== FASE 2: cobertura cross-tier ====================="
mkdir -p logs/compare/phase2
for d in checkpoints/BC/BC-Go2PushRecovery-*/ checkpoints/BC/BC-Go2RoughCurriculum-*/; do
  name="$(basename "${d%/}")"
  run_one "$d" "logs/compare/phase2/${name}.txt" 50 100
done

# ---------------------------------------------------------------- phase 3
# Sweep intermediate checkpoints of two runs with matched nominal score at convergence
# (both reach ~0.98) but very different perturbed tails. Every other checkpoint keeps
# the run tractable while still giving ~10 points to fit the nominal->perturbed curve.
echo "===================== FASE 3: sweep de checkpoints ====================="
mkdir -p logs/compare/sweep
for run in BC-Go2JoystickFlatTerrain-59e9c78e BC-Go2JoystickFlatTerrain-9c6e818a; do
  i=0
  for ck in $(ls checkpoints/BC/"$run"/checkpoint_[0-9]*.npz | sort -t_ -k2 -n); do
    i=$((i + 1))
    (( i % 2 == 0 )) && continue
    step="$(basename "$ck" .npz)"
    run_one "$ck" "logs/compare/sweep/${run}__${step}.txt" 50 50
  done
done

echo "===================== TERMINOU ====================="
