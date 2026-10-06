#!/usr/bin/env bash
# Export trained runs to portable actors (actor-*.pkl, written to the current
# directory).
#
# Each argument is a run directory:
#   - an offline run, checkpoints/<ALGO>/<ALGO>-<Env>-<hash>/ (BC, TD3-BC, IQL or AWAC);
#   - a PPO expert run, expert/logs/<Env>-<YYYYMMDD-HHMMSS>/ (its latest
#     checkpoint; STEP=<n> picks another one).
# The env comes from the run, and the observation and action sizes from the env.
#
# Usage:
#   ./sim2real/generate_checkpoints.sh checkpoints/BC/BC-Go2Getup-1a2b3c4d
#   ./sim2real/generate_checkpoints.sh expert/logs/Go2Getup-20260626-065620
#   STEP=212336640 ./sim2real/generate_checkpoints.sh expert/logs/Go2Getup-20260626-065620
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 <run_dir> [run_dir ...]" >&2
  exit 2
fi

SIM2REAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SIM2REAL_DIR")"
# The uv environment: .venv by default, UV_PROJECT_ENVIRONMENT when set.
PY="${UV_PROJECT_ENVIRONMENT:-$REPO/.venv}/bin/python"

# "<state_dim> <action_dim>" of env $1. The policies read the "state" observation.
env_dims() {
  "$PY" - "$1" <<'PY' 2>/dev/null | tail -n 1
import sys

from mujoco_playground import registry

env = registry.load(sys.argv[1])
obs = env.observation_size
obs = obs["state"] if isinstance(obs, dict) else obs
print(obs[0] if isinstance(obs, tuple) else obs, env.action_size)
PY
}

export_offline() {
  local run="$1" name algo env script
  name="$(basename "$run")"
  case "$name" in
    TD3-BC-*) script="checkpoint_td3_bc.py" ;;
    BC-*)     script="checkpoint_bc.py" ;;
    IQL-*)    script="checkpoint_iql.py" ;;
    AWAC-*)   script="checkpoint_awac.py" ;;
    *) echo "unsupported run '$name' (expected BC, TD3-BC, IQL or AWAC)" >&2; return 1 ;;
  esac
  env="$(sed -n 's/^env: //p' "$run/config.yaml")"
  [[ -n "$env" ]] || { echo "no 'env' in $run/config.yaml" >&2; return 1; }
  read -r state_dim action_dim <<< "$(env_dims "$env")"
  [[ -n "${action_dim:-}" ]] || { echo "could not load env $env" >&2; return 1; }

  echo "=== $name ($env, state_dim=$state_dim, action_dim=$action_dim)"
  "$PY" "$SIM2REAL_DIR/$script" \
    --checkpoint-path "$run" --env-name "$env" \
    --state-dim "$state_dim" --action-dim "$action_dim"
}

export_expert() {
  local run="$1" name env step
  name="$(basename "$run")"
  env="${name%-*-*}"  # <Env>-<YYYYMMDD>-<HHMMSS>
  step="${STEP:-$(ls "$run/checkpoints" | grep -E '^[0-9]+$' | sort -n | tail -n 1)}"
  [[ -n "$step" ]] || { echo "no checkpoint in $run/checkpoints" >&2; return 1; }
  read -r state_dim action_dim <<< "$(env_dims "$env")"
  [[ -n "${action_dim:-}" ]] || { echo "could not load env $env" >&2; return 1; }

  echo "=== $name step $step ($env, state_dim=$state_dim, action_dim=$action_dim)"
  "$PY" "$SIM2REAL_DIR/checkpoint_expert.py" \
    --checkpoints-dir "$run/checkpoints" --checkpoint-step "$step" \
    --run-id "${name#"$env"-}-$step" --env-name "$env" \
    --state-dim "$state_dim" --action-dim "$action_dim"
}

failed=()
for run in "$@"; do
  run="${run%/}"
  if [[ -f "$run/config.yaml" ]]; then
    export_offline "$run" || failed+=("$run")
  elif [[ -d "$run/checkpoints" ]]; then
    export_expert "$run" || failed+=("$run")
  else
    echo "not a run directory: $run" >&2
    failed+=("$run")
  fi
done

if [[ "${#failed[@]}" -gt 0 ]]; then
  echo "${#failed[@]} failure(s):" >&2
  printf '  %s\n' "${failed[@]}" >&2
  exit 1
fi
