#!/usr/bin/env bash
# Evaluate exported actors in simulation and record a video of each one.
#
# The env of each actor is read from its metadata. Optional settings:
#   N_EPISODES    episodes per actor (default 20)
#   COMMAND_TYPE  joystick command, e.g. forwardfixed (default: evaluate_actor.py's)
#   VIDEO_DIR     where videos go (default ./videos)
#   RANDOMIZE=1   turn on the Playground domain randomizer
#
# Usage:
#   ./sim2real/evaluate_actor.sh actor-*.pkl
#   COMMAND_TYPE=forwardfixed N_EPISODES=5 ./sim2real/evaluate_actor.sh actor-BC-Go2JoystickFlatTerrain-1a2b3c4d.pkl
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 <actor.pkl> [actor.pkl ...]" >&2
  exit 2
fi

SIM2REAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SIM2REAL_DIR")"
# The uv environment: .venv by default, UV_PROJECT_ENVIRONMENT when set.
PY="${UV_PROJECT_ENVIRONMENT:-$REPO/.venv}/bin/python"

# env_name stored in actor $1's metadata.
actor_env() {
  "$PY" - "$SIM2REAL_DIR" "$1" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from portable_actor import load_actor

print(load_actor(sys.argv[2]).get("env_name") or "")
PY
}

failed=()
for pkl in "$@"; do
  env="$(actor_env "$pkl")"
  if [[ -z "$env" ]]; then
    echo "no env_name in $pkl" >&2
    failed+=("$pkl")
    continue
  fi
  args=(--pickle-path "$pkl" --env-name "$env"
        --n-episodes "${N_EPISODES:-20}" --save-video --video-dir "${VIDEO_DIR:-./videos}")
  [[ -n "${COMMAND_TYPE:-}" ]] && args+=(--command-type "$COMMAND_TYPE")
  [[ "${RANDOMIZE:-0}" == "1" ]] && args+=(--randomize)

  echo "=== $pkl ($env)"
  "$PY" "$SIM2REAL_DIR/evaluate_actor.py" "${args[@]}" || failed+=("$pkl")
done

if [[ "${#failed[@]}" -gt 0 ]]; then
  echo "${#failed[@]} failure(s):" >&2
  printf '  %s\n' "${failed[@]}" >&2
  exit 1
fi
