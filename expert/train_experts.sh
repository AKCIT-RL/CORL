#!/usr/bin/env bash
# Train the PPO experts of the benchmark envs with train_jax_ppo.py. Each run is
# written to expert/logs/<Env>-<YYYYMMDD-HHMMSS>/.
#
# Every run uses the PPO settings in COMMON_ARGS below and the env's default
# Playground config for everything else. The training length matches the experts
# behind the benchmark datasets: 1e9 env steps for the Go2 envs and 2e9 for the
# G1 and H1 humanoids.
#
# Usage:
#   ./expert/train_experts.sh                                  # all benchmark envs, seeds 1-5
#   ./expert/train_experts.sh Go2Getup Go2Handstand            # a subset of envs
#   SEEDS="1 2" ./expert/train_experts.sh Go2Getup             # other seeds
#   NUM_TIMESTEPS=100000000 ./expert/train_experts.sh Go2Getup # override the length
#   USE_WANDB=0 ./expert/train_experts.sh                      # no W&B logging
#
# W&B: runs go to the "Experts-Offline-Benchmark" project of your default entity
# (or of WANDB_ENTITY). Without a W&B account, export WANDB_MODE=offline.
set -euo pipefail

EXPERT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$EXPERT_DIR")"
# The uv environment: .venv by default, UV_PROJECT_ENVIRONMENT when set.
PY="${UV_PROJECT_ENVIRONMENT:-$REPO/.venv}/bin/python"

SEEDS="${SEEDS:-1 2 3 4 5}"
USE_WANDB="${USE_WANDB:-1}"

# PPO settings shared by every run.
COMMON_ARGS=(
  --num_evals 50
  --num_minibatches 64
  --num_updates_per_batch 8
  --unroll_length 40
  --num_envs 16384
  --value_obs_key state
)
[[ "$USE_WANDB" == "1" ]] && COMMON_ARGS+=(--use_wandb)

if [[ "$#" -gt 0 ]]; then
  ENVS=("$@")
else
  ENVS=(
    Go2Getup Go2GetupWalk Go2Footstand Go2Handstand Go2PushRecovery
    Go2JoystickFlatTerrain Go2RoughCurriculum
    G1JoystickFlatTerrain H1JoystickGaitTracking
  )
fi

# Env steps for env $1: 2e9 for the humanoids, 1e9 otherwise.
num_timesteps() {
  case "$1" in
    G1*|H1*) echo 2000000000 ;;
    *)       echo 1000000000 ;;
  esac
}

failed=()
for env in "${ENVS[@]}"; do
  steps="${NUM_TIMESTEPS:-$(num_timesteps "$env")}"
  for seed in $SEEDS; do
    echo "=== ${env} | seed ${seed} | ${steps} steps ==="
    if ! "$PY" "$EXPERT_DIR/train_jax_ppo.py" \
        --env_name "$env" --seed "$seed" --num_timesteps "$steps" \
        "${COMMON_ARGS[@]}"; then
      failed+=("${env} seed ${seed}")
    fi
  done
done

if [[ "${#failed[@]}" -gt 0 ]]; then
  echo "${#failed[@]} failure(s):" >&2
  printf '  %s\n' "${failed[@]}" >&2
  exit 1
fi
