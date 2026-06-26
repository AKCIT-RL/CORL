#!/usr/bin/env bash
# Train PPO on all the newly created benchmark environments using the Warp MJX
# backend (the envs default to impl="warp" in their source config).
#
# Envs covered:
#   - Go2PushRecovery    (Tier 5 - robustness to external pushes)
#   - Go2GetupWalk       (Tier 4 - goal-conditioned stand-up + walk)
#   - H1Getup            (Tier 4 - humanoid fall recovery)
#   - Go2RoughCurriculum (Tier 5 - adaptive terrain curriculum)
#
# The Go2RoughCurriculum env automatically uses the custom curriculum
# auto-reset wrapper (selected by env name inside train_jax_ppo.py).
#
# Usage:
#   ./train_new_benchmark_envs.sh                 # train all envs
#   ./train_new_benchmark_envs.sh Go2GetupWalk    # train a subset
set -euo pipefail

cd "$(dirname "$0")"

SEED="${SEED:-1}"

# Common PPO hyperparameters shared by every run.
COMMON_ARGS=(
    --use_wandb
    --num_evals 50
    --num_minibatches 64
    --num_updates_per_batch 8
    --unroll_length 40
    --num_envs 16384
    --value_obs_key 'state'
    --seed "${SEED}"
    --num_timesteps 100000000
)

run_env() {
  uv run python train_jax_ppo.py --env_name "$1" "${COMMON_ARGS[@]}"
}

if [[ "$#" -gt 0 ]]; then
  ENVS=("$@")
else
  ENVS=(Go2PushRecovery Go2GetupWalk H1Getup Go2RoughCurriculum)
fi

for env in "${ENVS[@]}"; do
  echo "=== Training ${env} ==="
  run_env "${env}"
done
