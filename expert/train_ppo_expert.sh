#!/usr/bin/env bash
# Train PPO on the Go2/G1/H1 task using the Warp MJX backend.
#
# The Go2 environment defaults to impl="warp" in its source config, so running
# the standard training script already uses the NVIDIA Warp backend.
set -euo pipefail

cd "$(dirname "$0")"

# Common PPO hyperparameters shared by every run.
COMMON_ARGS=(
    --use_wandb
    --num_evals 50
    --num_minibatches 64
    --num_updates_per_batch 8
    --unroll_length 40
    --num_envs 16384
    --value_obs_key 'state'
)

run_training() {
    local env_name="$1"
    local seed="$2"
    local num_timesteps="$3"

    uv run python train_jax_ppo.py \
        --env_name "${env_name}" \
        "${COMMON_ARGS[@]}" \
        --seed "${seed}" \
        --num_timesteps "${num_timesteps}"
}

for env_name in Go2Footstand Go2Handstand Go2Getup; do
    for seed in 1 2 3 4 5; do
        run_training "${env_name}" "${seed}" 1000000000
    done
done

for env_name in G1JoystickFlatTerrain H1JoystickGaitTracking; do
    for _ in $(seq 1 5); do
        run_training "${env_name}" 1 2000000000
    done
done