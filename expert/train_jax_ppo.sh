python train_jax_ppo.py --env_name Go2Getup  --use_wandb \
    --num_evals 20 \
    --num_minibatches 64 \
    --num_updates_per_batch 8 \
    --unroll_length 40 \
    --num_envs 16384 \
    --value_obs_key 'state' \
    --seed 1 \
    --num_timesteps 1000000000

python train_jax_ppo.py --env_name Go2Footstand  --use_wandb \
    --num_evals 20 \
    --num_minibatches 64 \
    --num_updates_per_batch 8 \
    --unroll_length 40 \
    --num_envs 16384 \
    --value_obs_key 'state' \
    --seed 1 \
    --num_timesteps 1000000000

python train_jax_ppo.py --env_name Go2JoystickFlatTerrain  --use_wandb \
    --num_evals 20 \
    --num_minibatches 64 \
    --num_updates_per_batch 8 \
    --unroll_length 40 \
    --num_envs 16384 \
    --value_obs_key 'state' \
    --seed 1 \
    --num_timesteps 1000000000

python train_jax_ppo.py --env_name Go2JoystickRoughTerrain  --use_wandb \
    --num_evals 20 \
    --num_minibatches 64 \
    --num_updates_per_batch 8 \
    --unroll_length 40 \
    --num_envs 16384 \
    --value_obs_key 'state' \
    --seed 1 \
    --num_timesteps 1000000000

python train_jax_ppo.py --env_name Go2Handstand  --use_wandb \
    --num_evals 20 \
    --num_minibatches 64 \
    --num_updates_per_batch 8 \
    --unroll_length 40 \
    --num_envs 16384 \
    --value_obs_key 'state' \
    --seed 1 \
    --num_timesteps 1000000000