# Expert fowardfixed
python checkpoint_bc.py --checkpoint-path ../checkpoints/BC/BC-Go2JoystickFlatTerrain-4bb13594 --env-name Go2JoystickFlatTerrain

# Expert foward
python checkpoint_bc.py --checkpoint-path ../checkpoints/BC/BC-Go2JoystickFlatTerrain-6a6ba0be --env-name Go2JoystickFlatTerrain

# Expert direction
python checkpoint_bc.py --checkpoint-path ../checkpoints/BC/BC-Go2JoystickFlatTerrain-fd4e6c50 --env-name Go2JoystickFlatTerrain


# Expert fowardfixed
python checkpoint_awac.py --checkpoint-path ../checkpoints/AWAC/AWAC-Go2JoystickFlatTerrain-b1f58c5a --env-name Go2JoystickFlatTerrain

# Expert foward
python checkpoint_awac.py --checkpoint-path ../checkpoints/AWAC/AWAC-Go2JoystickFlatTerrain-9a9a0a54 --env-name Go2JoystickFlatTerrain

# Expert direction
python checkpoint_awac.py --checkpoint-path ../checkpoints/AWAC/AWAC-Go2JoystickFlatTerrain-f69a2d43 --env-name Go2JoystickFlatTerrain


# Expert fowardfixed
python checkpoint_iql.py --checkpoint-path ../checkpoints/IQL/IQL-Go2JoystickFlatTerrain-170a3036 --env-name Go2JoystickFlatTerrain

# Expert foward
python checkpoint_iql.py --checkpoint-path ../checkpoints/IQL/IQL-Go2JoystickFlatTerrain-f1ba53bb --env-name Go2JoystickFlatTerrain

# Expert direction
python checkpoint_iql.py --checkpoint-path ../checkpoints/IQL/IQL-Go2JoystickFlatTerrain-2aadc1c4 --env-name Go2JoystickFlatTerrain


# Expert fowardfixed
python checkpoint_td3_bc.py --checkpoint-path ../checkpoints/TD3-BC/TD3-BC-Go2JoystickFlatTerrain-68c5bf5c --env-name Go2JoystickFlatTerrain

# Expert foward
python checkpoint_td3_bc.py --checkpoint-path ../checkpoints/TD3-BC/TD3-BC-Go2JoystickFlatTerrain-9da07408 --env-name Go2JoystickFlatTerrain

# Expert direction
python checkpoint_td3_bc.py --checkpoint-path ../checkpoints/TD3-BC/TD3-BC-Go2JoystickFlatTerrain-9073de6c --env-name Go2JoystickFlatTerrain


# PPO last checkpoint
python checkpoint_expert.py --checkpoints-dir ../expert/logs/Go2JoystickFlatTerrain-20250904-225910/checkpoints --checkpoint-step 1008599040 --run-id "20250904-225910-last" --env-name Go2JoystickFlatTerrain
python checkpoint_expert.py --checkpoints-dir ../expert/logs/Go2JoystickRoughTerrain-20250905-054419/checkpoints --checkpoint-step 1008599040 --run-id "20250905-054419-last" --env-name Go2JoystickRoughTerrain
python checkpoint_expert.py --checkpoints-dir ../expert/logs/Go2JoystickRoughTerrain-20260303-201251/checkpoints --checkpoint-step 1008599040 --run-id "20260303-201251-last" --env-name Go2JoystickRoughTerrain


# PPO intermediate checkpoints
python checkpoint_expert.py --checkpoints-dir ../expert/logs/Go2JoystickFlatTerrain-20250904-225910/checkpoints --checkpoint-step 212336640 --run-id "20250904-225910-212M" --env-name Go2JoystickFlatTerrain
python checkpoint_expert.py --checkpoints-dir ../expert/logs/Go2JoystickFlatTerrain-20250904-225910/checkpoints --checkpoint-step 265420800 --run-id "20250904-225910-265M" --env-name Go2JoystickFlatTerrain

python checkpoint_expert.py --checkpoints-dir ../expert/logs/Go2JoystickRoughTerrain-20250905-054419/checkpoints --checkpoint-step 212336640 --run-id "20250905-054419-212M" --env-name Go2JoystickRoughTerrain
python checkpoint_expert.py --checkpoints-dir ../expert/logs/Go2JoystickRoughTerrain-20250905-054419/checkpoints --checkpoint-step 265420800 --run-id "20250905-054419-265M" --env-name Go2JoystickRoughTerrain
python checkpoint_expert.py --checkpoints-dir ../expert/logs/Go2JoystickRoughTerrain-20260303-201251/checkpoints --checkpoint-step 212336640 --run-id "20260303-201251-212M" --env-name Go2JoystickRoughTerrain
python checkpoint_expert.py --checkpoints-dir ../expert/logs/Go2JoystickRoughTerrain-20260303-201251/checkpoints --checkpoint-step 265420800 --run-id "20260303-201251-265M" --env-name Go2JoystickRoughTerrain