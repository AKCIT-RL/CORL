#!/usr/bin/env python3
"""
Script to evaluate expert models and generate rollouts.
Converted from Jupyter notebook get_results_expert.ipynb
"""

import pandas as pd
import numpy as np
import functools
import os
from datetime import datetime

import jax
import numpy as np
from etils import epath
from tqdm import tqdm

from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground.config import locomotion_params, manipulation_params

from mujoco_playground import registry
from mujoco_playground import wrapper, wrapper_torch

import mediapy as media
import mujoco
import mujoco.egl

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['MUJOCO_GL'] = 'egl'


def define_model_paths():
    """Define the model paths and configurations"""
    path_model = [
        # {'env': 'H1JoystickGaitTracking', "model": "PPO", "checkpoint_path": "../expert_checkpoints/H1JoystickGaitTracking"},
        # {'env': 'H1InplaceGaitTracking', "model": "PPO", "checkpoint_path": "../expert_checkpoints/H1InplaceGaitTracking"},
        # {'env': 'Go1JoystickRoughTerrain', "model": "PPO", "checkpoint_path": "../expert_checkpoints/Go1JoystickRoughTerrain"}, 
        # {'env': 'Go1JoystickFlatTerrain', "model": "PPO", "checkpoint_path": "../expert_checkpoints/Go1JoystickFlatTerrain", "policy_hidden_layer_sizes": (1024, 512, 256)},
        # {'env': 'Go1Handstand', "model": "PPO", "checkpoint_path": "../expert_checkpoints/Go1Handstand"}, 
        # {'env': 'Go1Getup', "model": "PPO", "checkpoint_path": "../expert_checkpoints/Go1Getup", "policy_hidden_layer_sizes": (512, 256, 128)}, 
        # {'env': 'Go1Footstand', "model": "PPO", "checkpoint_path": "../expert_checkpoints/Go1Footstand"},
        # {'env': 'G1JoystickRoughTerrain', "model": "PPO", "checkpoint_path": "../expert_checkpoints/G1JoystickRoughTerrain"}, 
        # {'env': 'G1JoystickFlatTerrain', "model": "PPO", "checkpoint_path": "../expert_checkpoints/G1JoystickFlatTerrain"},
        # {'env': 'PandaPickCube', "model": "PPO", "checkpoint_path": "../expert_checkpoints/PandaPickCube"}, 
        # {'env': 'PandaOpenCabinet', "model": "PPO", "checkpoint_path": "../expert_checkpoints/PandaOpenCabinet"}, 

        # {"env": "Go2Handstand", "model": "PPO", "checkpoint_path": "logs/Go2Handstand-20250729-104513/checkpoints"},
        # {"env": "Go2Handstand", "model": "PPO", "checkpoint_path": "logs/Go2Handstand-20250729-115600/checkpoints"},
        # {"env": "Go2Handstand", "model": "PPO", "checkpoint_path": "logs/Go2Handstand-20250729-130630/checkpoints"},
        # {"env": "Go2Handstand", "model": "PPO", "checkpoint_path": "logs/Go2Handstand-20250729-141705/checkpoints"},
        # {"env": "Go2Handstand", "model": "PPO", "checkpoint_path": "logs/Go2Handstand-20250729-152746/checkpoints"},

        {'env': 'Go2JoystickFlatTerrain', "model": "PPO", "checkpoint_path": "logs/Go2JoystickFlatTerrain-20250818-065703/checkpoints"},
        {'env': 'Go2JoystickFlatTerrain', "model": "PPO", "checkpoint_path": "logs/Go2JoystickFlatTerrain-20250818-115104/checkpoints"},
        {'env': 'Go2JoystickFlatTerrain', "model": "PPO", "checkpoint_path": "logs/Go2JoystickFlatTerrain-20250818-125223/checkpoints"},
        {'env': 'Go2JoystickFlatTerrain', "model": "PPO", "checkpoint_path": "logs/Go2JoystickFlatTerrain-20250818-135346/checkpoints"},
        {'env': 'Go2JoystickFlatTerrain', "model": "PPO", "checkpoint_path": "logs/Go2JoystickFlatTerrain-20250818-145517/checkpoints"}, 

        {"env": "Go2JoystickRoughTerrain", "model": "PPO", "checkpoint_path": "logs/Go2JoystickRoughTerrain-20250818-155634/checkpoints"}, 
        {"env": "Go2JoystickRoughTerrain", "model": "PPO", "checkpoint_path": "logs/Go2JoystickRoughTerrain-20250818-172316/checkpoints"}, 
        {"env": "Go2JoystickRoughTerrain", "model": "PPO", "checkpoint_path": "logs/Go2JoystickRoughTerrain-20250818-184955/checkpoints"}, 
        {"env": "Go2JoystickRoughTerrain", "model": "PPO", "checkpoint_path": "logs/Go2JoystickRoughTerrain-20250818-201636/checkpoints"}, 
        {"env": "Go2JoystickRoughTerrain", "model": "PPO", "checkpoint_path": "logs/Go2JoystickRoughTerrain-20250818-214317/checkpoints"},

        {"env": "Go2Getup", "model": "PPO", "checkpoint_path": "logs/Go2Getup-20250816-191527/checkpoints"},
        {"env": "Go2Getup", "model": "PPO", "checkpoint_path": "logs/Go2Getup-20250817-104354/checkpoints"},
        {"env": "Go2Getup", "model": "PPO", "checkpoint_path": "logs/Go2Getup-20250817-123933/checkpoints"},
        {"env": "Go2Getup", "model": "PPO", "checkpoint_path": "logs/Go2Getup-20250817-172203/checkpoints"},
        {"env": "Go2Getup", "model": "PPO", "checkpoint_path": "logs/Go2Getup-20250819-091222/checkpoints"},

        {"env": "Go2Handstand", "model": "PPO", "checkpoint_path": "logs/Go2Handstand-20250818-231000/checkpoints"},
        {"env": "Go2Handstand", "model": "PPO", "checkpoint_path": "logs/Go2Handstand-20250819-010441/checkpoints"},
        {"env": "Go2Handstand", "model": "PPO", "checkpoint_path": "logs/Go2Handstand-20250819-025956/checkpoints"},
        {"env": "Go2Handstand", "model": "PPO", "checkpoint_path": "logs/Go2Handstand-20250819-045444/checkpoints"},
        {"env": "Go2Handstand", "model": "PPO", "checkpoint_path": "logs/Go2Handstand-20250819-064935/checkpoints"},

        {"env": "Go2Footstand", "model": "PPO", "checkpoint_path": "logs/Go2Footstand-20250817-212125/checkpoints"},
        {"env": "Go2Footstand", "model": "PPO", "checkpoint_path": "logs/Go2Footstand-20250817-231711/checkpoints"},
        {"env": "Go2Footstand", "model": "PPO", "checkpoint_path": "logs/Go2Footstand-20250818-011229/checkpoints"},
        {"env": "Go2Footstand", "model": "PPO", "checkpoint_path": "logs/Go2Footstand-20250818-030725/checkpoints"},
        {"env": "Go2Footstand", "model": "PPO", "checkpoint_path": "logs/Go2Footstand-20250818-050219/checkpoints"},

        # {"env": "G1JoystickFlatTerrain", "model": "PPO", "checkpoint_path": "logs/G1JoystickFlatTerrain-20250806-214823/checkpoints"},
        # {"env": "G1JoystickFlatTerrain", "model": "PPO", "checkpoint_path": "logs/G1JoystickFlatTerrain-20250811-163413/checkpoints"},
        # {"env": "G1JoystickFlatTerrain", "model": "PPO", "checkpoint_path": "logs/G1JoystickFlatTerrain-20250811-163423/checkpoints"},
        # {"env": "G1JoystickFlatTerrain", "model": "PPO", "checkpoint_path": "logs/G1JoystickFlatTerrain-20250813-183648/checkpoints"},
        # {"env": "G1JoystickFlatTerrain", "model": "PPO", "checkpoint_path": "logs/G1JoystickFlatTerrain-20250815-174612/checkpoints"}

    ]
    return path_model


def eval_expert(env, n_episodes, jit_inference_fn):
    """Evaluate expert model and return episode rewards and rollout"""
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    rng = jax.random.PRNGKey(0)

    rollout = []
    episode_rewards = []
    for _ in tqdm(range(n_episodes)):
        rng, reset_rng = jax.random.split(rng)
        state = jit_reset(reset_rng)

        rollout.append(state)
        done = False
        episode_reward = 0.0
        for i in range(env._config.episode_length):
            act_rng, rng = jax.random.split(rng)
            action, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, action)
            rollout.append(state)
            episode_reward += wrapper_torch._jax_to_torch(state.reward).cpu().numpy()
            done = bool(wrapper_torch._jax_to_torch(state.done).cpu().numpy().item())
            if done:
                break
        episode_rewards.append(episode_reward)

    return np.asarray(episode_rewards), rollout


def process_model(p):
    """Process a single model configuration"""
    print("-"*100)
    print(f"ENV: {p['env']}")
    print("-"*100)
    print()

    env = registry.load(p["env"])
    env_cfg = registry.get_default_config(p["env"])
    randomizer = registry.get_domain_randomizer(p["env"])

    # ------------- EXPERT EVALUATION
    ckpt_path = str(epath.Path(p["checkpoint_path"]).resolve())
    FINETUNE_PATH = epath.Path(ckpt_path)
    latest_ckpts = list(FINETUNE_PATH.glob("*"))
    latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
    latest_ckpts.sort(key=lambda x: int(x.name))
    latest_ckpt = latest_ckpts[-1]
    restore_checkpoint_path = latest_ckpt

    try:
        ppo_params = locomotion_params.brax_ppo_config(p["env"])
    except:
        ppo_params = manipulation_params.brax_ppo_config(p["env"])

    ppo_training_params = dict(ppo_params)
    ppo_training_params["num_timesteps"] = 0

    if "policy_hidden_layer_sizes" in p:
        ppo_params["network_factory"]["policy_hidden_layer_sizes"] = p["policy_hidden_layer_sizes"]

    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **ppo_params.network_factory
        )

    train_fn = functools.partial(
        ppo.train,
        **dict(ppo_training_params),
        network_factory=network_factory,
        randomization_fn=randomizer,
    )

    make_inference_fn, params, metrics = train_fn(
        environment=registry.load(p["env"]),
        eval_env=registry.load(p["env"]),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        restore_checkpoint_path=restore_checkpoint_path,
        seed=1,
    )

    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
    
    episode_rewards, rollout = eval_expert(env, 20, jit_inference_fn)
    p["episodes_reward"] = episode_rewards
    p["episode_rewards_mean"] = episode_rewards.mean()
    p["episode_rewards_std"] = episode_rewards.std()

    render_every = 2
    fps = 1.0 / env.dt / render_every
    traj = rollout[::render_every]

    gl_context = mujoco.egl.GLContext(1024, 1024)
    gl_context.make_current()

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

    try:
        frames = env.render(
            traj,
            camera="track",
            scene_option=scene_option,
            width=640,
            height=480,
        )
    except:
        render_every = 1
        frames = env.render(rollout[::render_every])

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    media.write_video(f"rollouts/expert_rollout-{p['checkpoint_path'].split('/')[1]}.mp4", frames, fps=fps)
    
    return p


def save_results(path_model):
    """Save results to CSV file"""
    df_new = pd.DataFrame.from_dict(path_model)
    
    # Try to load existing results and concatenate
    try:
        df = pd.read_csv("rollouts/results_expert.csv")
        df_new = pd.concat([df_new, df], ignore_index=True)
    except FileNotFoundError:
        pass
    
    df_new.to_csv("rollouts/results_expert.csv", index=False)


def display_results():
    """Display the results"""
    try:
        df = pd.read_csv("rollouts/results_expert.csv")
        print("Results:")
        print(df[["env", "episode_rewards_mean", "episode_rewards_std"]].sort_values(by="env", ascending=True))
    except FileNotFoundError:
        print("No results file found.")


def main():
    """Main function to run the expert evaluation"""
    # Get model paths
    path_model = define_model_paths()
    
    # Process each model
    for p in path_model:
        try:
            p = process_model(p)
        except Exception as e:
            print(f"Error processing {p['env']}: {e}")
            continue
    
    # Save results
    save_results(path_model)
    
    # Display results
    display_results()


if __name__ == "__main__":
    main() 