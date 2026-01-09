#!/usr/bin/env python3

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
        # ------------------------------------------------------------- LOCOMOTION -------------------------------------------------------------

        # Go2
        {'env': 'Go2JoystickFlatTerrain', "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2JoystickFlatTerrain-20250904-121138/checkpoints"},
        {'env': 'Go2JoystickFlatTerrain', "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2JoystickFlatTerrain-20250904-195514/checkpoints"},
        {'env': 'Go2JoystickFlatTerrain', "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2JoystickFlatTerrain-20250904-205636/checkpoints"},
        {'env': 'Go2JoystickFlatTerrain', "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2JoystickFlatTerrain-20250904-215800/checkpoints"},
        {'env': 'Go2JoystickFlatTerrain', "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2JoystickFlatTerrain-20250904-225910/checkpoints"}, 

        {"env": "Go2JoystickRoughTerrain", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2JoystickRoughTerrain-20250905-000021/checkpoints"}, 
        {"env": "Go2JoystickRoughTerrain", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2JoystickRoughTerrain-20250905-012621/checkpoints"}, 
        {"env": "Go2JoystickRoughTerrain", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2JoystickRoughTerrain-20250905-025216/checkpoints"}, 
        {"env": "Go2JoystickRoughTerrain", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2JoystickRoughTerrain-20250905-041812/checkpoints"}, 
        {"env": "Go2JoystickRoughTerrain", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2JoystickRoughTerrain-20250905-054419/checkpoints"},

        {"env": "Go2Getup", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2Getup-20250904-193652/checkpoints"},
        {"env": "Go2Getup", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2Getup-20250904-203245/checkpoints"},
        {"env": "Go2Getup", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2Getup-20250904-212840/checkpoints"},
        {"env": "Go2Getup", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2Getup-20250904-222453/checkpoints"},
        {"env": "Go2Getup", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2Getup-20250904-232042/checkpoints"},

        {"env": "Go2Handstand", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2Handstand-20250923-020756/checkpoints"},
        {"env": "Go2Handstand", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2Handstand-20250923-162557/checkpoints"},
        {"env": "Go2Handstand", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2Handstand-20250923-184055/checkpoints"},
        {"env": "Go2Handstand", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2Handstand-20251204-230215/checkpoints"},
        {"env": "Go2Handstand", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2Handstand-20250905-081318/checkpoints"},

        {"env": "Go2Footstand", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2Footstand-20250905-001646/checkpoints"},
        {"env": "Go2Footstand", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2Footstand-20250905-010949/checkpoints"},
        {"env": "Go2Footstand", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2Footstand-20250923-085216/checkpoints"},
        {"env": "Go2Footstand", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2Footstand-20250905-034822/checkpoints"},
        {"env": "Go2Footstand", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/Go2Footstand-20250923-110623/checkpoints"},

        # G1
        {"env": "G1JoystickFlatTerrain", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/G1JoystickFlatTerrain-20250911-191009/checkpoints"},
        {"env": "G1JoystickFlatTerrain", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/G1JoystickFlatTerrain-20250912-004740/checkpoints"},
        {"env": "G1JoystickFlatTerrain", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/G1JoystickFlatTerrain-20250912-062526/checkpoints"},
        {"env": "G1JoystickFlatTerrain", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/G1JoystickFlatTerrain-20250912-120239/checkpoints"},
        {"env": "G1JoystickFlatTerrain", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/G1JoystickFlatTerrain-20250912-174017/checkpoints"},

        # H1
        {"env": "H1InplaceGaitTracking", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/H1InplaceGaitTracking-20250905-112321/checkpoints"},
        {"env": "H1InplaceGaitTracking", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/H1InplaceGaitTracking-20250905-124236/checkpoints"},
        {"env": "H1InplaceGaitTracking", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/H1InplaceGaitTracking-20250905-140100/checkpoints"},
        {"env": "H1InplaceGaitTracking", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/H1InplaceGaitTracking-20250905-151947/checkpoints"},
        {"env": "H1InplaceGaitTracking", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/H1InplaceGaitTracking-20250905-163800/checkpoints"},

        {"env": "H1JoystickGaitTracking", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/H1JoystickGaitTracking-20250905-175648/checkpoints"},
        {"env": "H1JoystickGaitTracking", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/H1JoystickGaitTracking-20250905-191408/checkpoints"},
        {"env": "H1JoystickGaitTracking", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/H1JoystickGaitTracking-20250905-203103/checkpoints"},
        {"env": "H1JoystickGaitTracking", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/H1JoystickGaitTracking-20250905-214821/checkpoints"},
        {"env": "H1JoystickGaitTracking", "model": "PPO", "checkpoint_path": "/CORL/expert/logs/H1JoystickGaitTracking-20250905-230604/checkpoints"},

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
        nf = ppo_params.network_factory
        nf["value_obs_key"] = "state"
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **nf
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

    rollout_path = p['checkpoint_path'].replace("checkpoints", "rollouts")
    os.makedirs(rollout_path, exist_ok=True)
    media.write_video(f"{rollout_path}/rollout.mp4", frames, fps=fps)
    
    return p


def save_results(path_model):
    """Save results to CSV file"""
    df_new = pd.DataFrame.from_dict(path_model)
    
    # Try to load existing results and concatenate
    try:
        df = pd.read_csv("results_expert.csv")
        df_new = pd.concat([df_new, df], ignore_index=True)
    except FileNotFoundError:
        pass
    
    df_new.to_csv("results_expert.csv", index=False)


def display_results():
    """Display the results"""
    try:
        df = pd.read_csv("results_expert.csv")
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
        p = process_model(p)

    # Save results
    save_results(path_model)
    
    # Display results
    display_results()


if __name__ == "__main__":
    main() 