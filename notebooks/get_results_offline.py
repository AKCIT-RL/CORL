#!/usr/bin/env python3
"""
Script to evaluate offline RL models and collect results.
This script converts the get_results_offline.ipynb notebook into a standalone Python script.
"""

import os
import sys
import yaml
import pandas as pd
import mujoco.egl
from mujoco_playground import registry
import mediapy as media
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append("..")
from algorithms.utils.wrapper_gym import get_env
from algorithms.offline.any_percent_bc import get_actor_from_checkpoint as get_actor_from_checkpoint_bc, eval_actor as eval_actor_bc
from algorithms.offline.td3_bc import get_actor_from_checkpoint as get_actor_from_checkpoint_td3_bc, eval_actor as eval_actor_td3_bc
from algorithms.offline.sac_n import get_actor_from_checkpoint as get_actor_from_checkpoint_sac_n, eval_actor as eval_actor_sac_n

# Set up environment variables
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# Configuration
N_EPISODES = 20
RENDER = True
SEED = 0

# Model mapping dictionaries
get_actor_from_checkpoint = {
    "BC": get_actor_from_checkpoint_bc,
    "TD3-BC": get_actor_from_checkpoint_td3_bc,
    "SAC-N": get_actor_from_checkpoint_sac_n,
}

eval_actor = {
    "BC": eval_actor_bc,
    "TD3-BC": eval_actor_td3_bc,
    "SAC-N": eval_actor_sac_n,
}


def collect_checkpoint_info():
    """
    Collect information about available checkpoints from the checkpoints directory.
    
    Returns:
        list: List of dictionaries containing checkpoint information
    """
    base_path = Path("../checkpoints")
    info_dict = []
    
    # Iterate through algorithm directories
    for algo_dir in base_path.iterdir():
        if algo_dir.is_dir():
            # Iterate through experiment directories
            for exp_dir in algo_dir.iterdir():
                if exp_dir.is_dir():
                    config_file = exp_dir / "config.yaml"
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config = yaml.safe_load(f)
                            info_dict.append({
                                'checkpoint_path': config.get('checkpoints_path'),
                                'dataset_id': config.get('dataset_id'),
                                'env': config.get('env'),
                                'model': exp_dir.name.split("-")[0],
                                'command_type': config.get('command_type', None)
                            })
    
    return info_dict


def load_existing_results():
    """
    Load existing results from CSV file if it exists.
    
    Returns:
        set: Set of checkpoint paths that have already been evaluated
    """
    try:
        existing_results = pd.read_csv("results_offline.csv")
        existing_checkpoints = set(existing_results['checkpoint_path'].values)
        print(f"Found {len(existing_checkpoints)} existing results")
    except FileNotFoundError:
        existing_checkpoints = set()
        print("No existing results found, starting fresh")
    
    return existing_checkpoints


def filter_unprocessed_checkpoints(path_model, existing_checkpoints):
    """
    Filter out checkpoints that have already been evaluated.
    
    Args:
        path_model (list): List of checkpoint information dictionaries
        existing_checkpoints (set): Set of already processed checkpoint paths
        
    Returns:
        list: Filtered list of checkpoint information
    """
    filtered_paths = [p for p in path_model if p["checkpoint_path"] not in existing_checkpoints]
    print(f"Found {len(filtered_paths)} new checkpoints to process")
    return filtered_paths


def evaluate_checkpoint(p, gl_context):
    """
    Evaluate a single checkpoint and return results.
    
    Args:
        p (dict): Checkpoint information dictionary
        gl_context: MuJoCo GL context for rendering
        
    Returns:
        dict: Updated checkpoint information with evaluation results
    """
    print("-" * 100)
    print(f"ENV: {p['env']}")
    print("-" * 100)
    print()
    
    checkpoint_path = os.path.join("..", p["checkpoint_path"])
    
    # Load config
    with open(os.path.join(checkpoint_path, "config.yaml")) as f:
        config = yaml.safe_load(f)

    # Normalize model names
    if p["model"] == "TD3":
        p["model"] = "TD3-BC"
    if p["model"] == "SAC":
        p["model"] = "SAC-N"

    # Set up rendering callback
    render_trajectory = []

    def render_callback(_, state):
        render_trajectory.append(state)

    # Create environment
    env_wrapped = get_env(env_name=p["env"], device="cuda", render_callback=render_callback, command_type=p["command_type"])

    # Get environment dimensions
    state_dim = env_wrapped.observation_space.shape[0] 
    action_dim = env_wrapped.action_space.shape[0]
    max_action = 1.0

    # Load actor and evaluate
    actor = get_actor_from_checkpoint[p["model"]](
        checkpoint_path=checkpoint_path, 
        state_dim=state_dim, 
        action_dim=action_dim, 
        max_action=max_action
    )
    
    episode_rewards = eval_actor[p["model"]](
        actor=actor, 
        env=env_wrapped, 
        device=config["device"], 
        n_episodes=N_EPISODES, 
        seed=SEED, 
        render=RENDER
    )
    
    # Store results
    p["episode_rewards_mean"] = episode_rewards.mean()
    p["episode_rewards_std"] = episode_rewards.std()

    # Save video if rendering was enabled
    if RENDER and render_trajectory:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        env_wrapped.save_video(render_trajectory, save_path=f"{checkpoint_path}/{p['env']}-{timestamp}.mp4")
        print(f"Saved video: {checkpoint_path}/{p['env']}-{timestamp}.mp4")
    
    print(f"Mean reward: {p['episode_rewards_mean']:.4f} ± {p['episode_rewards_std']:.4f}")
    
    return p


def save_results(path_model):
    """
    Save evaluation results to CSV file.
    
    Args:
        path_model (list): List of checkpoint information with evaluation results
    """
    # Load existing results
    try:
        df = pd.read_csv("results_offline.csv")
    except FileNotFoundError:
        df = pd.DataFrame()

    # Create new results dataframe
    df_new = pd.DataFrame.from_dict(path_model)
    df_new = df_new.dropna(subset=["episode_rewards_mean"])
    
    # Combine with existing results
    df = pd.concat([df, df_new], ignore_index=True)
    df.to_csv("results_offline.csv", index=False)
    
    print(f"Saved {len(df_new)} new results to results_offline.csv")


def main():
    """
    Main function to run the offline RL evaluation pipeline.
    """
    print("Starting offline RL evaluation pipeline...")
    
    # Collect checkpoint information
    print("Collecting checkpoint information...")
    path_model = collect_checkpoint_info()
    print(f"Found {len(path_model)} total checkpoints")
    
    # Load existing results and filter
    existing_checkpoints = load_existing_results()
    path_model = filter_unprocessed_checkpoints(path_model, existing_checkpoints)
    
    if not path_model:
        print("No new checkpoints to evaluate. Exiting.")
        return
    
    # Set up MuJoCo GL context
    print("Setting up MuJoCo GL context...")
    gl_context = mujoco.egl.GLContext(1024, 1024)
    gl_context.make_current()
    
    # Evaluate each checkpoint
    results = []
    for i, p in enumerate(path_model, 1):
        print(f"\nProcessing checkpoint {i}/{len(path_model)}")
        try:
            result = evaluate_checkpoint(p, gl_context)
            results.append(result)
        except Exception as e:
            print(f"Error evaluating checkpoint {p['checkpoint_path']}: {e}")
            continue
    
    # Save results
    if results:
        save_results(results)
        print(f"\nEvaluation complete! Processed {len(results)} checkpoints.")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()
