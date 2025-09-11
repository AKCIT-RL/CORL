#!/usr/bin/env python3
"""
Script to evaluate random policies on various environments.
Converted from get_results_random.ipynb
"""

import pandas as pd
import numpy as np
import os
import sys

# Set environment variable
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from mujoco_playground import registry  

# Add parent directory to path
sys.path.append("..")
from algorithms.utils.wrapper_gym import GymWrapper


def eval_random(env, action_dim, n_episodes):
    """
    Evaluate a random policy on the given environment.
    
    Args:
        env: The environment to evaluate on
        action_dim: Dimension of the action space
        n_episodes: Number of episodes to run
        
    Returns:
        numpy array of episode rewards
    """
    episode_rewards = []
    for _ in range(n_episodes):
        _, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = np.random.randn(action_dim)
            _, reward, done, _, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    return np.asarray(episode_rewards)


def main():
    """Main function to run random policy evaluation."""
    
    # Define environments and models to evaluate
    path_model = [
        # {'env': 'H1JoystickGaitTracking', "model": "random"},
        # {'env': 'H1InplaceGaitTracking', "model": "random"},
        # {'env': 'Go1JoystickRoughTerrain', "model": "random"}, 
        # {'env': 'Go1JoystickFlatTerrain', "model": "random"},
        # {'env': 'Go1Handstand', "model": "random"}, 
        # {'env': 'Go1Getup', "model": "random"}, 
        # {'env': 'Go1Footstand', "model": "random"},
        # {'env': 'G1JoystickRoughTerrain', "model": "random"}, 
        # {'env': 'G1JoystickFlatTerrain', "model": "random"},
        {'env': 'Go2JoystickRoughTerrain', "model": "random"}, 
        {'env': 'Go2JoystickFlatTerrain', "model": "random"},
        {'env': 'Go2Handstand', "model": "random"}, 
        {'env': 'Go2Getup', "model": "random"}, 
        {'env': 'Go2Footstand', "model": "random"},
    ]

    # Evaluate each environment
    for p in path_model:
        print("-" * 100)
        print(f"ENV: {p['env']}")
        print("-" * 100)
        print()
        
        # Load environment and configuration
        env = registry.load(p["env"])
        env_cfg = registry.get_default_config(p["env"])
        randomizer = registry.get_domain_randomizer(p["env"])

        render_trajectory = []

        def render_callback(_, state):
            render_trajectory.append(state)

        # Wrap environment
        env_wrapped = GymWrapper(
            env,
            num_actors=1,
            seed=1,
            episode_length=env_cfg.episode_length,
            action_repeat=1,
            render_callback=render_callback,
            randomization_fn=randomizer,
            device="cuda",
        )

        state_dim = env_wrapped.observation_space.shape[0] 
        action_dim = env_wrapped.action_space.shape[0]
        max_action = 1.0

        # Evaluate random policy
        episode_rewards = eval_random(env_wrapped, action_dim, 10)
        p["episode_rewards_mean"] = episode_rewards.mean()
        p["episode_rewards_std"] = episode_rewards.std()
        
        print(f"Mean reward: {p['episode_rewards_mean']:.4f}")
        print(f"Std reward: {p['episode_rewards_std']:.4f}")
        print()

    # Create and display results DataFrame
    results_df = pd.DataFrame.from_dict(path_model)
    print("Results:")
    print(results_df)
    
    # Save results to CSV
    results_df.to_csv("results_random.csv", index=False)
    print(f"\nResults saved to results_random.csv")


if __name__ == "__main__":
    main()
