"""
Example using standalone joint tracking wrapper.

NO LIBRARY MODIFICATIONS REQUIRED!

This script demonstrates how to use the JointTrackingWrapper
without modifying any library code.
"""

import os
import sys
from datetime import datetime

import numpy as np

# Add parent directory to path
sys.path.append("..")

# Important to avoid allocating all GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import cloudpickle
from utils.joint_tracker import JointTrackingWrapper  # Our standalone wrapper!

from algorithms.utils.wrapper_gym import get_env


def evaluate_with_tracking(actor, env, num_episodes=1, render=False):
    """Evaluate actor and track joint positions."""
    episode_returns = []

    for episode in range(num_episodes):
        episode_return = 0
        observation, _ = env.reset()
        done = truncated = False
        step_count = 0

        while not done and not truncated:
            action = actor(obs=observation)
            observation, reward, done, truncated, info = env.step(action)

            if render:
                env.render()

            episode_return += reward
            step_count += 1

        episode_returns.append(episode_return)
        print(
            f"Episode {episode + 1}/{num_episodes}: Return = {episode_return:}, Steps = {step_count}"
        )

    mean_return = np.mean(episode_returns)
    return mean_return


def main():
    # Configuration
    CHECKPOINT_PATH = "/home/joao/Projetos/AKCIT_RL/CORL/sim2real"
    ENV_NAME = "Go2JoystickFlatTerrain"  # or "G1JoystickFlatTerrain"
    OUTPUT_DIR = "/home/joao/Projetos/AKCIT_RL/CORL/sim2real/joint_tracking_results"

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load actor from checkpoint
    print("Loading actor checkpoint...")
    pickle_path = f"{CHECKPOINT_PATH}/actor.pkl"
    with open(pickle_path, "rb") as f:
        loaded_actor = cloudpickle.load(f)

    print(f"Actor loaded: {loaded_actor['env_name']}")
    print(
        f"State dim: {loaded_actor['state_dim']}, Action dim: {loaded_actor['action_dim']}"
    )

    # Create base environment (using standard get_env - NO MODIFICATIONS!)
    print("\nCreating environment...")
    base_env = get_env(
        ENV_NAME, "cuda", render_callback=None, command_type="fowardfixed"
    )

    # Wrap with our standalone tracking wrapper
    print("Adding joint tracking wrapper...")
    env = JointTrackingWrapper(base_env, dt=0.02)

    print(f"Environment created with tracking: {ENV_NAME}")

    # Run evaluation
    print("\nRunning evaluation...")
    mean_return = evaluate_with_tracking(
        actor=loaded_actor["get_action"], env=env, num_episodes=1, render=False
    )

    print(f"\nMean return: {mean_return:.2f}")

    # Save joint tracking results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Save plot
    plot_path = f"{OUTPUT_DIR}/joint_tracking_{ENV_NAME}_{timestamp}.png"
    print(f"\nSaving joint tracking plot to: {plot_path}")
    env.save_joint_tracking_plot(plot_path)

    # Save CSV
    csv_path = f"{OUTPUT_DIR}/joint_tracking_{ENV_NAME}_{timestamp}.csv"
    print(f"Saving joint tracking CSV to: {csv_path}")
    env.save_joint_tracking_csv(csv_path)

    # Get and print statistics
    print("\nJoint tracking error statistics:")
    stats = env.get_joint_tracking_stats()
    for joint_name, joint_stats in stats.items():
        print(f"\n{joint_name}:")
        print(f"  Mean error: {joint_stats['mean_error']:.6f} rad")
        print(f"  Std error:  {joint_stats['std_error']:.6f} rad")
        print(f"  Max error:  {joint_stats['max_error']:.6f} rad")
        print(f"  RMSE:       {joint_stats['rmse']:.6f} rad")

    print("\nDone! Check the output directory for results.")


if __name__ == "__main__":
    main()
