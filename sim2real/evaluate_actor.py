"""Evaluate a serialised actor pickle in simulation and optionally save a video."""

import argparse
import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

sys.path.append(os.path.join(os.path.dirname(__file__), "../CORL"))

import cloudpickle
import numpy as np
from datetime import datetime

from algorithms.utils.wrapper_gym import get_env

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_N_EPISODES = 20
DEFAULT_COMMAND_TYPE = "direction"

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(actor, env, num_episodes: int, render: bool = False) -> np.ndarray:
    """Run *num_episodes* rollouts and return per-episode returns as an array."""
    episode_returns = []
    for _ in range(num_episodes):
        episode_return = 0.0
        observation, _ = env.reset()
        done = truncated = False
        while not done and not truncated:
            action = actor(obs=observation)
            observation, reward, done, truncated, _ = env.step(action)
            if render:
                env.render()
            episode_return += reward
        episode_returns.append(episode_return)
    return np.array(episode_returns)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a serialised actor pickle in simulation."
    )
    parser.add_argument(
        "--pickle-path",
        type=str,
        required=True,
        help="Path to the .pkl actor file produced by a checkpoint_*.py script.",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        required=True,
        help="Gymnasium environment name (e.g. Go2JoystickFlatTerrain).",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=DEFAULT_N_EPISODES,
        help=f"Number of evaluation episodes (default: {DEFAULT_N_EPISODES}).",
    )
    parser.add_argument(
        "--command-type",
        type=str,
        default=DEFAULT_COMMAND_TYPE,
        help=f"Command type passed to get_env (default: {DEFAULT_COMMAND_TYPE}).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Call env.render() at every step.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Record the trajectory and save an .mp4 video alongside the pickle.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory to write the video file (defaults to the pickle's directory).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Load actor
    print(f"Loading actor from: {args.pickle_path}")
    with open(args.pickle_path, "rb") as f:
        loaded_actor = cloudpickle.load(f)

    # Build environment
    render_trajectory = []

    def render_callback(_, state):
        render_trajectory.append(state)

    render_cb = render_callback if args.save_video else None
    env = get_env(args.env_name, "cuda", render_cb, command_type=args.command_type, randomize=False)

    # Evaluate
    episode_returns = evaluate(
        actor=loaded_actor["get_action"],
        env=env,
        num_episodes=args.n_episodes,
        render=args.render,
    )

    print(f"\nResults over {args.n_episodes} episodes:")
    print(f"  Mean return : {episode_returns.mean():.4f}")
    print(f"  Std  return : {episode_returns.std():.4f}")
  

    # Save video
    if args.save_video:
        import mujoco.egl

        os.environ["MUJOCO_GL"] = "egl"
        gl_context = mujoco.egl.GLContext(1024, 1024)
        gl_context.make_current()

        video_dir = args.video_dir or os.path.dirname(os.path.abspath(args.pickle_path))
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"{args.pickle_path.split('.')[0]}.mp4")
        env.save_video(render_trajectory, save_path=video_path)
        print(f"\nVideo saved to: {video_path}")


if __name__ == "__main__":
    main()
