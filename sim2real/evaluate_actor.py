"""Evaluate a serialised actor pickle in simulation and optionally save a video."""

import argparse
import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Repo root for `algorithms`, this folder for `portable_actor`, wherever it is run from.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from algorithms.utils.wrapper_gym import get_env
from portable_actor import load_actor

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_N_EPISODES = 20
DEFAULT_COMMAND_TYPE = "direction"

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(actor, env, num_episodes: int, render: bool = False) -> np.ndarray:
    """Run *num_episodes* rollouts and return per-episode returns as an array.

    Only the first episode is rendered: every rendered step keeps a full simulator
    state on the device, and a few episodes of them exhaust GPU memory.
    """
    episode_returns = []
    for episode in range(num_episodes):
        episode_return = 0.0
        observation, _ = env.reset()
        done = truncated = False
        while not done and not truncated:
            action = actor(obs=observation)
            observation, reward, done, truncated, _ = env.step(action)
            if render and episode == 0:
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
        help="Call env.render() at every step of the first episode.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Record the first episode and save it as an .mp4 video.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory to write the video file (defaults to the pickle's directory).",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize the environment.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Load actor (portable format, or a legacy cloudpickle dict)
    print(f"Loading actor from: {args.pickle_path}")
    loaded_actor = load_actor(args.pickle_path)

    # Build environment
    render_trajectory = []

    def render_callback(_, state):
        render_trajectory.append(state)

    render_cb = render_callback if args.save_video else None
    env = get_env(args.env_name, "cuda", render_cb, command_type=args.command_type, randomize=args.randomize)

    # Evaluate
    episode_returns = evaluate(
        actor=loaded_actor["get_action"],
        env=env,
        num_episodes=args.n_episodes,
        # The wrapper hands frames to render_callback only on env.render().
        render=args.render or args.save_video,
    )

    import json

    mean_return = float(episode_returns.mean())
    std_return = float(episode_returns.std())
    results = {
        "pickle_path": args.pickle_path,
        "env_name": args.env_name,
        "command_type": args.command_type,
        "randomize": args.randomize,
        "n_episodes": args.n_episodes,
        "mean_return": mean_return,
        "std_return": std_return,
    }

    print(f"\nResults over {args.n_episodes} episodes:")
    print(f"  Mean return : {mean_return:.4f}")
    print(f"  Std  return : {std_return:.4f}")

    results_path = "results_evaluate_actor.json"

    # If file exists, load, append result, else create new list
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                prev_results = json.load(f)
            if isinstance(prev_results, list):
                prev_results.append(results)
                to_write = prev_results
            else:
                to_write = [prev_results, results]
        except Exception:
            to_write = [results]
    else:
        to_write = [results]

    with open(results_path, "w") as f:
        json.dump(to_write, f, indent=2)
    print(f"Results saved to: {results_path}")
  

    # Save video
    if args.save_video:
        import mujoco.egl

        os.environ["MUJOCO_GL"] = "egl"
        gl_context = mujoco.egl.GLContext(1024, 1024)
        gl_context.make_current()

        video_dir = args.video_dir or os.path.dirname(os.path.abspath(args.pickle_path))
        os.makedirs(video_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(args.pickle_path))[0]
        video_path = os.path.join(video_dir, f"{video_name}.mp4")
        env.save_video(render_trajectory, save_path=video_path)
        print(f"\nVideo saved to: {video_path}")


if __name__ == "__main__":
    main()
