#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import functools
import os

import jax
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


LOGS_DIR = Path(__file__).resolve().parent / "logs"


def define_model_paths(runs):
    """One entry per run directory (default: every run in expert/logs).

    The env is read from the run name, <Env>-<YYYYMMDD>-<HHMMSS>, as written by
    train_jax_ppo.py.
    """
    if not runs:
        runs = sorted(d for d in LOGS_DIR.iterdir() if (d / "checkpoints").is_dir())
    return [
        {
            "env": Path(run).resolve().name.rsplit("-", 2)[0],
            "model": "PPO",
            "checkpoint_path": str(Path(run).resolve() / "checkpoints"),
        }
        for run in runs
    ]


def eval_expert(env, n_episodes, jit_inference_fn):
    """Evaluate expert model and return episode rewards and rollout"""
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    rng = jax.random.PRNGKey(0)

    rollout = []
    episode_rewards = []
    # Only the first episode is kept for the video: every stored step is a full
    # simulator state on the device, and a few episodes exhaust GPU memory.
    for episode in tqdm(range(n_episodes)):
        rng, reset_rng = jax.random.split(rng)
        state = jit_reset(reset_rng)

        if episode == 0:
            rollout.append(state)
        done = False
        episode_reward = 0.0
        for i in range(env._config.episode_length):
            act_rng, rng = jax.random.split(rng)
            action, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, action)
            if episode == 0:
                rollout.append(state)
            episode_reward += wrapper_torch._jax_to_torch(state.reward).cpu().numpy()
            done = bool(wrapper_torch._jax_to_torch(state.done).cpu().numpy().item())
            if done:
                break
        episode_rewards.append(episode_reward)

    return np.asarray(episode_rewards), rollout

def process_model(p, n_episodes):
    """Process a single model configuration"""
    print("-"*100)
    print(f"ENV: {p['env']}")
    print("-"*100)
    print()

    env = registry.load(p["env"])
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
    except Exception:
        ppo_params = manipulation_params.brax_ppo_config(p["env"])

    ppo_training_params = dict(ppo_params)
    ppo_training_params["num_timesteps"] = 0

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
    
    episode_rewards, rollout = eval_expert(env, n_episodes, jit_inference_fn)
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
    except Exception:
        render_every = 1
        frames = env.render(rollout[::render_every])

    rollout_path = p['checkpoint_path'].replace("checkpoints", "rollouts")
    os.makedirs(rollout_path, exist_ok=True)
    media.write_video(f"{rollout_path}/rollout.mp4", frames, fps=fps)
    
    return p


def save_results(path_model, out):
    """Append the results to the CSV file *out*."""
    df_new = pd.DataFrame.from_dict(path_model)
    
    # Try to load existing results and concatenate
    try:
        df = pd.read_csv(out)
        df_new = pd.concat([df_new, df], ignore_index=True)
    except FileNotFoundError:
        pass
    
    df_new.to_csv(out, index=False)


def display_results(out):
    """Display the results"""
    try:
        df = pd.read_csv(out)
        print("Results:")
        print(df[["env", "episode_rewards_mean", "episode_rewards_std"]].sort_values(by="env", ascending=True))
    except FileNotFoundError:
        print("No results file found.")


def main():
    """Evaluate expert runs and append the returns to a CSV file."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("runs", nargs="*",
                        help="Run directories (default: every run in expert/logs).")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per run.")
    parser.add_argument("--out", default="results_expert.csv", help="CSV file to append to.")
    args = parser.parse_args()

    path_model = define_model_paths(args.runs)
    for p in path_model:
        process_model(p, args.episodes)

    save_results(path_model, args.out)
    display_results(args.out)


if __name__ == "__main__":
    main() 