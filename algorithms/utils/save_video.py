import os    
from etils import epath
from pathlib import Path
import yaml
import torch

from jax import numpy as jp

from mujoco_playground import registry
import mediapy as media

from algorithms.offline.any_percent_bc import BC, Actor
from algorithms.utils.save_video import render_callback
from algorithms.utils.wrapper_gym import GymWrapper



def save_video(
    env_name: str,
    actor: Actor,
    device: str,
    command: list[float],
    path_model: str,
    ):
    
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)
    randomizer = registry.get_domain_randomizer(env_name)

    device_rank = int(device.split(":")[-1]) if "cuda" in device else 0

    render_trajectory = []

    def render_callback(_, state):
        render_trajectory.append(state)

    env_wrapped = GymWrapper(
        env,
        num_actors=1,
        seed=1,
        episode_length=env_cfg.episode_length,
        action_repeat=1,
        render_callback=render_callback,
        randomization_fn=randomizer,
        device_rank=device_rank,
        device=device,
    )
            
    state_dim = env_wrapped.observation_space.shape[0]
    action_dim = env_wrapped.action_space.shape[0]
    max_action = 1.0

    command = jp.array([command])

    actor.eval()
    state, _ = env_wrapped.reset()
    env_wrapped.env_state.info["command"] = command
    done = False
    episode_reward = 0.0
    while not done:
        action = actor.act(state, device)
        state, reward, done, _, _ = env_wrapped.step(action)
        env_wrapped.env_state.info["command"] = command
        env_wrapped.render()
        episode_reward += reward

    print("Reward: ", episode_reward)

    import mujoco.egl
    gl_context = mujoco.egl.GLContext(1024, 1024)
    gl_context.make_current()

    # Render
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

    render_every = 2
    fps = 1.0 / env.dt / render_every
    traj = render_trajectory[::render_every]
    frames = env.render(
        traj,
        camera="track",
        height=480,
        width=640,
        scene_option=scene_option,
    )

    media.write_video(os.path.join(path_model, "rollout.mp4"), frames, fps=fps)
