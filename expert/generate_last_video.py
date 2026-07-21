#!/usr/bin/env python3
"""Gera um vídeo do rollout do último checkpoint treinado.

Por padrão usa o diretório de logs mais recente em ``expert/logs`` e o
checkpoint de maior passo dentro dele, renderizando com a câmera ``track``
que acompanha o robô.
"""

import os
import argparse
import functools
import warnings
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Render headless via EGL para evitar tentar abrir um display X11 (GLFW).
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Silencia os avisos do GLFW emitidos quando não há DISPLAY (ambiente headless).
try:
    from glfw import GLFWError

    warnings.filterwarnings("ignore", category=GLFWError)
except Exception:
    pass

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


LOGS_DIR = Path(__file__).resolve().parent / "logs"


def find_latest_run(logs_dir: Path) -> Path:
    """Retorna o diretório de run mais recente (por data de modificação)."""
    runs = [d for d in logs_dir.iterdir() if d.is_dir() and (d / "checkpoints").is_dir()]
    if not runs:
        raise FileNotFoundError(f"Nenhum run com checkpoints encontrado em {logs_dir}")
    runs.sort(key=lambda d: d.stat().st_mtime)
    return runs[-1]


def infer_env_name(run_dir: Path) -> str:
    """Extrai o nome do ambiente do nome do diretório (ex: Go2JoystickFlatTerrain-...)."""
    return run_dir.name.rsplit("-", 2)[0]


def latest_checkpoint(checkpoints_dir: Path) -> epath.Path:
    ckpts = [c for c in epath.Path(str(checkpoints_dir)).glob("*") if c.is_dir()]
    ckpts.sort(key=lambda x: int(x.name))
    return ckpts[-1]


def eval_expert(env, n_episodes, jit_inference_fn, seed=0):
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    rng = jax.random.PRNGKey(seed)

    rollout = []
    episode_rewards = []
    for _ in tqdm(range(n_episodes)):
        rng, reset_rng = jax.random.split(rng)
        state = jit_reset(reset_rng)
        rollout.append(state)
        episode_reward = 0.0
        for _ in range(env._config.episode_length):
            act_rng, rng = jax.random.split(rng)
            action, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, action)
            rollout.append(state)
            episode_reward += float(wrapper_torch._jax_to_torch(state.reward).cpu().numpy())
            done = bool(wrapper_torch._jax_to_torch(state.done).cpu().numpy().item())
            if done:
                break
        episode_rewards.append(episode_reward)

    return np.asarray(episode_rewards), rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None,
                        help="Caminho para o diretório do run. Padrão: o mais recente.")
    parser.add_argument("--env", type=str, default=None,
                        help="Nome do ambiente. Padrão: inferido do nome do run.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0,
                        help="Semente do gerador de números aleatórios do rollout.")
    parser.add_argument("--out", type=str, default=None,
                        help="Caminho do vídeo de saída. Padrão: <run>/rollout.mp4")
    parser.add_argument("--impl", type=str, default=None,
                        help="Override do backend do MJX (ex.: 'jax' para rodar em CPU, "
                             "'warp' para GPU). Padrão: usa o do config do ambiente.")
    args = parser.parse_args()

    run_dir = Path(args.run).resolve() if args.run else find_latest_run(LOGS_DIR)
    env_name = args.env or infer_env_name(run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    restore_checkpoint_path = latest_checkpoint(checkpoints_dir)

    print(f"Run:        {run_dir}")
    print(f"Env:        {env_name}")
    print(f"Checkpoint: {restore_checkpoint_path}")

    config_overrides = {"impl": args.impl} if args.impl else None

    def load_env():
        return registry.load(env_name, config_overrides=config_overrides)

    env = load_env()
    randomizer = registry.get_domain_randomizer(env_name)

    try:
        ppo_params = locomotion_params.brax_ppo_config(env_name)
    except Exception:
        ppo_params = manipulation_params.brax_ppo_config(env_name)

    ppo_training_params = dict(ppo_params)
    ppo_training_params["num_timesteps"] = 0

    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        nf = ppo_params.network_factory
        nf["value_obs_key"] = "state"
        network_factory = functools.partial(ppo_networks.make_ppo_networks, **nf)

    train_fn = functools.partial(
        ppo.train,
        **dict(ppo_training_params),
        network_factory=network_factory,
        randomization_fn=randomizer,
    )

    make_inference_fn, params, _ = train_fn(
        environment=load_env(),
        eval_env=load_env(),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        restore_checkpoint_path=restore_checkpoint_path,
        seed=1,
    )

    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    episode_rewards, rollout = eval_expert(env, args.episodes, jit_inference_fn, seed=args.seed)
    print(f"Reward médio: {episode_rewards.mean():.2f} +/- {episode_rewards.std():.2f}")

    render_every = 2
    fps = 1.0 / env.dt / render_every
    traj = rollout[::render_every]

    gl_context = mujoco.egl.GLContext(1024, 1024)
    gl_context.make_current()

    # Cenas de terreno irregular (ex.: Go2 rough) não definem nenhuma <light>
    # nem <headlight> no XML e usam uma textura de rocha escura — o vídeo fica
    # quase preto. Quando a cena não tem luzes, reforça a headlight e clareia a
    # textura do terreno diretamente no modelo compilado.
    if env.mj_model.nlight == 0:
        env.mj_model.vis.headlight.ambient[:] = [0.4, 0.4, 0.4]
        env.mj_model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
        env.mj_model.vis.headlight.specular[:] = [1.0, 1.0, 1.0]
        env.mj_model.tex_data[:] = np.clip(
            env.mj_model.tex_data.astype(np.float32) * 2.0, 0, 255
        ).astype(np.uint8)

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
        frames = env.render(rollout[::render_every])

    out_path = args.out or str(run_dir / "rollout.mp4")
    media.write_video(out_path, frames, fps=fps)
    print(f"Vídeo salvo em: {out_path}")


if __name__ == "__main__":
    main()
