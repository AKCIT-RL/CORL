"""Check the custom randomization suite on every env in OBS_LAYOUTS.

For each env:
  1. layout   -- with the env's own sensor noise off, every OBS_LAYOUTS slice (and
                 mirror) equals the simulator quantity it claims to hold;
  2. obs      -- the suite perturbs exactly the mapped slices, leaves every other
                 entry untouched and moves each mirror by the same amount;
  3. domain   -- friction, payload and motor strength stay inside the suite's
                 relative ranges and touch nothing else;
  4. rollout  -- the GymWrapper runs the suite end to end with finite observations,
                 and the env exposes the rng the observation noise needs.

CPU-bound (JIT + MJX rollouts): on a cluster, run it as a batch job.

Usage:
  python -m scripts.check_randomize_envs [--envs Go2Getup H1JoystickGaitTracking]
"""
import argparse
import sys
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from mujoco_playground import registry

from algorithms.utils.randomize_gym import (
   OBS_LAYOUTS,
   get_env,
   get_predefined_randomize_configs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE = REPO_ROOT / "configs/randomize/humanoid_gym_relative.yaml"
FRICTION = (0.5, 1.666667)
PAYLOAD = 0.065762
MOTOR = (0.95, 1.05)
ATOL = 1e-4


def _call(fn, data):
   """Playground sensor helpers take the IMU body on G1/H1 and nothing on Go2."""
   try:
      return fn(data)
   except TypeError:
      return fn(data, "pelvis")


def _vec(obs, layout):
   return np.asarray(obs[layout.key] if layout.key is not None else obs)


def check_layout(env_name, layout, key, n_steps=10):
   cfg = registry.get_default_config(env_name)
   for noise in ("noise_config", "obs_noise"):
      if noise in cfg:
         cfg[noise].level = 0.0
   if "impl" in cfg:
      cfg.impl = "jax"
   env = registry.load(env_name, config=cfg)
   reset, step = jax.jit(env.reset), jax.jit(env.step)
   state = reset(key)
   worst = {}
   for i in range(n_steps + 1):
      if i:
         key, akey = jax.random.split(key)
         state = step(state, jax.random.uniform(akey, (env.action_size,), minval=-1, maxval=1))
      v, d = _vec(state.obs, layout), state.data
      expected = {
         "joint_angles": np.asarray(d.qpos[7:] - env._default_pose),
         "joint_vel": np.asarray(d.qvel[6:]),
         "gyro": np.asarray(_call(env.get_gyro, d)),
      }
      if hasattr(env, "_pelvis_imu_site_id"):
         # G1 observes the downward gravity in the pelvis IMU frame; its
         # get_gravity sensor is the opposite (up) vector.
         expected["gravity"] = np.asarray(
            d.site_xmat[env._pelvis_imu_site_id].T @ jnp.array([0.0, 0.0, -1.0]))
      elif hasattr(env, "get_gravity"):
         try:
            expected["gravity"] = np.asarray(_call(env.get_gravity, d))
         except Exception:
            pass
      if hasattr(env, "get_local_linvel"):
         expected["linvel"] = np.asarray(_call(env.get_local_linvel, d))
      mirrors = {
         "joint_vel": np.asarray(d.qvel[6:]),
         "joint_angles": np.asarray(d.qpos[7:] - state.info["motor_targets"])
         if "motor_targets" in state.info else None,
      }
      for comp, (a, b) in layout.slices.items():
         if comp in expected:
            worst[comp] = max(worst.get(comp, 0.0), float(np.abs(v[a:b] - expected[comp]).max()))
         elif comp == "gravity":
            worst["|gravity|-1"] = max(worst.get("|gravity|-1", 0.0), abs(float(np.linalg.norm(v[a:b])) - 1))
      for comp, spans in layout.mirrors.items():
         for a, b in spans:
            ref = mirrors.get(comp)
            if ref is None:
               raise AssertionError(f"no reference for mirror {comp}[{a}:{b}]")
            worst[f"mirror {comp}"] = max(worst.get(f"mirror {comp}", 0.0), float(np.abs(v[a:b] - ref).max()))
   bad = {k: e for k, e in worst.items() if e > ATOL}
   assert not bad, f"layout mismatch (max abs error): {bad}"
   return env, state, worst


def check_obs(env_name, layout, obs, key):
   rc = get_predefined_randomize_configs("custom", str(SUITE), env_name)
   fns = rc.get_functions()
   base = _vec(obs, layout)
   out = np.asarray(fns.observation_randomize(obs, key))
   assert out.shape == base.shape, f"shape {out.shape} != {base.shape}"
   changed = ~np.isclose(out, base, atol=0, rtol=0)
   allowed = np.zeros_like(changed)
   for comp, (a, b) in layout.slices.items():
      if getattr(rc.configs, comp, None) is None:
         continue  # in the layout but not in the suite (e.g. linvel): must stay put
      allowed[a:b] = True
      assert changed[a:b].any(), f"{comp} was not perturbed"
      for ma, mb in layout.mirrors.get(comp, ()):
         allowed[ma:mb] = True
         np.testing.assert_allclose(out[ma:mb] - base[ma:mb], out[a:b] - base[a:b], atol=1e-5,
                                    err_msg=f"mirror {comp}[{ma}:{mb}] moved differently")
   assert not (changed & ~allowed).any(), f"touched unmapped entries {np.flatnonzero(changed & ~allowed)}"
   # batched path, as the wrapper calls it
   batch = {k: jnp.stack([v, v]) for k, v in obs.items()} if isinstance(obs, dict) else jnp.stack([obs, obs])
   out_b = np.asarray(fns.observation_randomize(batch, jax.random.split(key, 2)))
   assert out_b.shape == (2,) + base.shape, f"batched shape {out_b.shape}"
   return int(changed.sum())


def check_domain(env_name, env, key, n=256):
   fns = get_predefined_randomize_configs("custom", str(SUITE), env_name).get_functions()
   m0 = env.mjx_model
   mv, _ = fns.domain_randomize(m0, jax.random.split(key, n))
   mu0, mass0 = float(m0.geom_friction[0, 0]), np.asarray(m0.body_mass)
   total, kp0 = float(mass0.sum()), np.asarray(m0.actuator_gainprm[:, 0])

   ratio = np.asarray(mv.geom_friction[:, 0, 0]) / mu0
   assert FRICTION[0] - 1e-5 <= ratio.min() and ratio.max() <= FRICTION[1] + 1e-5, f"friction ratio {ratio.min()}..{ratio.max()}"
   rest = np.asarray(mv.geom_friction).copy(); rest[:, 0, 0] = np.asarray(m0.geom_friction)[0, 0]
   assert np.allclose(rest, np.asarray(m0.geom_friction)[None]), "friction changed outside geom 0 / slot 0"

   dm = np.asarray(mv.body_mass) - mass0[None]
   assert np.abs(dm[:, 1]).max() <= PAYLOAD * total + 1e-5, f"payload {np.abs(dm[:, 1]).max():.3f} kg > {PAYLOAD * total:.3f}"
   assert np.allclose(np.delete(dm, 1, axis=1), 0), "mass changed outside body 1"

   kr = np.asarray(mv.actuator_gainprm[:, :, 0]) / kp0[None]
   assert MOTOR[0] - 1e-5 <= kr.min() and kr.max() <= MOTOR[1] + 1e-5, f"motor scale {kr.min()}..{kr.max()}"
   assert np.allclose(np.asarray(mv.actuator_biasprm[:, :, 1]), -np.asarray(mv.actuator_gainprm[:, :, 0])), "PD bias no longer -kp"
   return {
      "mu0": mu0, "friction": (round(ratio.min() * mu0, 3), round(ratio.max() * mu0, 3)),
      "payload_kg": round(float(np.abs(dm[:, 1]).max()), 2), "payload_cap_kg": round(PAYLOAD * total, 2),
      "motor": (round(float(kr.min()), 3), round(float(kr.max()), 3)),
   }


def check_rollout(env_name, layout, n_actors=4, n_steps=20):
   env = get_env(device="cpu", num_actors=n_actors, randomize_configs="custom",
                 randomize_options=str(SUITE), env_name=env_name)
   obs, _ = env.reset()
   assert "rng" in env.env_state.info, "env info has no rng: the observation noise would be skipped"
   for _ in range(n_steps):
      obs, *_ = env.step(np.random.uniform(-1, 1, (n_actors,) + env.action_space.shape))
      assert np.isfinite(obs).all(), "non-finite observation"
   assert obs.shape == (n_actors, layout.size), f"obs shape {obs.shape}"


def main():
   ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
   ap.add_argument("--envs", nargs="+", default=list(OBS_LAYOUTS))
   args = ap.parse_args()

   failed = []
   for env_name in args.envs:
      layout = OBS_LAYOUTS[env_name]
      key = jax.random.PRNGKey(0)
      print(f"=== {env_name}", flush=True)
      try:
         env, state, worst = check_layout(env_name, layout, key)
         print(f"  layout  ok  max|err| {({k: f'{e:.1e}' for k, e in worst.items()})}")
         n = check_obs(env_name, layout, state.obs, jax.random.PRNGKey(1))
         print(f"  obs     ok  {n} entries perturbed, rest untouched")
         print(f"  domain  ok  {check_domain(env_name, env, jax.random.PRNGKey(2))}")
         check_rollout(env_name, layout)
         print("  rollout ok")
      except Exception as e:
         failed.append(env_name)
         print(f"  FAIL: {e}")
         traceback.print_exc()
   print(f"\n{len(args.envs) - len(failed)}/{len(args.envs)} envs ok" + (f"; falharam: {failed}" if failed else ""))
   return 1 if failed else 0


if __name__ == "__main__":
   sys.exit(main())
