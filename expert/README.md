# `expert/` — expert policies (PPO)

Trains the expert policies for the benchmark tasks with PPO (Brax) on the
[MuJoCo Playground](https://github.com/AKCIT-RL/mujoco_playground) envs, evaluates the
checkpoints and records rollout videos. To take an expert to the robot or to Isaac, export
it with `sim2real/checkpoint_expert.py` (see [sim2real/README.md](../sim2real/README.md)).

## Files

| File | Role |
| --- | --- |
| `train_jax_ppo.py` | Trains PPO on a Playground env; saves orbax checkpoints and a video of the final rollout. |
| `train_experts.sh` | Trains the experts of every benchmark env (or a subset), over several seeds. |
| `generate_last_video.py` | Reloads a run's latest checkpoint, evaluates it and records a video. |
| `get_results_expert.py` | Evaluates several runs (default: all of `logs/`) and appends the returns to `results_expert.csv`. |
| `logs/` | Training output. Ignored by git. |

---

## Training

`train_experts.sh` trains the experts of the benchmark envs with the settings used for the
benchmark datasets:

```bash
./expert/train_experts.sh                                  # all benchmark envs, seeds 1-5
./expert/train_experts.sh Go2Getup Go2Handstand            # a subset of envs
SEEDS="1 2" ./expert/train_experts.sh Go2Getup             # other seeds
NUM_TIMESTEPS=100000000 ./expert/train_experts.sh Go2Getup # override the length
USE_WANDB=0 ./expert/train_experts.sh                      # no W&B logging
```

| Setting | Value |
| --- | --- |
| Envs | `Go2Getup`, `Go2GetupWalk`, `Go2Footstand`, `Go2Handstand`, `Go2PushRecovery`, `Go2JoystickFlatTerrain`, `Go2RoughCurriculum`, `G1JoystickFlatTerrain`, `H1JoystickGaitTracking` (or the envs passed as arguments) |
| Seeds | `SEEDS`, default `1 2 3 4 5` |
| Env steps | 1e9 for the Go2 envs, 2e9 for G1 and H1 (`NUM_TIMESTEPS` overrides) |
| PPO | `--num_envs 16384 --num_minibatches 64 --num_updates_per_batch 8 --unroll_length 40 --num_evals 50 --value_obs_key state` |

A failed run doesn't stop the sweep; failures are listed at the end and the script exits
with status 1. Brax rounds the length up to whole training epochs, so the final step is a
bit above the requested one (e.g. 1,284,505,600 for 1e9 steps on `Go2Getup`).

### A single run

```bash
uv run python expert/train_jax_ppo.py \
  --env_name Go2Handstand \
  --use_wandb \
  --num_timesteps 1000000000 \
  --num_envs 16384 --num_minibatches 64 --num_updates_per_batch 8 \
  --unroll_length 40 --num_evals 50 \
  --value_obs_key state \
  --seed 1
```

Runs are written to `expert/logs/` wherever the script is run from (`--logdir` changes
that).

Hyperparameters start from the env's default Playground config
(`locomotion_params.brax_ppo_config`); only flags passed explicitly override it. Each env
uses the MJX backend from its own config: Warp for every benchmark env except
`Go2RoughCurriculum`, which uses `jax`.

### Main flags

| Flag | Script default | Description |
| --- | --- | --- |
| `--env_name` | — (required) | Env from the Playground registry. |
| `--logdir` | `expert/logs` | Directory the run folder is created in. |
| `--seed` | `1` | — |
| `--num_timesteps` | env config | Total environment steps. |
| `--num_envs` | env config | Parallel envs. |
| `--num_evals` | env config | Evaluations (and checkpoints) over training. |
| `--use_wandb` | `False` | Logs to the W&B project `Experts-Offline-Benchmark` (entity: `WANDB_ENTITY`, else your default). |
| `--use_tb` | `False` | Logs to TensorBoard in the run directory. |
| `--domain_randomization` | `False` | Trains with the env's domain randomizer. |
| `--load_checkpoint_path` | — | Resumes from a checkpoint (a checkpoints directory uses the highest step). |
| `--play_only` | `False` | Doesn't train, only produces the rollout (use with `--load_checkpoint_path`). |
| `--suffix` | — | Suffix for the run name. |
| `--policy_hidden_layer_sizes`, `--value_hidden_layer_sizes` | env config | Layer sizes, e.g. `512,256,128`. |
| `--policy_obs_key`, `--value_obs_key` | `state` | Observation key for each network. The sweeps pass `--value_obs_key state`. |

There are also `--learning_rate`, `--entropy_cost`, `--discounting`, `--batch_size`,
`--reward_scaling`, `--episode_length`, `--clipping_epsilon`, `--max_grad_norm`,
`--action_repeat`, `--normalize_observations`, `--num_eval_envs` and `--vision`.

`Go2RoughCurriculum` trains with its own auto-reset wrapper
(`rough_curriculum.wrap_for_curriculum_training`), which moves each agent to a difficulty
tile based on its performance; the other envs use the standard Brax wrapper. For
goal-reaching tasks (such as `Go2GetupWalk`), the log also includes `eval/arrival_rate`
and `eval/truncation_rate`.

### Output

```
logs/<Env>-<YYYYMMDD-HHMMSS>[-<suffix>]/
├── checkpoints/
│   ├── config.json        # env config used for training
│   ├── 0/                 # one orbax checkpoint per evaluation step
│   ├── 1032192000/
│   └── ...
├── ppo_config.json        # PPO config, including the network sizes and obs keys
├── run.json               # env name and seed
└── rollout.mp4            # deterministic rollout at the end of training
```

---

## Video of the latest checkpoint

```bash
cd expert
uv run python generate_last_video.py                            # most recent run in logs/
uv run python generate_last_video.py --run logs/Go2Getup-20260626-065620 --episodes 3
```

| Flag | Default | Description |
| --- | --- | --- |
| `--run` | most recent run in `logs/` | Run directory. |
| `--env` | inferred from the run name | Required if the run has a `--suffix`. |
| `--episodes` | `1` | Episodes evaluated (the video shows the first one). |
| `--seed` | `0` | Rollout seed. |
| `--out` | `<run>/rollout.mp4` | Video path. |
| `--impl` | env config | MJX backend (`jax` for CPU, `warp` for GPU). |

Uses the deterministic policy and prints the mean return. In scenes with no lights (such
as the Go2 rough terrain), it boosts the lighting and brightens the texture so the video
isn't dark.

## Evaluating several runs

`get_results_expert.py` evaluates each run's latest checkpoint with the deterministic
policy, writes a video of the first episode to `rollouts/rollout.mp4` next to the run's
`checkpoints/`, and appends the mean and std of the returns to a CSV file.

```bash
cd expert
uv run python get_results_expert.py                                  # every run in logs/
uv run python get_results_expert.py logs/Go2Getup-20260626-065620 --episodes 5
```

| Flag | Default | Description |
| --- | --- | --- |
| `runs` | every run in `logs/` | Run directories. The env is read from the run name (`<Env>-<YYYYMMDD-HHMMSS>`). |
| `--episodes` | `20` | Episodes per run. |
| `--out` | `results_expert.csv` | CSV file the results are appended to. |

---

## Notes

- The orbax checkpoint holds the policy weights and the observation normalization, but
  not the network shape. To reload one (in these scripts or in
  `sim2real/checkpoint_expert.py`), the network must match training. The reload scripts
  use the env's default config with `value_obs_key = state`, which is what
  `train_experts.sh` trains; `ppo_config.json` in each run records the exact settings.
