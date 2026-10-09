# CORL (Clean Offline Reinforcement Learning)

A fork of [corl-team/CORL](https://github.com/corl-team/CORL), adapted for an **offline RL benchmark on robotics locomotion tasks** built on top of [MuJoCo Playground](https://github.com/ANONYMOUS/mujoco_playground). It provides high-quality, single-file implementations of state-of-the-art offline reinforcement learning algorithms in **JAX**.

**Datasets:** Pre-collected trajectories are available on Hugging Face at [anonymous/playground](https://huggingface.co/datasets/anonymous/playground). The benchmark runner downloads any missing dataset automatically.

## Key Features

- **JAX Implementations** — Every algorithm is a single JAX/Flax file, tested on the benchmark
- **SOTA Algorithms** — AWAC, BC, CQL, DT, IQL and TD3+BC
- **Robotics-Focused** — Pre-configured for Go2, G1, and H1 locomotion environments
- **Config-Driven Training** — Shared per-algorithm `base.yaml` + a task registry (`_datasets.yaml`)
- **One-Command Sweeps** — `run_offline_all.sh` trains an algorithm across every task × difficulty
- **Sim2Real Robustness (SRR)** — Every run is scored under domain + sensor randomization at the end of training, on all benchmark envs
- **Experiment Tracking** — Integrated Weights & Biases logging
- **Production-Ready** — Docker support with CUDA

---

## Benchmark Datasets

Datasets live on disk under `datasets/playground/<task_id>/<difficulty>-v0` and mirror the Hugging Face repo [anonymous/playground](https://huggingface.co/datasets/anonymous/playground) one-to-one. Minari resolves them via the id `playground/<task_id>/<difficulty>-v0`.

### Tasks

| `task_id` | Env | Command | Description |
|-----------|-----|---------|-------------|
| `go2-getup` | `Go2Getup` | — | Recovery from a fallen state |
| `go2-getup-walk` | `Go2GetupWalk` | — | Get up and walk |
| `go2-footstand` | `Go2Footstand` | — | Standing on rear feet |
| `go2-handstand` | `Go2Handstand` | — | Handstand pose |
| `go2-push-recovery` | `Go2PushRecovery` | — | Recover from external pushes (Tier-5 shifted eval) |
| `go2-joystick-direction` | `Go2JoystickFlatTerrain` | — | Flat-terrain joystick locomotion |
| `go2-flat-forward` | `Go2JoystickFlatTerrain` | `forwardfixed` | Fixed forward-velocity locomotion |
| `go2-rough-terrain` | `Go2RoughCurriculum` | — | Locomotion over rough terrain |
| `g1-joystick-direction` | `G1JoystickFlatTerrain` | — | G1 humanoid flat-terrain locomotion |
| `h1-gait-tracking` | `H1JoystickGaitTracking` | `forwardfixed` | H1 humanoid gait tracking |

### Difficulty Levels

Every task is provided in four difficulties:

| Difficulty | Description |
|------------|-------------|
| **expert** | Demonstrations from a fully trained policy |
| **medium** | Suboptimal demonstrations from a partially trained policy |
| **medium-expert** | Mixture of medium and expert trajectories |
| **medium-replay** | Replay buffer collected while training up to medium level |

---

## Environment Setup

### Using `uv` (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Ensure you have `uv` installed:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Install dependencies:** download the repository from <https://anonymous.4open.science/r/CORL-97F5/>, then run from its root:

```bash
uv sync
```

`uv sync` installs the pinned MuJoCo Playground fork ([ANONYMOUS/mujoco_playground](https://github.com/ANONYMOUS/mujoco_playground), branch `go2`) that provides the benchmark environments. The interpreter comes from `.python-version` (Python 3.13).

The SRR bootstrap confidence intervals need `scipy`, which lives in an optional group:

```bash
uv sync --group stats
```

**Run a script:**

```bash
uv run python -m algorithms.offline.awac_jax \
  --config_path configs/offline/awac/base.yaml \
  --env Go2JoystickFlatTerrain \
  --dataset_id playground/go2-flat-forward/expert-v0 \
  --command_type forwardfixed
```

### Using Docker

**1. Build the Docker image:**

```bash
docker build -t corl .
```

The image holds the dependencies only; the repository (code, `datasets/`, outputs) is
mounted at `/CORL` when the container starts.

**2. Run the container** from the repository root:

```bash
docker run --gpus all -it --rm \
  -v "$PWD":/CORL \
  -e WANDB_API_KEY \
  corl
```

Inside the container, run the same commands as below (for example
`./run_offline_all.sh bc`). Drop `-e WANDB_API_KEY` and set `-e WANDB_MODE=offline` to
train without a W&B account.

---

## Usage & Training

Training is config-driven. Each algorithm has a single shared `base.yaml` with its
hyperparameters; the per-task fields (env, dataset id, command type, group, and DT
target returns) come from the registry at [configs/offline/_datasets.yaml](configs/offline/_datasets.yaml).

### Full sweeps with `run_offline_all.sh`

The easiest way to train and evaluate is the sweep runner, which loops an algorithm
over every task × difficulty, downloading any missing dataset from Hugging Face first.

```bash
# W&B: log in (or export WANDB_API_KEY=...); runs go to the "Offline-Benchmark"
# project of your default entity, or of WANDB_ENTITY when set.
# Without a W&B account: export WANDB_MODE=offline
export WANDB_API_KEY=...

# BC on all tasks × difficulties
./run_offline_all.sh bc

# IQL on a subset of tasks
./run_offline_all.sh iql go2-getup h1-gait-tracking

# Override the training seed
SEED=1 ./run_offline_all.sh td3_bc

# Retrain datasets that are already marked done
FORCE=1 ./run_offline_all.sh bc

# Only evaluate existing checkpoints (no training, no W&B key needed)
EVAL_ONLY=1 ./run_offline_all.sh cql

# Run from a different repo location
REPO=/path/to/CORL ./run_offline_all.sh bc
```

Supported algorithms: `bc`, `awac`, `td3_bc`, `cql`, `iql`, `dt`.

For each dataset, the runner:

1. trains the run and saves the log to `logs/matrix/<algo>-<task>-<difficulty>-seed<N>.log`;
2. checks that the end-of-training SRR evaluation wrote `logs/compare/metrics/<run>.json`;
3. if that JSON is missing (the proxy failed, or the process died after training),
   runs `scripts.recover_proxy` on the log. It finishes the evaluation and logs it to
   the same W&B run. It won't run if training itself didn't finish;
4. marks the run done under `.done_runs/<algo>-<task>-<difficulty>-seed<N>.done`, so a
   resubmission resumes instead of duplicating W&B runs.

With `EVAL_ONLY=1`, the runner skips training. It scores every finished checkpoint
(`checkpoint_final.npz`) of the selected datasets and `SEED` that has no metrics JSON
yet, using `scripts.compare_randomize`. Those results are written to the JSON only, not
to W&B.

A failed dataset doesn't stop the sweep; failures are listed at the end and the
script exits with status 1. Training needs a W&B login unless `WANDB_MODE` is
`offline` or `disabled`. `DEVICE` (default `cuda`) sets the evaluation device.

At the start of a sweep the runner updates the `playground` fork to the latest commit
of its branch (`uv sync --upgrade-package playground`), which also updates `uv.lock`.
To reproduce results with exactly the commit pinned in `uv.lock`, set
`SKIP_PLAYGROUND_UPGRADE=1`.

`run_offline_all.sh` uses its own directory as the repository root; override it by
exporting `REPO`.

### Running a single algorithm manually

```bash
python -m algorithms.offline.<algorithm>_jax \
  --config_path configs/offline/<algorithm>/base.yaml \
  --env <EnvName> \
  --dataset_id playground/<task_id>/<difficulty>-v0 \
  [--command_type <command>] \
  [--target_returns "[high, low]"]   # DT only
```

For DT, `--seed` is the training seed. The base config's `train_seed` is only a
legacy alias, used when `seed` is unset.

### What a run logs at the end

| W&B key | Description |
|---------|-------------|
| `eval/final_score`, `eval/final_raw_score` | D4RL-normalized and raw return over `n_eval_episodes_final` episodes |
| `eval/shifted_final_score`, `eval/robustness_gap` | Tier-5 shifted evaluation (tasks with `eval_shift` in the registry: push recovery, rough terrain) |
| `eval/proxy_results/<suite>/*` | SRR evaluation of `checkpoint_final.npz`: `default` baseline + `humanoid_gym_relative`, 100 episodes / 50 actors |

The SRR proxy is the same evaluation as `scripts/run_srr_eval.sh`, and it also writes
`logs/compare/metrics/<run>.json`, so the W&B numbers and the JSON match. A proxy
failure doesn't fail the run. See [scripts/README.md](scripts/README.md) for the SRR
suites and metrics.

### Common Configuration Parameters

Field names vary slightly between algorithms (e.g. AWAC's `num_train_ops` /
`eval_frequency` vs BC's `max_timesteps` / `eval_freq`); each `base.yaml` is the reference.

| Parameter | Description | Example |
|-----------|-------------|---------|
| `env` / `env_name` | Environment name (env_name for DT) | `Go2JoystickFlatTerrain` |
| `dataset_id` | Minari dataset identifier | `playground/go2-flat-forward/expert-v0` |
| `command_type` | Joystick command override (optional) | `forwardfixed` |
| `seed` | Random seed for reproducibility | `42` |
| `device` | Compute device | `cuda` or `cpu` |
| `batch_size` | Training batch size | `256` |
| `learning_rate` | Optimizer learning rate | `0.0003` |
| `num_train_ops` | Total training steps | `1000000` |
| `eval_frequency` | Evaluation interval (steps) | `5000` |
| `checkpoints_path` | Model checkpoint directory | `checkpoints/AWAC` |

### Example `base.yaml` (AWAC)

```yaml
# configs/offline/awac/base.yaml
project: Offline-Benchmark
checkpoints_path: checkpoints/AWAC
device: cuda
seed: 42
test_seed: 69

awac_lambda: 0.1
batch_size: 256
buffer_size: 10000000
gamma: 0.99
hidden_dim: 256
learning_rate: 0.0003
n_test_episodes: 10
num_train_ops: 1000000
eval_frequency: 5000
tau: 0.005
```

---

## Implemented Algorithms

| Algorithm | Paper | Module |
|-----------|-------|--------|
| **AWAC** | [Accelerating Online RL via Offline Datasets](https://arxiv.org/abs/2006.09359) | `awac_jax` |
| **BC** | Behavior Cloning | `bc_jax` |
| **CQL** | [Conservative Q-Learning](https://arxiv.org/abs/2006.04779) | `cql_jax` |
| **DT** | [Decision Transformer](https://arxiv.org/abs/2106.01345) | `dt_jax` |
| **IQL** | [Implicit Q-Learning](https://arxiv.org/abs/2110.06169) | `iql_jax` |
| **TD3+BC** | [A Minimalist Approach to Offline RL](https://arxiv.org/abs/2106.06860) | `td3_bc_jax` |
All modules live in `algorithms/offline/`, and `run_offline_all.sh` covers all six.

---

## Project Structure

```
CORL/
├── algorithms/
│   ├── offline/              # Algorithm implementations
│   │   ├── awac_jax.py       # AWAC
│   │   ├── bc_jax.py         # Behavior Cloning
│   │   ├── cql_jax.py        # CQL
│   │   ├── dt_jax.py         # Decision Transformer
│   │   ├── iql_jax.py        # IQL
│   │   └── td3_bc_jax.py     # TD3+BC
│   └── utils/
│       ├── dataset.py        # Minari dataset loading
│       ├── randomize_gym.py  # Env wrapper + domain/sensor randomization (OBS_LAYOUTS per env)
│       ├── proxy.py          # End-of-training SRR evaluation
│       ├── wrapper_gym.py    # Simpler env wrapper used by sim2real/evaluate_actor.py
│       └── space.py          # Gym space helpers
├── configs/
│   ├── offline/
│   │   ├── _datasets.yaml    # Task registry (env, command_type, DT targets, eval shift)
│   │   ├── awac/base.yaml    # Per-algorithm shared hyperparameters
│   │   ├── bc/base.yaml
│   │   ├── cql/base.yaml
│   │   ├── dt/base.yaml
│   │   ├── iql/base.yaml
│   │   └── td3_bc/base.yaml
│   └── randomize/            # SRR perturbation suites (humanoid_gym_relative is the default)
├── scripts/                  # SRR evaluation and run recovery (see scripts/README.md)
├── datasets/
│   └── playground/           # Local mirror of anonymous/playground
├── expert/                   # Expert policy training (PPO; see expert/README.md)
├── sim2real/                 # Export to portable NumPy actors for deployment (see sim2real/README.md)
├── run_offline_all.sh        # Sweep runner (task × difficulty)
├── Dockerfile                # CUDA container definition
└── pyproject.toml            # Project metadata & dependencies
```

---

## Requirements

- **Python** >= 3.11 (3.13 pinned in `.python-version`)
- **CUDA** 12.x (for GPU acceleration)
- **Core Dependencies:**
  - `torch==2.8.0`
  - `jax[cuda12]==0.6.0`
  - `minari[all]==0.5.3`
  - `mujoco==3.6.0` / `mujoco-mjx==3.6.0` / `warp-lang==1.11.0`
  - `numpy<2.5` (mediapy is incompatible with numpy 2.5)
  - `imageio-ffmpeg` (ffmpeg fallback for video on nodes without a system ffmpeg)
  - `pyrallis==0.3.1`
  - `wandb==0.25.1`
  - `playground` ([ANONYMOUS/mujoco_playground](https://github.com/ANONYMOUS/mujoco_playground), branch `go2`)
- **Optional:** `scipy` (`uv sync --group stats`) for SRR confidence intervals

---

## License

This project is distributed under the terms of the [LICENSE](LICENSE) file.
