# CORL (Clean Offline Reinforcement Learning)

A fork of [corl-team/CORL](https://github.com/corl-team/CORL), adapted for an **offline RL benchmark on robotics locomotion tasks** built on top of [MuJoCo Playground](https://github.com/AKCIT-RL/mujoco_playground). It provides high-quality, single-file implementations of state-of-the-art offline reinforcement learning algorithms in both **PyTorch** and **JAX**.

**Datasets:** Pre-collected trajectories are available on Hugging Face at [akcit-rl/playground](https://huggingface.co/datasets/akcit-rl/playground). The benchmark runner downloads any missing dataset automatically.

## Key Features

- **Dual-Framework Support** — Algorithms implemented in both PyTorch and JAX
- **SOTA Algorithms** — AWAC, BC, CQL, DT, IQL, TD3+BC, plus DAgger and DART
- **Robotics-Focused** — Pre-configured for Go2, G1, and H1 locomotion environments
- **Config-Driven Training** — Shared per-algorithm `base.yaml` + a task registry (`_datasets.yaml`)
- **One-Command Sweeps** — `run_offline_all.sh` trains an algorithm across every task × difficulty
- **Experiment Tracking** — Integrated Weights & Biases logging
- **Production-Ready** — Docker support with CUDA

---

## Benchmark Datasets

Datasets live on disk under `datasets/playground/<task_id>/<difficulty>-v0` and mirror the Hugging Face repo [akcit-rl/playground](https://huggingface.co/datasets/akcit-rl/playground) one-to-one. Minari resolves them via the id `playground/<task_id>/<difficulty>-v0`.

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

**Install dependencies:**

```bash
git clone git@github.com:AKCIT-RL/CORL.git && cd CORL
uv sync
```

`uv sync` installs the pinned MuJoCo Playground fork ([AKCIT-RL/mujoco_playground](https://github.com/AKCIT-RL/mujoco_playground), branch `go2`) that provides the benchmark environments.

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

> **Note:** To enable WandB logging, pass your API key during build:
> ```bash
> docker build --build-arg WANDB_API_KEY=<your_key> -t corl .
> ```

**2. Run the container:**

```bash
docker run --gpus all -it --rm corl
```

**3. Mount local datasets (optional):**

```bash
docker run --gpus all -it --rm \
  -v /path/to/datasets:/datasets \
  corl
```

---

## Usage & Training

Training is config-driven. Each algorithm has a single shared `base.yaml` with its
hyperparameters; the per-task fields (env, dataset id, command type, group, and DT
target returns) come from the registry at [configs/offline/_datasets.yaml](configs/offline/_datasets.yaml).

### Full sweeps with `run_offline_all.sh`

The easiest way to train is the sweep runner, which loops an algorithm over every
task × difficulty, downloading any missing dataset from Hugging Face first.

```bash
# BC on all tasks × difficulties
./run_offline_all.sh bc

# IQL on a subset of tasks
./run_offline_all.sh iql go2-getup h1-gait-tracking

# Override the training seed
SEED=1 ./run_offline_all.sh td3_bc

# Retrain datasets that are already marked done
FORCE=1 ./run_offline_all.sh bc

# Run from a different repo location
REPO=/path/to/CORL ./run_offline_all.sh bc
```

Supported algorithms: `bc`, `awac`, `td3_bc`, `cql`, `iql`, `dt`.

Finished runs drop a marker under `.done_runs/` so a resubmission resumes instead of
duplicating W&B runs. Set `SKIP_PLAYGROUND_UPGRADE=1` to skip re-resolving the
`playground` fork at the start of a sweep. The repository root defaults to a fixed
path but can be overridden by exporting `REPO` (works for both `run_offline_all.sh`
and `run_offline_all.slurm`).

### Running a single algorithm manually

```bash
python -m algorithms.offline.<algorithm>_jax \
  --config_path configs/offline/<algorithm>/base.yaml \
  --env <EnvName> \
  --dataset_id playground/<task_id>/<difficulty>-v0 \
  [--command_type <command>] \
  [--target_returns "[high, low]"]   # DT only
```

### Common Configuration Parameters

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

| Algorithm | Paper | PyTorch | JAX |
|-----------|-------|:-------:|:---:|
| **AWAC** | [Accelerating Online RL via Offline Datasets](https://arxiv.org/abs/2006.09359) | ✅ | ✅ |
| **BC** | Behavior Cloning | ✅ | ✅ |
| **CQL** | [Conservative Q-Learning](https://arxiv.org/abs/2006.04779) | ✅ | ✅ |
| **DT** | [Decision Transformer](https://arxiv.org/abs/2106.01345) | ✅ | ✅ |
| **IQL** | [Implicit Q-Learning](https://arxiv.org/abs/2110.06169) | ✅ | ✅ |
| **TD3+BC** | [A Minimalist Approach to Offline RL](https://arxiv.org/abs/2106.06860) | ✅ | ✅ |
| **DAgger** | [Dataset Aggregation](https://arxiv.org/abs/1011.0686) | — | ✅ |
| **DART** | [Noise Injection for Imitation Learning](https://arxiv.org/abs/1703.09327) | — | ✅ |

The `run_offline_all.sh` benchmark runner uses the JAX implementations.

---

## Project Structure

```
CORL/
├── algorithms/
│   ├── offline/              # Algorithm implementations
│   │   ├── any_percent_bc.py # PyTorch Behavior Cloning
│   │   ├── awac.py           # PyTorch AWAC
│   │   ├── awac_jax.py       # JAX AWAC
│   │   ├── bc_jax.py         # JAX Behavior Cloning
│   │   ├── cql.py            # PyTorch CQL
│   │   ├── cql_jax.py        # JAX CQL
│   │   ├── dagger_jax.py     # JAX DAgger
│   │   ├── dart_jax.py       # JAX DART
│   │   ├── dt.py             # PyTorch Decision Transformer
│   │   ├── dt_jax.py         # JAX Decision Transformer
│   │   ├── iql.py            # PyTorch IQL
│   │   ├── iql_jax.py        # JAX IQL
│   │   ├── td3_bc.py         # PyTorch TD3+BC
│   │   └── td3_bc_jax.py     # JAX TD3+BC
│   └── utils/                # Shared utilities (dataset loading, wrappers, video)
├── configs/
│   └── offline/
│       ├── _datasets.yaml    # Task registry (env, command_type, DT targets, eval shift)
│       ├── awac/base.yaml    # Per-algorithm shared hyperparameters
│       ├── bc/base.yaml
│       ├── cql/base.yaml
│       ├── dt/base.yaml
│       ├── iql/base.yaml
│       ├── sac_n/base.yaml
│       └── td3_bc/base.yaml
├── datasets/
│   └── playground/           # Local mirror of akcit-rl/playground
├── expert/                   # Expert policy training (PPO)
├── sim2real/                 # Checkpoint conversion for deployment
├── notebooks/                # Analysis & visualization notebooks
├── run_offline_all.sh        # Sweep runner (task × difficulty)
├── Dockerfile                # CUDA container definition
└── pyproject.toml            # Project metadata & dependencies
```

---

## Requirements

- **Python** >= 3.11
- **CUDA** 12.x (for GPU acceleration)
- **Core Dependencies:**
  - `torch==2.8.0`
  - `jax[cuda12]==0.6.0`
  - `minari[all]==0.5.3`
  - `mujoco==3.6.0` / `mujoco-mjx==3.6.0` / `warp-lang==1.11.0`
  - `pyrallis==0.3.1`
  - `wandb==0.25.1`
  - `playground` ([AKCIT-RL/mujoco_playground](https://github.com/AKCIT-RL/mujoco_playground), branch `go2`)

---

## License

This project is distributed under the terms of the [LICENSE](LICENSE) file.
