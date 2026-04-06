# CORL (Clean Offline Reinforcement Learning)

A fork of [corl-team/CORL](https://github.com/corl-team/CORL), modified to focus on **robotics locomotion tasks**. This repository provides high-quality, single-file implementations of state-of-the-art offline reinforcement learning algorithms in both **PyTorch** and **JAX**.

**Datasets:** Pre-collected expert and medium trajectories are available on Hugging Face at [akcit-rl/playground](https://huggingface.co/datasets/akcit-rl/playground).

## Key Features

- **Dual-Framework Support** — All algorithms implemented in both PyTorch and JAX
- **SOTA Algorithms** — AWAC, BC, CQL, DT, IQL, TD3+BC, DAgger, DART, and more
- **Robotics-Focused** — Pre-configured for Go1, Go2, G1, and H1 humanoid environments
- **Config-Driven Training** — YAML-based configuration via [pyrallis](https://github.com/eladrich/pyrallis)
- **Experiment Tracking** — Integrated Weights & Biases logging
- **Production-Ready** — Docker support with CUDA 12.4

---

## Available Datasets

All datasets follow the naming convention: `{Robot}{Task}-{variant}-{quality}-v1`

| Robot | Type | Description |
|-------|------|-------------|
| **Go2** | Quadruped | Unitree Go2 robot |
| **G1** | Humanoid | Unitree G1 humanoid robot |
| **H1** | Humanoid | Unitree H1 humanoid robot |

### Tasks & Variants

| Task | Robot | Variants | Description |
|------|-------|----------|-------------|
| **JoystickFlatTerrain** | Go2, G1 | `direction`, `forward`, `forwardfixed`, `forwardbackward`* | Locomotion on flat terrain with joystick velocity commands |
| **JoystickGaitTracking** | H1 | `direction`, `forward`, `forwardfixed` | Humanoid gait tracking with joystick control |
| **InplaceGaitTracking** | H1 | — | Stationary gait tracking |
| **Footstand** | Go2 | — | Standing on front feet |
| **Handstand** | Go2 | — | Handstand pose |
| **Getup** | Go2 | — | Recovery from fallen state |

*`forwardbackward` variant available only for Go2

### Dataset Quality Levels

| Quality | Description |
|---------|-------------|
| **expert** | High-quality demonstrations from a fully trained policy |
| **medium** | Suboptimal demonstrations from a partially trained policy |

### Full Dataset List

<details>
<summary>Click to expand all 28 datasets</summary>

**G1 Humanoid:**
- `G1JoystickFlatTerrain-direction-expert-v1` / `medium-v1`
- `G1JoystickFlatTerrain-forward-expert-v1` / `medium-v1`
- `G1JoystickFlatTerrain-forwardfixed-expert-v1` / `medium-v1`

**Go2 Quadruped:**
- `Go2JoystickFlatTerrain-direction-expert-v1` / `medium-v1`
- `Go2JoystickFlatTerrain-forward-expert-v1` / `medium-v1`
- `Go2JoystickFlatTerrain-forwardfixed-expert-v1` / `medium-v1`
- `Go2JoystickFlatTerrain-forwardbackward-expert-v1` / `medium-v1`
- `Go2Footstand-expert-v1` / `medium-v1`
- `Go2Handstand-expert-v1` / `medium-v1`
- `Go2Getup-expert-v1` / `medium-v1`

**H1 Humanoid:**
- `H1JoystickGaitTracking-direction-expert-v1` / `medium-v1`
- `H1JoystickGaitTracking-forward-expert-v1` / `medium-v1`
- `H1JoystickGaitTracking-forwardfixed-expert-v1` / `medium-v1`
- `H1InplaceGaitTracking-expert-v1` / `medium-v1`

</details>

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
git clone https://github.com/corl-team/CORL.git && cd CORL
uv sync
```

**Run scripts:**

```bash
uv run python algorithms/offline/awac.py --config configs/offline/awac/go2/joystick_flat_forward_expert.yaml
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

Training is fully config-driven. Each algorithm reads hyperparameters from a YAML file.

### Basic Training Command

```bash
python algorithms/offline/<algorithm>.py --config <config_path>
```

### Examples

| Algorithm | Framework | Command |
|-----------|-----------|---------|
| AWAC | PyTorch | `python algorithms/offline/awac.py --config configs/offline/awac/go2/joystick_flat_forward_expert.yaml` |
| AWAC | JAX | `python algorithms/offline/awac_jax.py --config configs/offline/awac/go2/joystick_flat_forward_expert.yaml` |
| IQL | PyTorch | `python algorithms/offline/iql.py --config configs/offline/iql/go2/joystick_flat_forward_expert.yaml` |
| TD3+BC | JAX | `python algorithms/offline/td3_bc_jax.py --config configs/offline/td3_bc/go2/joystick_flat_forward_expert.yaml` |
| BC | JAX | `python algorithms/offline/bc_jax.py --config configs/offline/bc/go2/joystick_flat_forward_expert.yaml` |

### Common Configuration Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `env` | Environment name | `Go2JoystickFlatTerrain` |
| `dataset_id` | Minari dataset identifier | `playground/Go2JoystickFlatTerrain-forward-expert-v1` |
| `seed` | Random seed for reproducibility | `42` |
| `device` | Compute device | `cuda` or `cpu` |
| `batch_size` | Training batch size | `256` |
| `learning_rate` | Optimizer learning rate | `0.0003` |
| `num_train_ops` | Total training steps | `1000000` |
| `eval_frequency` | Evaluation interval (steps) | `5000` |
| `checkpoints_path` | Model checkpoint directory | `checkpoints/AWAC` |

### Example Configuration File

```yaml
# configs/offline/awac/go2/joystick_flat_forward_expert.yaml
env: Go2JoystickFlatTerrain
dataset_id: playground/Go2JoystickFlatTerrain-forward-expert-v1

# Training
batch_size: 256
seed: 42
device: cuda
learning_rate: 0.0003
num_train_ops: 1000000
eval_frequency: 5000
n_test_episodes: 10

# Algorithm-specific
hidden_dim: 256
tau: 0.005
awac_lambda: 0.1
gamma: 0.99

# Logging
project: Offline-Benchmark
group: awac-go2-joystick-flat-expert-v1
checkpoints_path: checkpoints/AWAC
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

---

## Project Structure

```
CORL/
├── algorithms/
│   ├── offline/              # Algorithm implementations
│   │   ├── awac.py           # PyTorch AWAC
│   │   ├── awac_jax.py       # JAX AWAC
│   │   ├── bc_jax.py         # JAX Behavior Cloning
│   │   ├── cql.py            # PyTorch CQL
│   │   ├── cql_jax.py        # JAX CQL
│   │   ├── dt.py             # PyTorch Decision Transformer
│   │   ├── dt_jax.py         # JAX Decision Transformer
│   │   ├── iql.py            # PyTorch IQL
│   │   ├── iql_jax.py        # JAX IQL
│   │   ├── td3_bc.py         # PyTorch TD3+BC
│   │   ├── td3_bc_jax.py     # JAX TD3+BC
│   │   └── ...
│   └── utils/                # Shared utilities
│       ├── dataset.py        # Dataset loading & preprocessing
│       ├── wrapper_gym.py    # Gymnasium wrappers
│       └── save_video.py     # Video recording utilities
├── configs/
│   └── offline/              # Training configurations
│       ├── awac/             # AWAC configs by robot
│       ├── bc/               # BC configs
│       ├── cql/              # CQL configs
│       ├── dt/               # Decision Transformer configs
│       ├── iql/              # IQL configs
│       └── td3_bc/           # TD3+BC configs
├── expert/                   # Expert policy training (PPO)
├── sim2real/                 # Checkpoint conversion for deployment
├── notebooks/                # Analysis & visualization notebooks
├── requirements/             # Dependency specifications
├── Dockerfile                # CUDA 12.4 container definition
└── pyproject.toml            # Project metadata & dependencies
```

---

## Requirements

- **Python** >= 3.10
- **CUDA** 12.x (for GPU acceleration)
- **Core Dependencies:**
  - `torch==2.8.0`
  - `jax[cuda12-local]==0.6.0`
  - `minari[all]==0.5.3`
  - `gymnasium`
  - `pyrallis==0.3.1`
  - `wandb==0.19.11`

---

## License



---

## Citation




