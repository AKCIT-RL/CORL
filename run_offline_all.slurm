#!/usr/bin/env bash
#SBATCH --job-name=offline-train
#SBATCH --nodelist=Sebastian
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=72:00:00
#SBATCH --output=logs/offline-%j.out
#SBATCH --error=logs/offline-%j.err
#
# Train an offline algorithm on every dataset; wandb -> akcit-offlinerl/Offline-Benchmark
#
# Submit from the CORL repo root (first argument is the algorithm):
#   sbatch run_offline_all.slurm bc                          # BC, all datasets
#   sbatch run_offline_all.slurm iql go2-getup               # subset of tasks
#   SEED=1 sbatch run_offline_all.slurm td3_bc               # override the training seed
#
# <algo> is one of: bc awac td3_bc cql iql dt
set -euo pipefail

# Repository root. Override by exporting REPO before submitting,
# e.g. REPO=/path/to/CORL sbatch run_offline_all.slurm bc
REPO="${REPO:-/home/luana/Documents/OfflineRL/Benchmark/CORL}"
cd "$REPO"
mkdir -p logs

echo "Job $SLURM_JOB_ID on $(hostname) starting $(date)"
nvidia-smi -L || true

srun ./run_offline_all.sh "$@"

echo "Job $SLURM_JOB_ID finished $(date)"
