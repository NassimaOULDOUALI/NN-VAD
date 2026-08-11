#!/bin/bash
#SBATCH --job-name=vad-smoke
#SBATCH -A lsq@a100
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

set -euo pipefail

cd /lustre/fsn1/projects/rech/lsq/umv83if/NN-VAD

echo "===== HOST ====="
hostname

echo "===== SLURM ====="
echo "JOB_ID=$SLURM_JOB_ID"
echo "CPUS=$SLURM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "===== ENVIRONMENT ====="

module load anaconda-py3

# Initialise Conda inside the non-interactive Slurm shell
eval "$(conda shell.bash hook)"

# Use the exact prefix that works interactively
conda activate /lustre/fsn1/projects/rech/lsq/umv83if/conda_envs/vad

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "PATH=$PATH"
echo "python=$(command -v python)"

echo "===== PYTHON ====="
python --version

python - <<'PY'
import sys
import torch

print("sys.executable:", sys.executable)
print("torch:", torch.__version__)
print("torch location:", torch.__file__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the A100 Slurm job.")

print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU memory GB:",
    torch.cuda.get_device_properties(0).total_memory / 1024**3
)
PY

echo "===== CONFIG ====="
grep -A5 "^augment:" configs/config_smoke.yaml

echo "===== TRAINING ====="

python -u -m src.train \
    --config configs/config_smoke.yaml \
    --epochs 1 \
    --limit 100 \
    --no-wandb

echo "===== FINISHED ====="
