#!/bin/bash
#SBATCH --job-name=vad-full
#SBATCH -A lsq@a100
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err

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
eval "$(conda shell.bash hook)"
conda activate /lustre/fsn1/projects/rech/lsq/umv83if/conda_envs/vad

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "python=$(command -v python)"

echo "===== TORCH / GPU ====="

python - <<'PY'
import sys
import torch

print("Python:", sys.version)
print("Executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the Slurm job.")

print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU memory:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
    "GB",
)
PY

echo "===== CONFIG ====="
grep -A5 "^augment:" configs/config_full.yaml

echo "===== FULL TRAINING ====="

python -u -m src.train \
    --config configs/config_full.yaml

echo "===== FINISHED ====="
