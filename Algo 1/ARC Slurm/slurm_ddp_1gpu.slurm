#!/bin/bash
#SBATCH --job-name=ddp_1gpu
#SBATCH --partition=gpu1v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/ddp_1gpu_%j.out
#SBATCH --error=logs/ddp_1gpu_%j.err

# ---- Environment ----
# NOTE: Do NOT load cudatoolkit — PyTorch ships its own CUDA 12.1 runtime,
#       which is compatible with your driver 545.23.08 (supports up to CUDA 12.3).
#       Loading cudatoolkit/12.8+ would cause version conflicts.
module purge
module load anaconda3/2024.10-1
eval "$(conda shell.bash hook)"
conda activate hpml_env

mkdir -p logs results_ddp

echo "=========================================="
echo "  DDP Baseline — 1 GPU (V100S-32GB)"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $(hostname)"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"

torchrun --standalone --nproc_per_node=1 \
    ddp_training.py \
    --gpus 1 \
    --epochs 30 \
    --batch_size 128 \
    --lr 0.1 \
    --output_dir ./results_ddp

echo "Done."
