#!/bin/bash
#SBATCH --job-name=ddp_a100
#SBATCH --partition=gpu1a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/ddp_a100_%j.out
#SBATCH --error=logs/ddp_a100_%j.err

module purge
module load anaconda3/2024.10-1
eval "$(conda shell.bash hook)"
conda activate hpml_env

mkdir -p logs results_ddp_a100

echo "=========================================="
echo "  DDP Baseline — 1 GPU (A100)"
echo "  Cross-platform comparison vs V100S"
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
    --output_dir ./results_ddp_a100

echo "Done."
