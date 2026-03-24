#!/bin/bash
#SBATCH --job-name=ddp_4gpu
#SBATCH --partition=gpu4v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=logs/ddp_4gpu_%j.out
#SBATCH --error=logs/ddp_4gpu_%j.err

module purge
module load anaconda3/2024.10-1
eval "$(conda shell.bash hook)"
conda activate hpml_env

mkdir -p logs results_ddp

echo "=========================================="
echo "  DDP Baseline — 4 GPUs (V100S-32GB)"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $(hostname)"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"

torchrun --standalone --nproc_per_node=4 \
    ddp_training.py \
    --gpus 4 \
    --epochs 30 \
    --batch_size 128 \
    --lr 0.1 \
    --output_dir ./results_ddp

echo "Done."
