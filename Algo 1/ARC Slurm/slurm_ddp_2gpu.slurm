#!/bin/bash
#SBATCH --job-name=ddp_2gpu
#SBATCH --partition=gpu2v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/ddp_2gpu_%j.out
#SBATCH --error=logs/ddp_2gpu_%j.err

module purge
module load anaconda3/2024.10-1
eval "$(conda shell.bash hook)"
conda activate hpml_env

mkdir -p logs results_ddp

echo "=========================================="
echo "  DDP Baseline — 2 GPUs (V100S-32GB)"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $(hostname)"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"

torchrun --standalone --nproc_per_node=2 \
    ddp_training.py \
    --gpus 2 \
    --epochs 30 \
    --batch_size 128 \
    --lr 0.1 \
    --output_dir ./results_ddp

echo "Done."
