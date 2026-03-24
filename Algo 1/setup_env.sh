#!/bin/bash
# ==========================================================
#  ONE-TIME SETUP: Run on a GPU node (interactive session)
#
#  srun -p gpu1v100 -n 1 --gres=gpu:1 -t 00:30:00 --pty bash
#  bash setup_env.sh
# ==========================================================

set -e

echo "=== Setting up HPML project environment ==="
echo ""

module purge
module load anaconda3/2024.10-1

echo "Creating conda environment..."
conda create -n hpml_env python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate hpml_env

echo "Installing PyTorch (cu121 — compatible with your V100S driver 545)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "Installing utilities..."
pip install matplotlib numpy

echo ""
echo "=== Verifying installation ==="
python -c "
import torch
print(f'PyTorch version:  {torch.__version__}')
print(f'CUDA available:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version:     {torch.version.cuda}')
    print(f'cuDNN version:    {torch.backends.cudnn.version()}')
    print(f'GPU count:        {torch.cuda.device_count()}')
    print(f'GPU name:         {torch.cuda.get_device_name(0)}')
    print(f'GPU memory:       {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    print(f'NCCL available:   {torch.distributed.is_nccl_available()}')
else:
    print('WARNING: CUDA not available! Check driver compatibility.')
    exit(1)
print()
print('Setup complete!')
"

echo ""
echo "=== Done! ==="
echo "Now uncomment 'conda activate hpml_env' in each SLURM script."
echo ""
echo "Quick test (from interactive GPU session):"
echo "  torchrun --standalone --nproc_per_node=1 ddp_training.py --gpus 1 --epochs 2"
