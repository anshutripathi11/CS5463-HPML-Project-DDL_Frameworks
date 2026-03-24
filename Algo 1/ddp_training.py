#!/usr/bin/env python3
"""
Method 1: PyTorch Distributed Data Parallel (DDP) — Baseline
CS5463 High Performance Machine Learning — Spring 2026
Comparative Analysis of Distributed Deep Learning Frameworks

This script trains ResNet-18 on CIFAR-10 using PyTorch DDP with NCCL backend.
It collects per-epoch timing, communication overhead estimates, GPU memory usage,
training/validation accuracy, and loss curves for scalability analysis.

Usage:
  Single GPU:
    python ddp_training.py --gpus 1

  Multi-GPU (e.g., 4 GPUs on one node):
    torchrun --nproc_per_node=4 ddp_training.py --gpus 4

  Multi-Node via SLURM (example):
    srun --ntasks-per-node=4 torchrun \
        --nnodes=$SLURM_NNODES --nproc_per_node=4 \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        ddp_training.py --gpus 4
"""

import os
import sys
import time
import json
import argparse
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    """Initialize the distributed process group via NCCL."""
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Return True on rank-0 (or when not distributed)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def log(msg):
    """Print only on rank-0."""
    if is_main_process():
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_dataloaders(batch_size, data_dir="./data", num_workers=4):
    """
    CIFAR-10 with standard data augmentation.
    Returns (train_loader, val_loader, train_sampler).
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    )

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_transform
    )

    distributed = dist.is_initialized()
    train_sampler = DistributedSampler(train_dataset) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_sampler


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes=10):
    """
    ResNet-18 adapted for CIFAR-10 (32x32 images).
    We replace the first 7x7 conv with 3x3, remove the initial max-pool,
    and adjust the final FC layer.
    """
    model = models.resnet18(weights=None, num_classes=num_classes)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  
    return model


# ---------------------------------------------------------------------------
# Training & evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch. Returns dict with timing breakdown and metrics.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Timing accumulators
    compute_time = 0.0
    data_time = 0.0

    torch.cuda.synchronize()
    epoch_start = time.perf_counter()
    data_start = time.perf_counter()

    for batch_idx, (inputs, targets) in enumerate(loader):
        # ---- data loading time ----
        torch.cuda.synchronize()
        data_end = time.perf_counter()
        data_time += data_end - data_start

        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        # ---- computation time (forward + backward + step) ----
        torch.cuda.synchronize()
        comp_start = time.perf_counter()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()       # DDP synchronizes gradients via AllReduce
        optimizer.step()

        torch.cuda.synchronize()
        comp_end = time.perf_counter()
        compute_time += comp_end - comp_start

        # ---- stats ----
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        data_start = time.perf_counter()

    torch.cuda.synchronize()
    epoch_end = time.perf_counter()
    epoch_time = epoch_end - epoch_start

    # Communication time estimate: total - compute - data
    comm_time = max(epoch_time - compute_time - data_time, 0.0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    return {
        "epoch": epoch,
        "train_loss": avg_loss,
        "train_acc": accuracy,
        "epoch_time_s": epoch_time,
        "compute_time_s": compute_time,
        "data_time_s": data_time,
        "comm_overhead_s": comm_time,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on validation set. Returns (val_loss, val_accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# GPU memory helper
# ---------------------------------------------------------------------------

def gpu_memory_stats(device):
    """Return dict of GPU memory usage in MB."""
    return {
        "allocated_MB": torch.cuda.memory_allocated(device) / 1e6,
        "reserved_MB": torch.cuda.memory_reserved(device) / 1e6,
        "max_allocated_MB": torch.cuda.max_memory_allocated(device) / 1e6,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Method 1: PyTorch DDP Baseline")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs (for logging)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--data_dir", type=str, default="./data", help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="./results_ddp", help="Output dir for logs/plots")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args()

    # ---- Distributed setup ----
    distributed = "LOCAL_RANK" in os.environ
    local_rank = 0
    if distributed:
        local_rank = setup_distributed()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    world_size = get_world_size()

    log(f"{'='*60}")
    log(f"  Method 1: PyTorch DDP (FP32 Baseline)")
    log(f"  World size : {world_size} GPU(s)")
    log(f"  Per-GPU BS : {args.batch_size}")
    log(f"  Global BS  : {args.batch_size * world_size}")
    log(f"  Epochs     : {args.epochs}")
    log(f"  LR         : {args.lr}")
    log(f"{'='*60}")

    # ---- Data ----
    train_loader, val_loader, train_sampler = get_dataloaders(
        args.batch_size, args.data_dir, args.num_workers
    )

    # ---- Model ----
    model = build_model(num_classes=10).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ---- Optimizer & scheduler ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Training loop ----
    history = []
    total_train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        mem = gpu_memory_stats(device)
        train_stats["val_loss"] = val_loss
        train_stats["val_acc"] = val_acc
        train_stats["gpu_mem_allocated_MB"] = mem["allocated_MB"]
        train_stats["gpu_mem_max_allocated_MB"] = mem["max_allocated_MB"]
        train_stats["world_size"] = world_size
        train_stats["global_batch_size"] = args.batch_size * world_size
        train_stats["lr"] = scheduler.get_last_lr()[0]
        history.append(train_stats)

        log(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Time {train_stats['epoch_time_s']:.2f}s | "
            f"Compute {train_stats['compute_time_s']:.2f}s | "
            f"Comm {train_stats['comm_overhead_s']:.2f}s | "
            f"Train Acc {train_stats['train_acc']:.2f}% | "
            f"Val Acc {val_acc:.2f}% | "
            f"Mem {mem['max_allocated_MB']:.0f}MB"
        )

    total_train_time = time.perf_counter() - total_train_start
    log(f"\nTotal training time: {total_train_time:.2f}s")

    # ---- Save results (rank 0 only) ----
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

        # Save JSON log
        results = {
            "method": "PyTorch_DDP_FP32",
            "world_size": world_size,
            "per_gpu_batch_size": args.batch_size,
            "global_batch_size": args.batch_size * world_size,
            "epochs": args.epochs,
            "total_training_time_s": total_train_time,
            "final_train_acc": history[-1]["train_acc"],
            "final_val_acc": history[-1]["val_acc"],
            "best_val_acc": max(h["val_acc"] for h in history),
            "avg_epoch_time_s": sum(h["epoch_time_s"] for h in history) / len(history),
            "avg_compute_time_s": sum(h["compute_time_s"] for h in history) / len(history),
            "avg_comm_overhead_s": sum(h["comm_overhead_s"] for h in history) / len(history),
            "peak_gpu_mem_MB": max(h["gpu_mem_max_allocated_MB"] for h in history),
            "epoch_history": history,
        }

        out_file = os.path.join(args.output_dir, f"ddp_gpu{world_size}_results.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        log(f"Results saved to {out_file}")

        # Save model checkpoint
        ckpt_path = os.path.join(args.output_dir, f"ddp_gpu{world_size}_model.pt")
        state = model.module.state_dict() if distributed else model.state_dict()
        torch.save(state, ckpt_path)
        log(f"Model saved to {ckpt_path}")

    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
