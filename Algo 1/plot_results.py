#!/usr/bin/env python3
"""
Plot performance results for Method 1 (PyTorch DDP).
Reads JSON result files and produces publication-quality plots.

Usage:
  python plot_results.py --results_dir ./results_ddp
"""

import os
import json
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir):
    """Load all ddp_gpu*_results.json files, keyed by GPU count."""
    runs = {}
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("ddp_gpu") and fname.endswith("_results.json"):
            fpath = os.path.join(results_dir, fname)
            with open(fpath) as f:
                data = json.load(f)
            gpu_count = data["world_size"]
            runs[gpu_count] = data
    return runs


def plot_epoch_time(runs, out_dir):
    """Bar chart of average epoch time vs. GPU count."""
    gpus = sorted(runs.keys())
    times = [runs[g]["avg_epoch_time_s"] for g in gpus]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([str(g) for g in gpus], times, color="#2563EB", width=0.5)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{t:.1f}s", ha="center", fontsize=10)
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Average Epoch Time (s)")
    ax.set_title("DDP: Training Time per Epoch vs. GPU Count")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "epoch_time_vs_gpus.png"), dpi=150)
    plt.close(fig)


def plot_speedup(runs, out_dir):
    """Speedup and ideal scaling vs. GPU count."""
    gpus = sorted(runs.keys())
    if 1 not in runs:
        print("Warning: single-GPU baseline not found; skipping speedup plot.")
        return
    base_time = runs[1]["avg_epoch_time_s"]
    speedups = [base_time / runs[g]["avg_epoch_time_s"] for g in gpus]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(gpus, speedups, "o-", color="#2563EB", label="Measured Speedup", linewidth=2)
    ax.plot(gpus, gpus, "--", color="#9CA3AF", label="Ideal Linear Speedup")
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Speedup")
    ax.set_title("DDP: Speedup vs. GPU Count")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "speedup_vs_gpus.png"), dpi=150)
    plt.close(fig)


def plot_efficiency(runs, out_dir):
    """Parallel efficiency (speedup / num_gpus) vs. GPU count."""
    gpus = sorted(runs.keys())
    if 1 not in runs:
        return
    base_time = runs[1]["avg_epoch_time_s"]
    efficiencies = [(base_time / runs[g]["avg_epoch_time_s"]) / g * 100 for g in gpus]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(gpus, efficiencies, "s-", color="#059669", linewidth=2)
    ax.axhline(100, linestyle="--", color="#9CA3AF", label="Ideal (100%)")
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Parallel Efficiency (%)")
    ax.set_title("DDP: Parallel Efficiency vs. GPU Count")
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "efficiency_vs_gpus.png"), dpi=150)
    plt.close(fig)


def plot_time_breakdown(runs, out_dir):
    """Stacked bar chart: compute vs. communication vs. data loading."""
    gpus = sorted(runs.keys())
    compute = [runs[g]["avg_compute_time_s"] for g in gpus]
    comm = [runs[g]["avg_comm_overhead_s"] for g in gpus]
    data_t = [runs[g]["avg_epoch_time_s"] - runs[g]["avg_compute_time_s"] - runs[g]["avg_comm_overhead_s"] for g in gpus]
    data_t = [max(0, d) for d in data_t]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(gpus))
    width = 0.5
    ax.bar(x, compute, width, label="Compute", color="#2563EB")
    ax.bar(x, comm, width, bottom=compute, label="Communication", color="#DC2626")
    ax.bar(x, data_t, width, bottom=[c + co for c, co in zip(compute, comm)],
           label="Data Loading", color="#F59E0B")
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in gpus])
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Time (s)")
    ax.set_title("DDP: Epoch Time Breakdown")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "time_breakdown.png"), dpi=150)
    plt.close(fig)


def plot_accuracy_curves(runs, out_dir):
    """Training and validation accuracy over epochs for each GPU config."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, g in enumerate(sorted(runs.keys())):
        history = runs[g]["epoch_history"]
        epochs = [h["epoch"] for h in history]
        train_acc = [h["train_acc"] for h in history]
        val_acc = [h["val_acc"] for h in history]

        axes[0].plot(epochs, train_acc, color=colors[idx], label=f"{g} GPU(s)")
        axes[1].plot(epochs, val_acc, color=colors[idx], label=f"{g} GPU(s)")

    axes[0].set_title("Training Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("DDP: Convergence Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_curves.png"), dpi=150)
    plt.close(fig)


def plot_gpu_memory(runs, out_dir):
    """Peak GPU memory across GPU configurations."""
    gpus = sorted(runs.keys())
    mem = [runs[g]["peak_gpu_mem_MB"] for g in gpus]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([str(g) for g in gpus], mem, color="#7C3AED", width=0.5)
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_title("DDP: Peak GPU Memory Usage")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gpu_memory.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results_ddp")
    args = parser.parse_args()

    runs = load_results(args.results_dir)
    if not runs:
        print(f"No result files found in {args.results_dir}")
        return

    print(f"Found results for GPU counts: {sorted(runs.keys())}")
    plot_epoch_time(runs, args.results_dir)
    plot_speedup(runs, args.results_dir)
    plot_efficiency(runs, args.results_dir)
    plot_time_breakdown(runs, args.results_dir)
    plot_accuracy_curves(runs, args.results_dir)
    plot_gpu_memory(runs, args.results_dir)
    print(f"Plots saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
