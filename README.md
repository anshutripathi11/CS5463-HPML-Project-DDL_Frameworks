# HPML_Project
## Comparative Analysis of Distributed Deep Learning Frameworks

### Problem Statement:
The rapid advancement of deep learning has made multi-GPU and distributed systems essential for efficient model training and large-scale experimentation. Several distributed deep learning frameworks have been proposed. The scalability of the frameworks and the optimization of communication depend on the techniques used. Additionally, performance varies significantly across different frameworks. Therefore, evaluating the performance trade-offs among these frameworks is necessary to facilitate efficient high-performance computing (HPC) applications.

### Objective:
Effectively training deep neural networks across multiple GPUs is increasingly critical for modern high-performance computing. This project systematically compares three distributed deep learning training approaches on a multi-GPU platform:
  **1.	PyTorch Distributed Data Parallel (DDP):** Baseline, full model replication with AllReduce gradient sync via NCCL
  
  **2.	PyTorch FSDP (Fully Sharded Data Parallel):** Shards model parameters, gradients, and optimizer states across GPUs instead of replicating them, fundamentally different communication pattern (gather before forward, reduce-scatter after backward)
  
  **3.	DeepSpeed ZeRO:** Microsoft's memory-efficient distributed training with partitioned optimizer states and gradients, different communication strategy from both DDP and FSDP

The experiments will be conducted on the UTSA ARC cluster using GPUs. A standard image classification task will be used to ensure controlled and reproducible evaluation.

The comparison will focus on:

- Training time per epoch
- Scalability as the number of GPUs increases
- Speedup and efficiency
- Communication overhead
- GPU memory utilization
- Final model accuracy
- Convergence behavior
  
The underlying communication mechanisms will be analyzed to understand synchronization costs and communication patterns. The project will also evaluate the impact of mixed precision training on throughput and memory efficiency.

This study clarifies practical trade-offs and identifies which distributed deep learning framework best balances performance, scalability, and accuracy on HPC systems.
