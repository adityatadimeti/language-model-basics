#!/bin/bash
#SBATCH --job-name=allreduce-bench
#SBATCH --partition=a2
#SBATCH --qos=a2-qos
#SBATCH --nodes=1
#SBATCH --gpus=6
#SBATCH --cpus-per-gpu=4     # (=24 CPUs total)
#SBATCH --output=bench_%j.out
#SBATCH --error=bench_%j.err

uv run python distributed_communication_single_node.py
