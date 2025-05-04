#!/bin/bash
#SBATCH --job-name=naive-ddp-bench
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err
#SBATCH --partition=a2
#SBATCH --qos=a2-qos
#SBATCH --gpus=2   
#SBATCH --time=00:05:00                
#SBATCH --cpus-per-task=8           
#SBATCH --nodes=1




uv run distributed_train_lm.py --world_size 2 --batch_size 8 --warmup 5 --steps 25
#uv run distributed_communication_single_node.py 
