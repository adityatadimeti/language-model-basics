#!/bin/bash
#SBATCH --job-name=train_lm_owt
#SBATCH --partition=a1-batch
#SBATCH --qos=a1-batch-qos
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --output=train_lm_%j.out
#SBATCH --error=train_lm_%j.err

# Run the training script
uv run cs336_basics/train_lm.py train --config tinystories.yaml
