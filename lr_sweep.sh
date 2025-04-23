#!/bin/bash
#SBATCH --job-name=train_lm_tiny_sweep
#SBATCH --partition=a1-batch
#SBATCH --qos=a1-batch-qos
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --output=train_lm_tinystories_%A_%a.out
#SBATCH --error=train_lm_tinystories_%A_%a.err
#SBATCH --array=0-3            # four different LRs

# 1) define the list of max_lr values you want to sweep
LR_LIST=(0.001 0.0005 0.0002 0.0001)

# 2) pick the one for this array index
MAX_LR=${LR_LIST[$SLURM_ARRAY_TASK_ID]}

# 3) optionally compute min_lr as 1% of max_lr
MIN_LR=$(python3 - <<EOF
print(${MAX_LR} * 1e-2)
EOF
)

# 4) generate a one‐off tmp config in model_configs
TMP_CFG=$(mktemp model_configs/tinystories_XXXX.yaml)

# 5) patch in the new LR values (assumes lines like "max_lr: ..." and "min_lr: ..." at top level)
sed -e "s/^max_lr:.*$/max_lr: ${MAX_LR}/" \
    -e "s/^min_lr:.*$/min_lr: ${MIN_LR}/" \
    model_configs/tinystories.yaml > $TMP_CFG

# 6) launch your training, passing only the basename so train_lm.py sees model_configs/${CFG_FILE}
CFG_FILE=$(basename $TMP_CFG)
uv run cs336_basics/train_lm.py train --config $CFG_FILE

# 7) cleanup
rm $TMP_CFG
