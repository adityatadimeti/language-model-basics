#!/bin/bash
#SBATCH --job-name=nsys-benchmark
#SBATCH --partition=a1-batch
#SBATCH --qos=a1-batch-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=benchmarking_%j.out
#SBATCH --error=benchmarking_%j.err

set -euo pipefail

# ensure we run from the submission directory
cd "$SLURM_SUBMIT_DIR"

# (uncomment & adjust if you need modules or a conda env)
# module load cuda/11.7 nsight-systems/2023.2
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate language-model-basics

# the sweep parameters
declare -a ctxs=(128 256 512)
declare -a pass_types=("forward" "backward")

# fixed model config
d=2560
declare -A d_ff=(   [2560]=10240 )
declare -A layers=( [2560]=32    )
declare -A heads=(  [2560]=32    )

f=${d_ff[$d]}
nl=${layers[$d]}
nh=${heads[$d]}

# loop & profile
for ctx in "${ctxs[@]}"; do
  for pass in "${pass_types[@]}"; do
    out="trace_ctx${ctx}_d${d}_ff${f}_l${nl}_h${nh}_${pass}.qdrep"

    if [[ -f "$out" ]]; then
      echo "Skipping $out (already exists)"
      continue
    fi

    echo "Profiling ctx=$ctx, pass=$pass → $out"
    nsys profile \
      --trace=cuda,nvtx \
      --output="$out" \
      uv run python attention_benchmarking.py \
        --context_length="$ctx" \
        --d_model="$d" \
        d_ff="$f" \
        num_layers="$nl" \
        num_heads="$nh" \
        pass_type="$pass"
  done
done

echo "✅ All runs complete."
