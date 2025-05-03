#!/bin/bash
#SBATCH --job-name=benchmarking
#SBATCH --partition=a1-batch
#SBATCH --qos=a1-batch-qos
#SBATCH --output=benchmarking_%j.out
#SBATCH --error=benchmarking_%j.err
#SBATCH --gpus=1
set -euo pipefail

declare -a ctxs=(128 256 512)
declare -a pass_types=("forward" "backward")

# Fixed model config
d=2560
declare -A d_ff=(   [2560]=10240 )
declare -A layers=( [2560]=32    )
declare -A heads=(  [2560]=32    )

f=${d_ff[$d]}
nl=${layers[$d]}
nh=${heads[$d]}

# 2) Loop and run nsys with param-derived output name
for ctx in "${ctxs[@]}"; do
  for pass in "${pass_types[@]}"; do
    out="nsys_ctx${ctx}_d${d}_ff${f}_l${nl}_h${nh}_${pass}.qdrep"

    # skip if already profiled
    if [[ -f "${out}" ]]; then
      echo "Skipping ${out}, already exists."
      continue
    fi

    uv run nsys profile -o "${out}" \
      python benchmarking_script.py \
        --config benchmarking.yaml \
        context_length=${ctx} \
        d_model=${d} \
        d_ff=${f} \
        num_layers=${nl} \
        num_heads=${nh} \
        pass_type=${pass}
  done
done


# #!/bin/bash
# #SBATCH --job-name=benchmarking
# #SBATCH --partition=a1-batch           
# #SBATCH --qos=a1-batch-qos
# #SBATCH --output=benchmarking_%j.out
# #SBATCH --error=benchmarking_%j.err
# #SBATCH --gpus=1
# set -euo pipefail

# mkdir -p amp


# # 1) Define “sweeps”:
# #declare -a ctxs=(128 256 512 1024)
# declare -a ctxs=(128)
# # key = d_model, values = d_ff, num_layers, num_heads
# declare -A d_ff=(   [768]=3072   [1024]=4096   [1280]=5120   [1600]=6400   [2560]=10240 )
# declare -A layers=( [768]=12     [1024]=24     [1280]=36     [1600]=48     [2560]=32    )
# declare -A heads=(  [768]=12     [1024]=16     [1280]=20     [1600]=25     [2560]=32    )
# #declare -a d_models=(768 1024 1280 1600 2560)
# declare -a d_models=(2560)

# # 2) Loop and run nsys with a param-derived output name:
# for d in "${d_models[@]}"; do
#   for ctx in "${ctxs[@]}"; do
#     f=${d_ff[$d]}
#     nl=${layers[$d]}
#     nh=${heads[$d]}
#     out="amp/nsys_ctx${ctx}_d${d}_ff${f}_l${nl}_h${nh}.qdrep"

#     # skip if we've already profiled this setting
#     if [[ -f "${out}" ]]; then
#       echo "Skipping ${out}, already exists."
#       continue
#     fi

#     uv run nsys profile -o "${out}" \
#       python benchmarking_script.py \
#         --config benchmarking.yaml \
#         context_length=${ctx} \
#         d_model=${d} \
#         d_ff=${f} \
#         num_layers=${nl} \
#         num_heads=${nh}
#   done
# done

