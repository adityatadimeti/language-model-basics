#!/bin/bash
#SBATCH --job-name=benchmarking
#SBATCH --partition=a1-batch           
#SBATCH --qos=a1-batch-qos
#SBATCH --output=benchmarking_%j.out
#SBATCH --error=benchmarking_%j.err
#SBATCH --gpus=1

# Run commands
uv run benchmarking_script.py --config benchmarking.yaml context_length=128 d_model=768 d_ff=3072 num_layers=12 num_heads=12
uv run benchmarking_script.py --config benchmarking.yaml context_length=256 d_model=768 d_ff=3072 num_layers=12 num_heads=12
uv run benchmarking_script.py --config benchmarking.yaml context_length=512 d_model=768 d_ff=3072 num_layers=12 num_heads=12
uv run benchmarking_script.py --config benchmarking.yaml context_length=1024 d_model=768 d_ff=3072 num_layers=12 num_heads=12

uv run benchmarking_script.py --config benchmarking.yaml context_length=128 d_model=1024 d_ff=4096 num_layers=24 num_heads=16
uv run benchmarking_script.py --config benchmarking.yaml context_length=256 d_model=1024 d_ff=4096 num_layers=24 num_heads=16
uv run benchmarking_script.py --config benchmarking.yaml context_length=512 d_model=1024 d_ff=4096 num_layers=24 num_heads=16
uv run benchmarking_script.py --config benchmarking.yaml context_length=1024 d_model=1024 d_ff=4096 num_layers=24 num_heads=16

uv run benchmarking_script.py --config benchmarking.yaml context_length=128 d_model=1280 d_ff=5120 num_layers=36 num_heads=20
uv run benchmarking_script.py --config benchmarking.yaml context_length=256 d_model=1280 d_ff=5120 num_layers=36 num_heads=20
uv run benchmarking_script.py --config benchmarking.yaml context_length=512 d_model=1280 d_ff=5120 num_layers=36 num_heads=20
uv run benchmarking_script.py --config benchmarking.yaml context_length=1024 d_model=1280 d_ff=5120 num_layers=36 num_heads=20

uv run benchmarking_script.py --config benchmarking.yaml context_length=128 d_model=1600 d_ff=6400 num_layers=48 num_heads=25
uv run benchmarking_script.py --config benchmarking.yaml context_length=256 d_model=1600 d_ff=6400 num_layers=48 num_heads=25
uv run benchmarking_script.py --config benchmarking.yaml context_length=512 d_model=1600 d_ff=6400 num_layers=48 num_heads=25
uv run benchmarking_script.py --config benchmarking.yaml context_length=1024 d_model=1600 d_ff=6400 num_layers=48 num_heads=25

uv run benchmarking_script.py --config benchmarking.yaml context_length=128 d_model=2560 d_ff=10240 num_layers=32 num_heads=32
uv run benchmarking_script.py --config benchmarking.yaml context_length=256 d_model=2560 d_ff=10240 num_layers=32 num_heads=32
uv run benchmarking_script.py --config benchmarking.yaml context_length=512 d_model=2560 d_ff=10240 num_layers=32 num_heads=32
uv run benchmarking_script.py --config benchmarking.yaml context_length=1024 d_model=2560 d_ff=10240 num_layers=32 num_heads=32
