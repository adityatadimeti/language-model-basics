from contextlib import nullcontext
import os
import time
import torch
from tqdm import tqdm
import yaml
from cs336_basics.optimizer import (
    learning_rate_schedule,
    gradient_clipping,
    SGD,
    AdamW,
)
from cs336_basics.model_utils import load_data, save_checkpoint, load_checkpoint, get_cluster_data_path
import sys
from cs336_basics.transformer_modules import TransformerLM, cross_entropy, perplexity, decode, annotated_scaled_dot_product_attention
import cs336_basics
cs336_basics.transformer_modules.scaled_dot_product_attention = annotated_scaled_dot_product_attention
import timeit
from timeit import Timer
import numpy as np
import pickle
import torch.cuda.nvtx as nvtx

def profile(cfg):
    # Hyperparameters   
    vocab_size        = int(cfg['vocab_size'])
    context_length    = int(cfg.get('context_length', 128))
    d_model           = int(cfg.get('d_model', 512))
    d_ff              = int(cfg.get('d_ff', 2048))
    num_heads         = int(cfg.get('num_heads', 8))
    num_layers        = int(cfg.get('num_layers', 6))
    batch_size        = int(cfg.get('batch_size', 32))
    max_iters         = int(cfg.get('max_iters', 10000))
    max_lr            = float(cfg.get('max_lr', 1e-3))
    min_lr            = float(cfg.get('min_lr', 1e-5))
    warmup_iters_frac = float(cfg.get('warmup_iters_frac', 0.05))
    cosine_cycle_frac = float(cfg.get('cosine_cycle_frac', 0.8))
    assert 0.0 <= warmup_iters_frac <= 1.0, "warmup_frac must be in [0,1]"
    assert 0.0 <= cosine_cycle_frac <= 1.0, "cosine_frac must be in [0,1]"

    warmup_cosine_iters = int(warmup_iters_frac * max_iters)
    cosine_cycle_iters = min(int(cosine_cycle_frac * max_iters), max_iters)

    precision_type    = str(cfg.get('precision_type', "torch.float32")) 
    assert precision_type == "torch.bfloat16" or precision_type == "torch.float32", "precition_type must be either torch.bfloat16 or torch.bfloat32"
    weight_decay      = float(cfg.get('weight_decay', 1e-2))
    beta1             = float(cfg.get('beta1', 0.9))
    beta2             = float(cfg.get('beta2', 0.999))
    eps               = float(cfg.get('eps', 1e-8))
    max_grad_norm     = float(cfg.get('max_grad_norm', 1.0))
    optimizer_type    = str(cfg.get('optimizer', 'adamw')).lower()
    log_interval      = int(cfg.get('log_interval', 100))
    val_interval      = int(cfg.get('val_interval', 500))
    save_interval     = int(cfg.get('save_interval', 1000))
    theta             = int(cfg.get('theta', 0))
    gradient_accum    = int(cfg.get('gradient_accumulation_steps', 1))
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # Profiling parameters
    profile_warmup    = int(cfg.get('profile_warmup', 5))
    profile_measurement_steps    = int(cfg.get('profile_measurement_steps', 10))
    profile_component = str(cfg.get('profile_component', 'forward'))
    profile_type      = str(cfg.get('profile_type', 'time'))
    pass_type         = str(cfg.get('pass_type', 'forward/backward'))
    assert profile_warmup + profile_measurement_steps < max_iters, "profile warmup + profile_measurement must be less than max_iters"\



    print(f"Training on device: {device}")

    # Model
    model = TransformerLM(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        weights={},
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        theta=theta if theta else None,
        token_positions=None,
        max_seq_len=context_length
    ).to(device)

    model = torch.compile(model)

    # Optimizer
    if optimizer_type.lower() == 'sgd':
        optimizer = SGD(model.parameters(), lr=max_lr)
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=max_lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )

    # Resume checkpoint
    start_iter = 0
    it = start_iter

    optimizer.zero_grad()
    t = Timer()

    precision_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if precision_type=="torch.bfloat16" else nullcontext()
    training_ctx = nvtx.range("Training step") if "time" in profile_type and "backward" in profile_component else nullcontext()
    forward_ctx = nvtx.range("Model forward pass") if "time" in profile_type and "forward" in profile_component else nullcontext()
    backward_ctx = nvtx.range("Model backward pass") if "time" in profile_type and "backward" in profile_component else nullcontext()
    optimizer_ctx = nvtx.range("Optimizer step") if "time" in profile_type and "optimizer" in profile_component else nullcontext()

    # Training loop
    with tqdm(total=profile_warmup + profile_measurement_steps) as pbar, precision_ctx:
        while it < profile_warmup:
            xb = torch.randint(low = 0, high = vocab_size, size=(batch_size, context_length)).to(device)
            yb =  torch.randint(low = 0, high = vocab_size, size=(batch_size, context_length)).to(device)
            
            logits = model(xb)
            B, T, V = logits.shape

            # LR schedule
            lr = learning_rate_schedule(it, max_lr, min_lr, warmup_cosine_iters, cosine_cycle_iters)
            for group in optimizer.param_groups:
                group['lr'] = lr

            # Loss and update
            loss = cross_entropy(
                logits.view(-1, V),          
                yb.reshape(-1)               
            )
            loss = loss / gradient_accum
            loss.backward()


            if max_grad_norm > 0:
                gradient_clipping(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            
            it += 1
            pbar.update(1)

        

        while it < profile_warmup + profile_measurement_steps:
            
            if "memory" in profile_type:
                torch.cuda.memory._record_memory_history(max_entries=1000000) 

            # generate random batch of data
            with training_ctx:
                xb = torch.randint(low = 0, high = vocab_size, size=(batch_size, context_length)).to(device)
                yb =  torch.randint(low = 0, high = vocab_size, size=(batch_size, context_length)).to(device)

                with forward_ctx:
                    logits = model(xb)
                if "time" in profile_type:
                    torch.cuda.synchronize() 
                
                if pass_type == "forward":
                    torch.cuda.memory._dump_snapshot(f"memory_snapshot_forward_{context_length}_{d_model}_{d_ff}_{num_layers}_{num_heads}_{it}.pickle")
                
                else:  #passtype is forward/backward
                    B, T, V = logits.shape

                    # LR schedule
                    lr = learning_rate_schedule(it, max_lr, min_lr, warmup_cosine_iters, cosine_cycle_iters)
                    for group in optimizer.param_groups:
                        group['lr'] = lr

                    # Loss and update
                    loss = cross_entropy(
                        logits.view(-1, V),          
                        yb.reshape(-1)               
                    )
                    loss = loss / gradient_accum

                    with backward_ctx:
                        loss.backward()
                    if "time" in profile_type:
                        torch.cuda.synchronize() 


                    if max_grad_norm > 0:
                        gradient_clipping(model.parameters(), max_grad_norm)

                    with optimizer_ctx:
                        optimizer.step()
                    if "time" in profile_type:
                        torch.cuda.synchronize() 
                    optimizer.zero_grad()

                    torch.cuda.memory._dump_snapshot(f"memory_snapshot_forward_backward_{context_length}_{d_model}_{d_ff}_{num_layers}_{num_heads}_{it}.pickle")
                    #torch.cuda.memory._record_memory_history(enabled=None)
                
                it += 1
                pbar.update(1)
        torch.cuda.memory._record_memory_history(enabled=None)
    print(f"Training complete at step {it}")


if __name__ == "__main__":
    # usage: python script.py train --config config.yaml
    if len(sys.argv) >= 2 and sys.argv[1]=='--config':
        mode = sys.argv[1]
        cfg_path = sys.argv[2]
        with open(f"systems_configs/{cfg_path}") as f:
            cfg = yaml.safe_load(f)
        # argv:  uv run benchmarking_script.py --config benchmarking.yaml key1=val1 key2=val2 ...
        for override in sys.argv[3:]:
            if "=" not in override:
                raise ValueError(f"Bad override '{override}', expected key=value")
            k, v = override.split("=", 1)
            if k in cfg:
                cfg[k] = type(cfg[k])(v)       # cast to original type
            else:
                cfg[k] = yaml.safe_load(v)     # new key  best-effort cast


        # Automatically patch paths for GPU cluster if needed
        for k in ['train_data', 'val_data', 'checkpoint_path', 'vocab_file', 'merges_file']:
            if k in cfg:
                cfg[k] = get_cluster_data_path(cfg[k])

        profile(cfg)
    else:
        print(f"Usage: {sys.argv[0]} --config CONFIG_YAML_PATH")