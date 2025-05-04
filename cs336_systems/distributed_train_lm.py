#!/usr/bin/env python3
import os
import time
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


from tqdm import tqdm

XL = dict(
    d_model    = 1600,
    d_ff       = 6400,
    num_heads  = 25,
    num_layers = 48,
    vocab_size = 50257,
    context_length = 128,
)
from cs336_basics.transformer_modules import TransformerLM

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    return torch.device(f"cuda:{rank}")

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def naive_allreduce(model):
    """All-reduce each grad; returns time spent in communication."""
    t0 = time.perf_counter()
    for p in model.parameters():
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(dist.get_world_size())
    torch.cuda.synchronize()
    return time.perf_counter() - t0

def minimal_ddp_flat_benchmarking(model):
    # collect all grads (in order)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0

    # flatten
    flat = _flatten_dense_tensors(grads)
    t0 = time.perf_counter()
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(dist.get_world_size())
    torch.cuda.synchronize()
    t_comm = time.perf_counter() - t0

    for buf, synced in zip(grads, _unflatten_dense_tensors(flat, grads)):
        buf.copy_(synced)

    return t_comm


def benchmark(rank, args):
    device = setup(rank, args.world_size)

    # model + optim
    torch.manual_seed(0)


    model = TransformerLM(
        d_model=XL['d_model'],
        num_heads=XL['num_heads'],
        d_ff=XL['d_ff'],
        weights = {},
        vocab_size=XL['vocab_size'],
        context_length=XL['context_length'],
        num_layers=XL['num_layers'],
        max_seq_len = XL['context_length']
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = args.batch_size
    dummy_x = torch.randint(0, XL['vocab_size'], (batch, XL['context_length']), device=device)
    dummy_y = torch.randint(0, XL['vocab_size'], (batch, XL['context_length']), device=device)

    times_compute = []
    times_comm    = []

    print(f"[Rank {rank}] Warmup...")

    for _ in tqdm(range(args.warmup), desc=f"Rank {rank} Warmup", position=rank):
        optimizer.zero_grad()
        logits = model(dummy_x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), dummy_y.view(-1)
        )
        loss.backward()
        #naive_allreduce(model)
        comm_time = minimal_ddp_flat_benchmarking(model)
        optimizer.step()
    dist.barrier()

    # timed runs
    print(f"[Rank {rank}] Benchmarking...")

    for _ in tqdm(range(args.steps), desc=f"Rank {rank} Benchmark", position=rank):
        t_start = time.perf_counter()

        optimizer.zero_grad()
        logits = model(dummy_x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), dummy_y.view(-1)
        )
        # measure compute (fwd+back)
        t_c0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t_c1 = time.perf_counter()

        # measure comm
        #t_comm = naive_allreduce(model)
        t_comm = minimal_ddp_flat_benchmarking(model)

        optimizer.step()
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        times_compute.append((t_c1 - t_c0))
        times_comm.append(t_comm)

    

    # reduce across ranks
    comp = torch.tensor(times_compute, device=device).mean()
    comm = torch.tensor(times_comm,    device=device).mean()
    dist.reduce(comp, 0, op=dist.ReduceOp.SUM)
    dist.reduce(comm, 0, op=dist.ReduceOp.SUM)
    cleanup()

    if rank == 0:
        comp = comp.item() / args.world_size
        comm = comm.item() / args.world_size
        total = comp + comm
        print(f"\n=== Naive DDP benchmark (XL model, 1 nodex2 GPUs) ===")
        print(f"Steps:   {args.steps}  Batch: {batch}")
        print(f"Compute: {comp*1000:.1f} ms/step")
        print(f"Comm:    {comm*1000:.1f} ms/step")
        print(f"Total:   {total*1000:.1f} ms/step")
        print(f"Comm fraction: {100*comm/total:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--warmup",    type=int, default=5)
    parser.add_argument("--steps",     type=int, default=50)
    args = parser.parse_args()

    mp.spawn(benchmark, args=(args,), nprocs=args.world_size, join=True)
