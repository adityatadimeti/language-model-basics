#!/usr/bin/env python3
import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Manager
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.fc1(x))


def setup_process_group(rank, world_size, backend="nccl"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    if backend == "nccl":
        torch.cuda.set_device(rank)
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")

def cleanup_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()

WORLD_SIZE = 2
BATCH_SIZE = 4
FEATURES = 1024

def naive_ddp(model, data, world_size: int):
    logits = model(data)
    loss = logits.pow(2).mean()
    loss.backward()

    for p in model.parameters():
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)     
        p.grad.div_(world_size)
         
    return loss.detach()        

WORLD_SIZE, BATCH, FEAT = 2, 4, 1024

def _test(rank: int, world_size: int):
    device = setup_process_group(rank, world_size)
    dist.barrier()

    dtype = torch.float64

    # -------- full-batch baseline --------
    torch.manual_seed(0)
    x_full = torch.randn(BATCH, FEAT, device=device, dtype=dtype)

    baseline = ToyModel().to(device).double()
    baseline.zero_grad()
    baseline(x_full).pow(2).mean().backward()

    # -------- per-rank shard --------
    ddp_model = ToyModel().to(device).double()
    ddp_model.load_state_dict(baseline.state_dict())
    ddp_model.zero_grad()

    shard = x_full[rank * (BATCH // world_size) : (rank + 1) * (BATCH // world_size)]
    naive_ddp(ddp_model, shard, world_size)

    # gradients should now be identical
    for (b_name, b_param), (_, d_param) in zip(baseline.named_parameters(),
                                               ddp_model.named_parameters()):

        assert torch.allclose(d_param.grad, b_param.grad, rtol=1e-4, atol=1e-5), \
               f"{b_name} mismatch. Max diff: {torch.max(torch.abs(d_param.grad - b_param.grad))}"

    cleanup_process_group()

def test_ddp_naive():
    mp.spawn(_test, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)



if __name__ == "__main__":
    #profile()
    test_ddp_naive()

















# def benchmark_allreduce(rank, world_size, backend, tensor_numel, n_iters, return_dict):
#     # 1) init PG
#     setup_process_group(backend, rank, world_size)

#     # 2) pin each rank to a unique GPU when using NCCL
#     if backend == 'nccl':
#         torch.cuda.set_device(rank)
#         device = torch.device(f'cuda:{rank}')
#     else:
#         device = torch.device('cpu')

#     # 3) prepare tensor
#     x = torch.randn(tensor_numel, dtype=torch.float32, device=device)

#     # 4) warm-up (5 iters)
#     for _ in range(5):
#         dist.all_reduce(x, op=dist.ReduceOp.SUM)
#     if backend == 'nccl':
#         torch.cuda.synchronize()

#     # 5) timed runs
#     times = []
#     for _ in range(n_iters):
#         if backend == 'nccl':
#             torch.cuda.synchronize()
#         t0 = time.perf_counter()
#         dist.all_reduce(x, op=dist.ReduceOp.SUM)
#         if backend == 'nccl':
#             torch.cuda.synchronize()
#         t1 = time.perf_counter()
#         times.append(t1 - t0)

#     # 6) average locally, then max across ranks
#     local_avg = torch.tensor([sum(times) / len(times)], device=device)
#     dist.all_reduce(local_avg, op=dist.ReduceOp.MAX)
#     avg_max = local_avg.item()

#     if rank == 0:
#         return_dict[(backend, tensor_numel, world_size)] = avg_max

#     dist.destroy_process_group()

# def run_experiment(backend, world_size, tensor_numel, n_iters, manager_dict):
#     mp.spawn(
#         benchmark_allreduce,
#         args=(world_size, backend, tensor_numel, n_iters, manager_dict),
#         nprocs=world_size,
#         join=True
#     )

# def profile():
#     backends = ['gloo', 'nccl']
#     world_sizes = [2, 4, 6]
#     bytes_per_elem = 4
#     sizes_bytes = [1_000_000, 10_000_000, 100_000_000, 1_000_000_000]
#     tensor_numels = [b // bytes_per_elem for b in sizes_bytes]
#     n_iters = 20

#     mgr = Manager()
#     results = mgr.dict()

#     for backend in backends:
#         for world_size in world_sizes:
#             for nelem in tensor_numels:
#                 size_mb = nelem * bytes_per_elem / 1e6
#                 print(f'→ {backend.upper()}, procs={world_size}, size≈{size_mb:.0f} MB …', end='', flush=True)
#                 run_experiment(backend, world_size, nelem, n_iters, results)
#                 print(' done')

#     # assemble DataFrame
#     rows = []
#     for (backend, nelem, ws), t in results.items():
#         rows.append({
#             'backend': backend,
#             'world_size': ws,
#             'size_MB': nelem * bytes_per_elem / 1e6,
#             'avg_allreduce_s': t
#         })
#     df = pd.DataFrame(rows).sort_values(['backend','world_size','size_MB'])
#     print('\n=== Summary ===')
#     print(df.to_string(index=False))

#     # write CSV
#     out_csv = 'allreduce_bench_results.csv'
#     df.to_csv(out_csv, index=False)
#     print(f'Wrote raw timings to {out_csv}')

#     # plot & save
#     plt.figure(figsize=(8,5))
#     for backend in backends:
#         for ws in world_sizes:
#             sub = df[(df.backend==backend)&(df.world_size==ws)]
#             if not sub.empty:
#                 plt.plot(sub.size_MB, sub.avg_allreduce_s, marker='o',
#                          label=f'{backend}, procs={ws}')
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlabel('Tensor size (MB)')
#     plt.ylabel('All-reduce latency (s)')
#     plt.title('All-reduce benchmark over Gloo/CPU vs NCCL/GPU')
#     plt.legend()
#     plt.grid(True, which='both', ls=':')
#     plt.tight_layout()

#     plot_file = 'allreduce_bench_plot.png'
#     plt.savefig(plot_file)
#     print(f'Wrote latency plot to {plot_file}')