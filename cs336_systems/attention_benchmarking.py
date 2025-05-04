#!/usr/bin/env python3
import itertools
import json
import time

import torch
import pandas as pd
from cs336_basics.transformer_modules import CausalMultiHeadAttention

BATCH      = 8
DIMS       = [16, 32, 64, 128]
SEQS       = [256, 1024, 4096, 8192, 16384]
N_ITERS    = 100
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_SIZE = 4   # fp32

def est_attention_bytes(b, s, d, bytes_per_elem=DTYPE_SIZE):
    qkv   = 3 * b * s * d
    attn  = b * s * s
    out   = b * s * d
    return (qkv + attn + out) * bytes_per_elem    # bytes

def run_cfg(d_model: int, seq_len: int):
    torch.cuda.empty_cache()
    result = dict(d_model=d_model, seq_len=seq_len)
    try:
        model = CausalMultiHeadAttention(d_model, num_heads=1).to(DEVICE)
        x     = torch.randn(BATCH, seq_len, d_model,
                            device=DEVICE, requires_grad=True)

        for _ in range(10):
            _ = model(x); torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            _ = model(x); torch.cuda.synchronize()
        fwd_ms = (time.perf_counter() - t0) * 1e3 / N_ITERS

        torch.cuda.reset_peak_memory_stats()
        _ = model(x); torch.cuda.synchronize()
        mem_gib = torch.cuda.max_memory_allocated() / 1024**3

        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            out  = model(x)
            loss = out.sum()
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
        bwd_ms = (time.perf_counter() - t0) * 1e3 / N_ITERS

        result.update(
            forward_ms   = round(fwd_ms,   4),
            backward_ms  = round(bwd_ms,   4),
            mem_GiB      = round(mem_gib,  3),
            attn_est_GiB = round(est_attention_bytes(BATCH, seq_len, d_model)
                                 / 1024**3, 3),
            status       = "OK"
        )
    except RuntimeError as e:
        if "out of memory" in str(e):
            result["status"] = "OOM"
        else:
            result["status"] = f"ERROR: {e}"
        torch.cuda.empty_cache()
    return result

if __name__ == "__main__":
    combos = list(itertools.product(DIMS, SEQS))
    results = []
    total = len(combos)

    for idx, (d, s) in enumerate(combos, start=1):
        res = run_cfg(d, s)
        results.append(res)
        # periodic console logging
        print(f"[{idx}/{total}] d_model={d:4d}, seq_len={s:5d} → "
              f"status={res['status']:<4}  "
              f"fwd={res.get('forward_ms','-'):>6}ms  "
              f"bwd={res.get('backward_ms','-'):>6}ms  "
              f"mem={res.get('mem_GiB','-'):>5}GiB")

    df = pd.DataFrame(results)
    latex = df.to_latex(index=False, float_format="%.3f")

    print("\n=== LaTeX Table ===\n")
    print(latex)

    # also write to file
    with open("attention_bench.tex", "w") as f:
        f.write(latex)
    print("Wrote LaTeX table to attention_bench.tex")
