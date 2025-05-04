#!/usr/bin/env python3
# benchmark_flash_attention.py
# ------------------------------------------------------------
# Compare vanilla CausalMHA vs. Triton-FlashAttention-2 version
# ------------------------------------------------------------
import itertools
import math
import os
import sys
import statistics
from contextlib import nullcontext

import torch
import pandas as pd
from einops import rearrange, einsum

# ---------------------------------------------------------------------
# 1)  Project imports
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath("."))          # repo root on PYTHONPATH
from cs336_basics.transformer_modules import (   # noqa: E402
    CausalMultiHeadAttention, Linear
)
from cs336_systems.flash_attention import FlashAttentionTriton  # noqa: E402


# ---------------------------------------------------------------------
# 2)  A wrapper that swaps vanilla attention for FlashAttention-2
# ---------------------------------------------------------------------
class CausalMultiHeadAttentionFlash(CausalMultiHeadAttention):
    """
    Same weights & projections as the baseline class – only the
    attention kernel changes to FlashAttentionTriton.
    (assignment spec keeps num_heads == 1)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, S, d_model)
        B, S, D = x.shape
        assert self.num_heads == 1, "wrapper kept simple – single-head only"

        # — Q K V projections —
        Q = einsum(x, self.q_proj.weights, "... s d_in, d_out d_in -> ... s d_out")
        K = einsum(x, self.k_proj.weights, "... s d_in, d_out d_in -> ... s d_out")
        V = einsum(x, self.v_proj.weights, "... s d_in, d_out d_in -> ... s d_out")

        # FlashAttention-2 (causal)
        out_single = FlashAttentionTriton.apply(Q, K, V, True)

        # output projection
        return einsum(self.o_proj.weights,
                      out_single,
                      "d_model d_in, ... d_in -> ... d_model")


# ---------------------------------------------------------------------
# 3)  Timing helpers (CUDA events, median of n runs)
# ---------------------------------------------------------------------
def time_forward_backward(model, x, n_iters: int = 100):
    """Return (fwd_ms, bwd_ms, total_ms) as medians over n_iters."""
    fwd_times, bwd_times, tot_times = [], [], []
    stream = torch.cuda.current_stream()

    start_fwd = torch.cuda.Event(enable_timing=True)
    end_fwd   = torch.cuda.Event(enable_timing=True)
    end_tot   = torch.cuda.Event(enable_timing=True)

    for _ in range(n_iters):
        model.zero_grad(set_to_none=True)

        # ----- forward -----
        start_fwd.record(stream)
        out = model(x)
        end_fwd.record(stream)

        # ----- backward -----
        loss = out.sum()
        loss.backward()
        end_tot.record(stream)

        torch.cuda.synchronize()
        fwd_times.append(start_fwd.elapsed_time(end_fwd))       # ms
        bwd_times.append(end_fwd.elapsed_time(end_tot))
        tot_times.append(start_fwd.elapsed_time(end_tot))

    return (statistics.median(fwd_times),
            statistics.median(bwd_times),
            statistics.median(tot_times))


# ---------------------------------------------------------------------
# 4)  Main sweep
# ---------------------------------------------------------------------
def run(device="cuda"):
    seq_lens   = [128, 1024, 8192, 65536]      # keep table reasonable
    d_models   = [16, 32, 64, 128]
    dtypes     = [torch.bfloat16, torch.float32]
    batch_size = 1
    num_heads  = 1
    n_iters    = 100

    rows = []
    for dtype, d_model, seq_len in itertools.product(dtypes, d_models, seq_lens):
        dtype_name = str(dtype).split(".")[-1]
        try:
            x = torch.randn(batch_size, seq_len, d_model,
                            device=device, dtype=dtype)
        except RuntimeError as e:
            print(f"SKIP {dtype_name=} {d_model=} {seq_len=}: {e}")
            continue

        for impl, cls in [("pytorch", CausalMultiHeadAttention),
                          ("triton",  CausalMultiHeadAttentionFlash)]:
            try:
                model = cls(d_model=d_model, num_heads=num_heads)\
                        .to(device, dtype=dtype)
                fwd_ms, bwd_ms, tot_ms = time_forward_backward(
                    model, x, n_iters=n_iters
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                fwd_ms = bwd_ms = tot_ms = float("nan")

            rows.append(dict(
                impl    = impl,
                dtype   = dtype_name,
                d_model = d_model,
                seq_len = seq_len,
                fwd_ms  = fwd_ms,
                bwd_ms  = bwd_ms,
                total_ms = tot_ms,
            ))

            print(f"[{impl:7}] dtype={dtype_name:<8} "
                  f"d={d_model:<3} S={seq_len:<6}  "
                  f"fwd={fwd_ms:7.2f}  bwd={bwd_ms:7.2f}  total={tot_ms:7.2f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 5)  Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    torch.cuda.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(False)   # keep baseline “vanilla”

    df = run("cuda")

    # ---------- pretty-print LaTeX table ----------
    latex = df.to_latex(index=False,
                        float_format="%.2f",
                        column_format="llrrrrr",
                        escape=False)
    print("\n% --------- ⇣ paste straight into Overleaf ⇣ ---------")
    print(latex)
