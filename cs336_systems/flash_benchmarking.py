#!/usr/bin/env python3
# benchmark_flash_attention.py
# ------------------------------------------------------------
# Compare vanilla CausalMHA vs. Triton-FlashAttention-2 version
# ------------------------------------------------------------
import itertools
import math
import os
import sys
from contextlib import nullcontext

import torch
import pandas as pd
from einops import rearrange, einsum
from triton.testing import do_bench

# ---------------------------------------------------------------------
# 1)  Project imports
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath("."))          # root on PYTHONPATH
from cs336_basics.transformer_modules import (   # noqa: E402
    CausalMultiHeadAttention, Linear
)
from cs336_systems.flash_attention import FlashAttentionTriton  # noqa: E402

# ---------------------------------------------------------------------
# 2)  A drop-in MHA wrapper that swaps in FlashAttention-2
# ---------------------------------------------------------------------
class CausalMultiHeadAttentionFlash(CausalMultiHeadAttention):
    """
    Same weights & projections as the baseline class – only the
    attention kernel changes to FlashAttentionTriton.
    Currently supports num_heads == 1 (per assignment spec).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, S, d_model)
        B, S, D = x.shape
        assert self.num_heads == 1, "Wrapper kept simple – single-head only"

        # — QKV projections —
        Q = einsum(x, self.q_proj.weights, "... s d_in, d_out d_in -> ... s d_out")
        K = einsum(x, self.k_proj.weights, "... s d_in, d_out d_in -> ... s d_out")
        V = einsum(x, self.v_proj.weights, "... s d_in, d_out d_in -> ... s d_out")

        # remove the trivial head dimension and run FlashAttention-2
        out_single = FlashAttentionTriton.apply(Q, K, V, True)   # causal=True

        # final output projection
        mha_out = einsum(self.o_proj.weights,
                         out_single,
                         "d_model d_in, ... d_in -> ... d_model")
        return mha_out

# ---------------------------------------------------------------------
# 3)  Helpers for timing forward / backward with do_bench
# ---------------------------------------------------------------------
def bench_forward(model, x):
    model.eval()
    model(x); torch.cuda.synchronize()          # warm-up & compile
    return do_bench(lambda: model(x))

def bench_forward_backward(model, x):
    def _fwd_bwd():
        model.zero_grad(set_to_none=True)
        out = model(x)
        out.sum().backward()
    _fwd_bwd(); torch.cuda.synchronize()        # warm-up & compile
    return do_bench(_fwd_bwd)

# ---------------------------------------------------------------------
# 4)  Main sweep loop
# ---------------------------------------------------------------------
def run(device="cuda"):
    seq_lens   = [2 ** p for p in range(7, 17)]            # 128 … 65 536
    d_models   = [16, 32, 64, 128]
    dtypes     = [torch.bfloat16, torch.float32]
    batch_size = 1
    num_heads  = 1

    rows = []

    for dtype, d_model, seq_len in itertools.product(dtypes, d_models, seq_lens):
        dtype_str = str(dtype).split(".")[-1]   # e.g. "bfloat16" or "float32"

        # -----------------------------------------------------------------
        # allocate input once and reuse for both models
        # -----------------------------------------------------------------
        try:
            x = torch.randn(batch_size, seq_len, d_model,
                            device=device, dtype=dtype)
        except RuntimeError as e:     # bf16 might not be available on CPU
            print(f"SKIP {dtype=} {d_model=} {seq_len=}: {e}")
            continue

        for impl, cls in [("pytorch", CausalMultiHeadAttention),
                          ("triton",  CausalMultiHeadAttentionFlash)]:

            try:
                model = cls(d_model=d_model, num_heads=num_heads).to(device, dtype)
                fwd_ms  = bench_forward(model, x)
                both_ms = bench_forward_backward(model, x)
                bwd_ms  = both_ms - fwd_ms
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                fwd_ms = bwd_ms = both_ms = float("nan")

            rows.append(dict(
                impl       = impl,
                dtype      = str(dtype).split(".")[-1],
                d_model    = d_model,
                seq_len    = seq_len,
                fwd_ms     = fwd_ms,
                bwd_ms     = bwd_ms,
                total_ms   = both_ms,
            ))

            print(f"[{impl:7}] dtype={dtype_str:<8} "
              f"d={d_model:<3} S={seq_len:<6}  "
              f"fwd={fwd_ms:7.2f}  bwd={bwd_ms:7.2f}  total={both_ms:7.2f}")


    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# 5)  Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    torch.cuda.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(False)   # ensure baseline is *not* Flash

    df = run("cuda")
    pd.set_option("display.max_rows", None)
    print("\n=== FlashAttention-2 vs. vanilla CausalMHA (batch=1, causal) ===")
    print(df)

    # Uncomment to persist for your report:
    # df.to_csv("flash_benchmark.csv", index=False)
    # with open("flash_benchmark.tex", "w") as f:
    #     f.write(df.to_latex(index=False, float_format='%.2f'))
