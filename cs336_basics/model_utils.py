import typing
import numpy as np
import torch
import os
import torch
from typing import Union, IO
from torch import nn, optim

def load_data(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_tokens = len(x)
    max_start = n_tokens - context_length - 1
    if max_start < 0:
        raise ValueError("context_length is too large for this dataset")

    starts = np.random.randint(0, max_start + 1, size=batch_size, dtype=np.int64)

    batch = np.stack([x[s : s + context_length + 1] for s in starts])

    batch = torch.from_numpy(batch.astype(np.int64))        
    batch = batch.to(device, non_blocking=True)             

    inputs  = batch[:, :-1]   
    targets = batch[:, 1:]   

    return inputs, targets




def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: optim.Optimizer
) -> int:

    checkpoint = torch.load(src, map_location=lambda storage, loc: storage)
    new_sd = {
        key.replace("_orig_mod.", ""): value
        for key, value in checkpoint["model_state"].items()
    }
    #model.load_state_dict(checkpoint["model_state"])
    model.load_state_dict(new_sd)

    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["iteration"]
