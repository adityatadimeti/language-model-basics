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
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    x_tensor = torch.from_numpy(x).long().to(device)
    n_tokens = x_tensor.size(0)

    max_start = n_tokens - context_length - 1
    if max_start < 0:
        raise ValueError("Context length is too large for the provided data array.")
    starts = torch.randint(0, max_start + 1, (batch_size,), device=device)

    offsets = torch.arange(context_length, device=device).unsqueeze(0)  # shape (1, context_length)
    idx = starts.unsqueeze(1) + offsets  # shape (batch_size, context_length)

    inputs = x_tensor[idx]                              
    targets = x_tensor[idx + 1]                        

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
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["iteration"]
