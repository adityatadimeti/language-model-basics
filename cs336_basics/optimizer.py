from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

def learning_rate_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, 
                           cosine_cycle_iters: int) -> float:
    if it < warmup_iters:
        # warmup
        return max_learning_rate * it / warmup_iters
    elif it <= cosine_cycle_iters:
        # cosine annealing
        return min_learning_rate + (0.5) * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    else:
        # post annealing
        return min_learning_rate

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float ) -> None:
    eps = 1e-6
    for param in parameters:
        if param.grad is None:
            continue
        norm = torch.linalg.norm(param.grad)
        if norm >= max_l2_norm:
            param.grad.mul_(max_l2_norm / (norm + eps))

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=None):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.

        return loss
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=( 0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        beta1, beta2 = betas
        if lr < 0 or beta1 < 0 or beta2 < 0 or eps < 0 or weight_decay < 0:
            raise ValueError(f"Invalid param")
        defaults = {"lr": lr, "beta1": beta1, "beta2": beta2,"eps": eps, "weight_decay":weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m, v = state["m"], state["v"]
                t = state["t"] + 1

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-alpha_t)

                p.data.add_(p.data, alpha=-lr * weight_decay)

                # Save state
                state["t"] = t
                state["m"] = m
                state["v"] = v

        return loss
# weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# opt = SGD([weights], lr=1e2)

# for t in range(10):
#     opt.zero_grad() # Reset the gradients for all learnable parameters.
#     loss = (weights**2).mean() # Compute a scalar loss value.
#     print(loss.cpu().item())
#     loss.backward() # Run backward pass, which computes gradients.
#     opt.step() # Run optimizer step.