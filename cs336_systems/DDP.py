import torch
import torch.nn as nn
import torch.distributed as dist


class DDP_Class(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._handles = []
        if self.world_size > 1:
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0)
                if p.requires_grad:                          # guard wuz needed  "cannot register a hook on a tensor that doesn't require gradient"
                    p.register_post_accumulate_grad_hook(self._hook)

    def _hook(self, p: nn.Parameter):
        if p.grad is None:
            return
        h = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
        self._handles.append((h, p))

    def forward(self, *args, **kw):
        return self.module(*args, **kw)

    def finish_gradient_synchronization(self):
        if self.world_size == 1:
            return
        for h, p in self._handles:
            h.wait()
            p.grad.div_(self.world_size)
        self._handles.clear()
