import torch
import torch.nn as nn
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors



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




class DDPBucketed(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.bucket_bytes = int(bucket_size_mb * 1024 * 1024)
        self.param_to_bucket = {}
        self.buckets = []
        
        if self.world_size > 1:
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0)
            
            params_with_grad = [p for p in self.module.parameters() if p.requires_grad]
            params_with_grad.sort(key=lambda p: p.numel() * p.element_size(), reverse=True)
            
            cur_bucket, cur_size = [], 0
            for p in params_with_grad:
                size = p.numel() * p.element_size()
                if cur_size + size > self.bucket_bytes and cur_bucket:
                    self.buckets.append({"params": cur_bucket, "ready": 0, "handle": None, "flat": None, "grads": None})
                    cur_bucket, cur_size = [], 0
                cur_bucket.append(p)
                cur_size += size
                self.param_to_bucket[p] = len(self.buckets)
            
            if cur_bucket:
                self.buckets.append({"params": cur_bucket, "ready": 0, "handle": None, "flat": None, "grads": None})
            
            for p in params_with_grad:
                p.register_post_accumulate_grad_hook(self._hook)

    def _hook(self, param: nn.Parameter):
        if param.grad is None or self.world_size == 1:
            return
        
        bucket_idx = self.param_to_bucket.get(param)
        if bucket_idx is None:
            return
            
        bucket = self.buckets[bucket_idx]
        bucket["ready"] += 1
        
        if bucket["ready"] == len(bucket["params"]):
            grads = [p.grad.data for p in bucket["params"]]
            flat = _flatten_dense_tensors(grads)
            handle = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
            bucket.update(handle=handle, flat=flat, grads=grads)
    
    def reset_buckets(self):
        for bucket in self.buckets:
            bucket["ready"] = 0
            bucket["handle"] = None
            bucket["flat"] = None
            bucket["grads"] = None

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        if self.world_size == 1:
            return
            
        for bucket in self.buckets:
            handle = bucket.get("handle")
            if handle is None:
                continue
            
            handle.wait()
            flat = bucket["flat"]
            grads = bucket["grads"]
            flat.div_(self.world_size)
            
            outputs = _unflatten_dense_tensors(flat, grads)
            for grad_out, grad_in in zip(outputs, grads):
                grad_in.copy_(grad_out)
            
            bucket["handle"] = None
            bucket["flat"] = None
            bucket["grads"] = None
            bucket["ready"] = 0

def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float = 25.0) -> torch.nn.Module:
    return DDPBucketed(module, bucket_size_mb)

def ddp_bucketed_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    ddp_model.finish_gradient_synchronization()

def ddp_bucketed_on_train_batch_start(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    optimizer.zero_grad()
    if hasattr(ddp_model, "reset_buckets"):
        ddp_model.reset_buckets()