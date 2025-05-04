import torch
import torch.distributed as dist
from typing import Type

class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.param_to_rank = {}
        self.sharded_param_groups = []
        self.local_optimizer = None
        self.optimizer_cls = optimizer_cls
        self._global_idx = 0
        super().__init__([{"params": [torch.nn.Parameter(torch.zeros(1))]}], defaults=kwargs)
        self.param_groups = []
        if params is not None:
            self._add_param_groups(params)
    
    def _add_param_groups(self, params):
        if isinstance(params, torch.Tensor): self.add_param_group({"params": [params]})
        elif isinstance(params, dict): self.add_param_group(params)
        else:
            for g in params:
                if isinstance(g, torch.Tensor): self.add_param_group({"params": [g]})
                elif isinstance(g, dict): self.add_param_group(g)
                else: self.add_param_group({"params": list(g)})
    
    def _initialize_local_optimizer(self):
        if self.local_optimizer is None and self.sharded_param_groups:
            self.local_optimizer = self.optimizer_cls(self.sharded_param_groups, **self.defaults)
    
    def add_param_group(self, param_group):
        if not isinstance(param_group, dict): raise TypeError("param_group must be a dict")
        params = list(param_group.get("params", []))
        if not params: return
        
        group_no_params = {k: v for k, v in param_group.items() if k != "params"}
        self.param_groups.append(dict(group_no_params, params=params))
        
        local_params = []
        for p in params:
            owner = self._global_idx % self.world_size
            self.param_to_rank[p] = owner
            self._global_idx += 1
            if owner == self.rank: local_params.append(p)
        
        if local_params:
            local_group = dict(group_no_params, params=local_params)
            self.sharded_param_groups.append(local_group)
            if self.local_optimizer is not None:
                self.local_optimizer.add_param_group(local_group)
        
        self._initialize_local_optimizer()
    
    def step(self, closure=None):
        if self.world_size > 1: self._all_reduce_gradients()
        
        loss = None
        if self.local_optimizer is not None:
            loss = self.local_optimizer.step(closure) if closure else self.local_optimizer.step()
        
        if self.world_size > 1: self._broadcast_updated_parameters()
        return loss
    
    def _all_reduce_gradients(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(self.world_size)
    
    def _broadcast_updated_parameters(self):
        for src in range(self.world_size):
            for group in self.param_groups:
                for p in group["params"]:
                    if self.param_to_rank[p] == src:
                        dist.broadcast(p.data, src=src)
            dist.barrier()
    
    def zero_grad(self, set_to_none=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none: p.grad = None
                    else: p.grad.zero_()
        if self.local_optimizer is not None:
            self.local_optimizer.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self):
        return self.local_optimizer.state_dict() if self.local_optimizer else {"state": {}, "param_groups": []}
    
    def load_state_dict(self, state_dict):
        if self.local_optimizer is not None:
            self.local_optimizer.load_state_dict(state_dict)

def get_sharded_optimizer(params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    return ShardedOptimizer(params, optimizer_cls, **kwargs)