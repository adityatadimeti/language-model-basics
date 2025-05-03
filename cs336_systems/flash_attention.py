import einops
from einops import rearrange, einsum
import math
import torch


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        query_tile_dim = 16
        key_tile_dim   = 16

        B, num_queries, d_model = Q.shape
        num_keys = K.shape[1]

        num_query_tiles = math.ceil(num_queries / query_tile_dim)
        num_key_tiles   = math.ceil(num_keys   / key_tile_dim)

        # put tile index first so we can index with [i]
        Q_tiles = rearrange(Q, 'b (t tq) d -> t b tq d', tq=query_tile_dim)    
        K_tiles = rearrange(K, 'b (t tk) d -> t b tk d', tk=key_tile_dim)      
        V_tiles = rearrange(V, 'b (t tk) d -> t b tk d', tk=key_tile_dim)      

        O = torch.empty_like(Q)                             
        L = torch.empty(B, num_queries, device=Q.device)    

        for i in range(num_query_tiles):
            Q_i = Q_tiles[i]                 # (B, tq, d)
            O_i = torch.zeros_like(Q_i)
            l_i = torch.zeros(B, query_tile_dim, device=Q.device)              
            max_i = torch.full_like(l_i, -float('inf'))                        

            for j in range(num_key_tiles):
                K_j = K_tiles[j]            # (B, tk, d)
                V_j = V_tiles[j]            # (B, tk, d)
                S_i = einsum(Q_i, K_j, 'b tq d, b tk d -> b tq tk') / math.sqrt(d_model)

                row_maxes = S_i.max(dim=-1).values           # (B, tq)        
                new_max_i = torch.maximum(max_i, row_maxes)

                P_tilde   = torch.exp(S_i - new_max_i.unsqueeze(-1))           
                l_i       = torch.exp(max_i - new_max_i) * l_i + P_tilde.sum(dim=-1)
                O_i       = torch.exp(max_i - new_max_i).unsqueeze(-1) * O_i + P_tilde @ V_j
                max_i     = new_max_i

            O_i  = O_i / l_i.unsqueeze(-1)
            L_i  = max_i + torch.log(l_i)

            start = i * query_tile_dim
            end   = min(num_queries, start + query_tile_dim)
            O[:, start:end] = O_i[:, :end - start]
            L[:, start:end] = L_i[:, :end - start]

        # keep tensors for the backward pass
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, grad_O, grad_L):
        raise NotImplementedError   
