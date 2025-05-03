import einops
from einops import rearrange, einsum
import math
import torch
import triton
import triton.language as tl


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        query_tile_dim = 16
        key_tile_dim   = 16

        B, num_queries, d_model = Q.shape
        num_keys = K.shape[1]

        num_query_tiles = math.ceil(num_queries / query_tile_dim)
        num_key_tiles   = math.ceil(num_keys   / key_tile_dim)

        Q_tiles = rearrange(Q, 'b (t tq) d -> t b tq d', tq=query_tile_dim)    
        K_tiles = rearrange(K, 'b (t tk) d -> t b tk d', tk=key_tile_dim)      
        V_tiles = rearrange(V, 'b (t tk) d -> t b tk d', tk=key_tile_dim)      

        O = torch.empty_like(Q)                             
        L = torch.empty(B, num_queries, device=Q.device)    

        for i in range(num_query_tiles):
            Q_i = Q_tiles[i]                  
            O_i = torch.zeros_like(Q_i)
            l_i = torch.zeros(B, query_tile_dim, device=Q.device)              
            max_i = torch.full_like(l_i, -float('inf'))                        

            for j in range(num_key_tiles):
                K_j = K_tiles[j]             
                V_j = V_tiles[j]           
                S_i = einsum(Q_i, K_j, 'b tq d, b tk d -> b tq tk') / math.sqrt(d_model)

                row_maxes = S_i.max(dim=-1).values           
                new_max_i = torch.maximum(max_i, row_maxes)

                P_tilde   = torch.exp(S_i - new_max_i.unsqueeze(-1))           
                l_i       = torch.exp(max_i - new_max_i) * l_i + P_tilde.sum(dim=-1)
                O_i       = torch.exp(max_i - new_max_i).unsqueeze(-1) * O_i + P_tilde.to(V_j.dtype) @ V_j
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


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        B, N_Q, D = Q.shape
        _, N_K, _ = K.shape
        Q_TILE_SZ = 16
        K_TILE_SZ = 16 
        T_q = (N_Q + Q_TILE_SZ - 1) // Q_TILE_SZ

        O = torch.empty_like(Q)
        L = torch.empty(B, N_Q, device=Q.device, dtype=Q.dtype)

        stride_qb, stride_qq, stride_qd = Q.stride()
        stride_kb, stride_kk, stride_kd = K.stride()
        stride_vb, stride_vk, stride_vd = V.stride()
        stride_ob, stride_oq, stride_od = O.stride()
        stride_lb, stride_lq            = L.stride()

        scale = 1.0 / math.sqrt(D)

        flash_fwd_kernel[(T_q, B)](
            Q, K, V,                   
            O, L,                     
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            N_Q, N_K,
            scale, 
            is_causal,
            D, 
            Q_TILE_SZ, 
            K_TILE_SZ
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal        
        return O

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError  # left for later

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    IS_CAUSAL: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
# """
# 1 - launch grid should be set as (T_q, batch_size) <- each triton program instance only loads elements
# from single batch index, and only read/write to single query tile of Q,O,L
# 2 - kernel has single loop over the key tiles
# 3 - advance block pointers at the end of the loop
# 4 - the on chip buffers (O_i, l, m) should have dtype tl.float32. if accumulating into output
# buffer, use acc = tl.dot(..., acc=acc)
# 5- cast P_tilde to dtype of V
# """
    # Program indices (get tile and batch indices)
    query_tile_index = tl.program_id(0)
    batch_index      = tl.program_id(1)

    # Construct a block pointer for the Q matrix
    # offset each pointer with the corresponding batch index multiplied with the 
    # batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0,0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D), # same shape as Keys
        order=(1, 0)
    )

    O_block_ptr = tl.make_block_ptr(
    base = O_ptr + batch_index*stride_ob,
    shape=(N_QUERIES, D),
    strides=(stride_oq, stride_od),
    offsets=(query_tile_index*Q_TILE_SIZE, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(1,0)
    )
    L_block_ptr = tl.make_block_ptr(
        base = L_ptr + batch_index*stride_lb,
        shape=(N_QUERIES, 1),
        strides=(stride_lq, 1),
        offsets=(query_tile_index*Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1,0)
    )

    query = tl.load(Q_block_ptr)
    output = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32) # this should be query tile dimension
    l = tl.zeros((Q_TILE_SIZE, ), dtype = tl.float32)
    m = tl.full((Q_TILE_SIZE, ), -1e9, dtype = tl.float32)

    # Single loop over the key tiles. Computing QK for this program's query across 
    # ALL keys. 
    T_k = (N_KEYS + K_TILE_SIZE - 1) // K_TILE_SIZE
    for j in range(0, T_k):
         # Now move to the next key and value
        key = tl.load(K_block_ptr)
        value = tl.load(V_block_ptr)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        #scores = tl.einsum("qd, kd -> qk", query, key) * scale
        #scores = einsum("qd, kd -> qk", query, key) * scale
        scores = tl.dot(query, tl.trans(key)) * scale

        if IS_CAUSAL:
            q_ids = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_ids = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

            mask = q_ids[:, None] < k_ids[None, :]           
            scores = scores + mask * (-1e6)


        row_max = tl.max(scores, -1) # gets the maxes for each qk pair in the tile
        new_max = tl.maximum(m, row_max)

        P_tilde = tl.exp(scores - new_max[:, None]) # no unsqueeze operator
        P_tile = P_tilde.to(V_block_ptr.type.element_ty)

        l       = tl.exp(m - new_max) * l + tl.sum(P_tilde, axis=1)
        #output  = tl.exp(m - new_max)[:, None] * output + tl.dot(P_tilde, value)
        output = tl.dot(P_tile, value, acc=tl.exp(m - new_max)[:, None] * output)

        m   = new_max
    
    out_final = (output / l[:, None]).to(O_ptr.type.element_ty)

    tl.store(O_block_ptr, out_final)
    L_val = (m + tl.log(l))[:, None].to(L_block_ptr.type.element_ty)
    tl.store(L_block_ptr, L_val)

    



       

        

