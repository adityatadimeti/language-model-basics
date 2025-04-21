import torch
from torch import nn
from einops import rearrange, einsum
from torch.nn.init import trunc_normal_
import math


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None=None, dtype: torch.dtype | None=None):
        """
        Linear transformation module, without bias term.
        """

        super().__init__()

        self.std = 2.0/(in_features + out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.device = device

        self.weights = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        trunc_val = 3.0*math.sqrt(self.std)
        self.initialize_weights(self.weights, mean=0, std=self.std, trunc_low=-trunc_val, trunc_high=trunc_val)

    def initialize_weights(self, weights, mean: float, std: float, trunc_low: float | None = None, trunc_high: float | None = None ):
        trunc_normal_(tensor=weights, mean=mean, std = std, a = trunc_low, b = trunc_high)

        return einsum(self.weights, x, 'd_out d_in, ... d_in -> ... d_out')


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None=None, dtype: torch.dtype | None=None):
        """
        Performs an embedding lookup.
        num_embeddings: Size of the vocabulary
        embedding_dim: Dimension of the embedding vectors (d_model)
        device: Device to store parameters on
        dtype: Data type of the parameters
        """

        super().__init__()
        self.std = 1
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weights = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device))
        trunc_val = 3.0
        self.initialize_weights(self.weights, mean=0, std=self.std, trunc_low=-trunc_val, trunc_high=trunc_val)


    def initialize_weights(self, weights, mean: float, std: float, trunc_low: float | None = None, trunc_high: float | None = None ):
        trunc_normal_(tensor=weights, mean=mean, std = std, a = trunc_low, b = trunc_high)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]



class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """

        # First upcast input to torch.float32 to prevent overflow when squaring the input
        in_dtype = x.dtype
        x_upcasted = x.to(torch.float32)

        squared = x_upcasted.pow(2)
        mean_sq = squared.mean(dim=-1, keepdim=True)


        rms_val = torch.sqrt(mean_sq + self.eps)  

        result = einsum((x_upcasted / rms_val), self.weights, "... d_model, ... d_model -> ... d_model")

        return result.to(in_dtype)
    
class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype:  torch.dtype | None = None):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        

        self.w1 = Linear(in_features=self.d_model, out_features=self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=self.d_ff, out_features=self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=self.d_model, out_features=self.d_ff, device=device, dtype=dtype)
    
    def silu(self, x: torch.Tensor) -> torch.Tensor:
        x_sig = torch.sigmoid(x)
        return x_sig * x

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        silu_tensor = self.silu(einsum(self.w1.weights, x, "d_ff d_model, ... d_model -> ... d_ff"))
        w3_x = einsum(self.w3.weights, x, "d_ff d_model, ... d_model -> ... d_ff")
        w2_input = einsum(silu_tensor, w3_x, "... d_ff, ... d_ff -> ... d_ff")
        
        return self.w2(w2_input)

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        positions = torch.arange(0, max_seq_len) 
        frac_value = torch.arange(0, self.d_k//2) 
        inv_freq = 1.0 / (torch.pow(theta, 2 * frac_value / d_k))

        angle = einsum(positions, inv_freq, "i, j -> i j")
        
        
        cos_tensor = torch.cos(angle)
        sin_tensor = torch.sin(angle)


        self.register_buffer('sin', sin_tensor, persistent=False)
        self.register_buffer('cos', cos_tensor, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_evens = x[..., ::2]
        x_odds  = x[...,  1::2]

        cos = self.cos[token_positions]
        sin = self.sin[token_positions]


        if cos.ndim == 2:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        x_rotated_even = x_evens * cos - x_odds * sin
        x_rotated_odd  = x_evens * sin + x_odds * cos

        x_rot = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)  # (..., seq, d//2, 2)
        x_rot = rearrange(x_rot, "... s d p -> ... s (d p)")           # (..., seq, d)

        return x_rot

def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    """
    x is the tensor to apply softmax to
    We apply softmax to the i-th dimension of the input tensor.
    """

    max_val = x.max(dim=i, keepdim=True).values
    normalized_x = x - max_val
    return torch.exp(normalized_x) / torch.sum(torch.exp(normalized_x), dim=i, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    mask is an optional boolean mask of shape (seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    q_k = einsum(Q, K, "batch_size ... n d_k, batch_size ... m d_k -> batch_size ... n m") / math.sqrt(d_k)
    if mask is not None:
        q_k = softmax(torch.where(mask, q_k, float('-inf')), -1)
    q_k_v = einsum(q_k, V, "batch_size ... n m, batch_size ... m d_v -> batch_size ... n d_v")
    return q_k_v




class CausalMultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, q_proj: torch.Tensor, k_proj: torch.Tensor, v_proj: torch.Tensor,
                 o_proj: torch.Tensor, theta: float | None =  None,token_positions: torch.Tensor | None = None, max_seq_len: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = nn.Parameter(q_proj)
        self.k_proj = nn.Parameter(k_proj)
        self.v_proj = nn.Parameter(v_proj)
        self.o_proj = nn.Parameter(o_proj)
        
        """
        Optional RoPE parameters
        """
        self.theta = theta
        self.token_positions = token_positions

        if token_positions is not None:
            """
            Applying RoPE
            """

            d_k = self.d_model // self.num_heads
            self.rope = RotaryPositionalEmbedding(theta=self.theta, d_k=d_k, max_seq_len=max_seq_len)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        sequence_length = in_features.shape[-2]
        
        Q = einsum(self.q_proj, in_features, "d_k d_in, ... sequence_length d_in -> ... sequence_length d_k")
        K = einsum(self.k_proj, in_features, "d_k d_in, ... sequence_length d_in -> ... sequence_length d_k")
        V = einsum(self.v_proj, in_features, "d_v d_in, ... sequence_length d_in -> ... sequence_length d_v")

        # now split across heads?

        Q_i = rearrange(Q, "... sequence_length (h head_dim) -> ... h sequence_length head_dim", h = self.num_heads)
        K_i = rearrange(K, "... sequence_length (h head_dim) -> ... h sequence_length head_dim", h = self.num_heads)
        V_i = rearrange(V, "... sequence_length (h head_dim) -> ... h sequence_length head_dim", h = self.num_heads)

        mask = ~torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.bool), diagonal=1)
        
        # seq_len, seq_len
        mask = rearrange(mask, 'i j -> 1 1 i j').to(in_features.device)


        if self.token_positions is not None:
            """
            Have to apply RoPE here if provided token positions
            """
            

            Q_i = self.rope(Q_i, self.token_positions)
            K_i = self.rope(K_i, self.token_positions)
        

        q_k_v = scaled_dot_product_attention(Q_i, K_i, V_i, mask=mask)
        q_k_v = rearrange(q_k_v, "... h sequence_length head_dim -> ... sequence_length (h head_dim)", h = self.num_heads)

        mha = einsum(self.o_proj, q_k_v, "d_model hd_v, ... hd_v -> ... d_model")

        return mha

        
