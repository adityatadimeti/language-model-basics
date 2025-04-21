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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        frac_value = torch.arange(1, self.d_k//2)


        self.register_buffer('rope', persistent=False)

    #def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor: