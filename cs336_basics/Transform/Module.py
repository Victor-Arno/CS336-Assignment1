import torch
import torch.nn as nn
import numpy as np
import einops
from jaxtyping import Bool, Float, Int
import math
from . import function as F

"""
    所有参数默认在cpu上运行?
    需要检查
"""
class Linear(nn.Module):
    def __init__(self, 
        in_features: int, 
        out_features: int, 
        device: torch.device = None, 
        dtype: torch.dtype = None,
    ):
        super().__init__()
        weight = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = (2.0/(in_features + out_features))**0.5
        nn.init.trunc_normal_(weight, mean = 0.0, std = std, a = -std, b = std)
        nn.init.trunc_normal_(weight)
        # 初始化所有参数
        self.W = nn.Parameter(weight)
        self.device = device
        self.dtype = dtype
    
    #  注:这里的tensor必须要大写!!!
    def forward(self, 
        x: torch.Tensor,
    ) -> torch.Tensor:
       """
            pyTorch 自动处理运算
            当你执行 x @ self.W.T 时：
            如果 x 和 self.W 在同一设备上，运算自动在该设备进行
            输出张量会自动继承输入的 device/dtype
       """
       ## einops好像要慢许多
       # return einops.einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")

       return x @ self.W.T

class Embedding(nn.Module):
    def __init__(self, 
        num_embeddings: int, 
        embedding_dim: int, 
        device: torch.device = None, 
        dtype: torch.device = None
    ): 
        """
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        weights = torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype)
        std = (2.0/(num_embeddings + embedding_dim))**0.5
        nn.init.trunc_normal_(weights, mean = 0.0, std = std, a = -std, b = std)
        # 嵌入矩阵的每一行就是一个词的嵌入向量, 比如一个词有64维向量, 嵌入矩阵取一行就是某个词的64维向量
        # self.weight = nn.Parameter(weights)
        self.weight = weights
        self.device = device
        self.dtype = dtype

    def forward(self, 
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        # token_ids = token_ids.to(device = self.device, dtype = self.dtype)
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device = None, 
        dtype: torch.device = None             
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.G_matrix = nn.Parameter(torch.randn(d_model,device=device,dtype=dtype))

    def forward(self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """
            Process an input tensor of shape (batch_size, sequence_length, d_model) 
            and return a tensor of the same shape.
        """

        # upcast your input to torch.float32 to prevent overflow when you square the input
        _in_dtype = x.dtype
        x = x.to(torch.float32)

        # perform rms
        rms = torch.sqrt(self.eps + torch.mean(x**2, dim = -1, keepdim = True)) # keepdim的意思是保持维度, 否则x的维度会减少, rms的维度是(batch_size, sequence_length, 1)
        result = x / rms * self.G_matrix # 计算方式：x/rms表示让x的第三维每个元素都除以对应的rms值, 然后再乘以G_matrix

        # Return the result in the original dtype
        return result.to(_in_dtype)
        
class SwiGLU(nn.Module):
    def __init__(self,
        d_model: int,
        d_ff: int,
        device: torch.device = None,
        dtype: torch.dtype = None
    ):
        """
        SwiGLU feed-forward network.
        SwiGLU(x) = W2 * (SiLU(W1 * x) ⊙ W3 * x)

        Args:
            d_model: Input/output dimension
            d_ff: Hidden dimension
            device: Device to store parameters
            dtype: Data type of parameters
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # W1: d_model -> d_ff (gate branch)
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        # W3: d_model -> d_ff (value branch)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        # W2: d_ff -> d_model (output projection)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)

    def forward(self,
        x: Float[torch.Tensor, " ... d_model"]
    ) -> Float[torch.Tensor, " ... d_model"]:
        """
            SwiGLU(x) = W2 * (SiLU(W1 * x) ⊙ W3 * x)
        """
        
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device = None
    ):
        super().__init__()    
        fre = 1 / (theta ** (torch.arange(0, d_k, 2) / d_k))    # shape (d_k//2,)
        pos = torch.arange(max_seq_len)                          # shape (max_seq_len,)
        angle_matrix = torch.outer(pos, fre)                     # shape (max_seq_len, d_k//2)
        self.register_buffer("cos_cache", torch.cos(angle_matrix), persistent = False) 
        self.register_buffer("sin_cache", torch.sin(angle_matrix), persistent = False) 
    
    def forward(self,
        x: torch.Tensor,
        token_positions: torch.Tensor
    ) -> torch.Tensor:
        """
            注：貌似这里不能用大模型常用的前后分割
            __init__:
            fre:          (d_k//2,)                    例: (4,)
            pos:          (max_seq_len,)               例: (5,)
            angle_matrix: (max_seq_len, d_k//2)        例: (5, 4)
            cos_cache:    (max_seq_len, d_k//2)        例: (5, 4)
            sin_cache:    (max_seq_len, d_k//2)        例: (5, 4)

            forward:
            输入 x:           (batch, seq_len, d_k)    例: (2, 3, 8)
            token_positions:  (batch, seq_len)         例: (2, 3)
            
            x_even:           (batch, seq_len, d_k//2) 例: (2, 3, 4)
            x_odd:            (batch, seq_len, d_k//2) 例: (2, 3, 4)
            cos:              (batch, seq_len, d_k//2) 例: (2, 3, 4)
            sin:              (batch, seq_len, d_k//2) 例: (2, 3, 4)
            x_even_rotate:    (batch, seq_len, d_k//2) 例: (2, 3, 4)
            x_odd_rotate:     (batch, seq_len, d_k//2) 例: (2, 3, 4)
            stack 后:         (batch, seq_len, d_k//2, 2) 例: (2, 3, 4, 2)
            输出 result:      (batch, seq_len, d_k)    例: (2, 3, 8)
        """
        # 奇偶交错分割: pairs are (x0,x1), (x2,x3), (x4,x5), ...
        x_even = x[..., 0::2]  # 取索引 0, 2, 4, ... 形状 (..., seq_len, d_k//2)
        x_odd = x[..., 1::2]   # 取索引 1, 3, 5, ... 形状 (..., seq_len, d_k//2)

        cos = self.cos_cache[token_positions]   # 形状为 (..., seq_len, d_k//2)
        sin = self.sin_cache[token_positions]   # 形状为 (..., seq_len, d_k//2)

        # 旋转公式
        x_even_rotate = x_even * cos - x_odd * sin
        x_odd_rotate = x_even * sin + x_odd * cos

        # 交错合并回去: [e0, o0, e1, o1, e2, o2, ...]
        result = torch.stack([x_even_rotate, x_odd_rotate], dim=-1)  # (..., seq_len, d_k//2, 2)
        result = result.flatten(-2)  # (..., seq_len, d_k)
        return result
    
class multihead_self_attention(nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = None,
        theta: float = None,
        device: torch.device = None,
        dtype: torch.dtype = None
    ):
        """
            adapters.py 中
            q_proj_weight: Float[Tensor, " d_k d_in"]  # d_k = num_heads * d_k_per_head
            这里的 d_k 其实是 所有头的总维度，即 num_heads * (d_model // num_heads) = d_model。
            d_k = d_model(所有头合在一起)
            d_in = d_model

            为什么写 d_k 而不是 d_model?
            adapters.py 的命名是从概念角度来写的：
                q_proj_weight 投影到 Query 空间，输出是"所有头的 Q"
                形状 (d_k, d_in) 表示：输出 d_k 维，输入 d_in 维
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.W_Q = Linear(d_model, d_model, device, dtype)
        self.W_K = Linear(d_model, d_model, device, dtype)
        self.W_V = Linear(d_model, d_model, device, dtype)
        self.W_O = Linear(d_model, d_model, device, dtype)
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)
        else:
            self.rope = None

    def forward(self,
        in_features: torch.Tensor,
        token_positions: torch.Tensor = None  # 可选
    ) -> torch.Tensor:
        # 1. 线性变换
        Q = self.W_Q(in_features)
        K = self.W_K(in_features)
        V = self.W_V(in_features)
        # 2. 拆分成多头注意力
        Q = einops.rearrange(Q,"b s (h d) -> b h s d",h = self.num_heads)
        K = einops.rearrange(K,"b s (h d) -> b h s d",h = self.num_heads)
        V = einops.rearrange(V,"b s (h d) -> b h s d",h = self.num_heads)
        # 3. 应用rope, 如果存在
        if self.rope is not None:
            Q = self.rope(Q,token_positions)
            K = self.rope(K,token_positions)
        # 4. 生成因果编码
        seq_len = in_features.size(-2)
        mask = torch.tril(torch.ones(seq_len,seq_len)).bool()
        # 调用scaled_dot_product_attention
        output = F.scaled_dot_product_attention(Q,K,V,mask)
        # 合并多头
        output = einops.rearrange(output,"b h s d -> b s (h d)")
        # 经过W_O线性层
        output = self.W_O(output)
        return output