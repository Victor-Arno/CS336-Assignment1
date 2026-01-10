import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from jaxtyping import Bool, Float, Int

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
    
class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,
        x: Float[torch.Tensor, " ..."]
    ) -> Float[torch.Tensor, " ..."]:
        return x * torch.sigmoid(x)
        
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
        
        silu = SiLU()
        return self.w2(silu(self.w1(x)) * self.w3(x))