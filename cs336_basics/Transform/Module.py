import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import einops
from jaxtyping import Bool, Float, Int


class Linear(nn.Module):
    def __init__(self, 
        in_features: int, 
        out_features: int, 
        device: torch.device = None, 
        dtype: torch.dtype = None,
    ):
        super(Linear,self).__init__()
        weight = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = (2.0/(in_features + out_features))**0.5
        nn.init.trunc_normal_(weight, mean = 0.0, std = std, a = -std, b = std)
        nn.init.trunc_normal_(weight)
        self.W = nn.Parameter(weight)

    #  注:这里的tensor必须要大写!!!
    def forward(self, 
        x: torch.Tensor,
    ) -> torch.Tensor:
       ## einops好像要慢许多
       # return einops.einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
       return x @ self.W.T

nn.Embedding
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
        super(Embedding,self).__init__()
        weights = torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype)
        std = (2.0/(num_embeddings + embedding_dim))**0.5
        nn.init.trunc_normal_(weights, mean = 0.0, std = std, a = -std, b = std)
        # 嵌入矩阵的每一行就是一个词的嵌入向量, 比如一个词有64维向量, 嵌入矩阵取一行就是某个词的64维向量
        self.weight = nn.Parameter(weights)
    def forward(self, 
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.weight[token_ids]