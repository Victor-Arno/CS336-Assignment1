import torch
import torch.nn as nn
import numpy as np
import einops
from jaxtyping import Bool, Float, Int
import math



def silu(
        x: torch.Tensor
    ) -> torch.Tensor:

    return x * torch.sigmoid(x)

def softmax(
        x: torch.Tensor, 
        dim: int = -1
    ) -> torch.Tensor:

    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(
        Q: Float[torch.Tensor, " ... queries d_k"],
        K: Float[torch.Tensor, " ... keys d_k"],
        V: Float[torch.Tensor, " ... values d_v"],
        mask:  Bool[torch.Tensor, " ... queries keys"] | None = None
) -> Float[torch.Tensor, " ... queries d_v"]:
    
    # 1. 获取d_k
    d_k = Q.size(-1)
    # 2. 计算scores,原始分数
    scores = einops.einsum(Q,K,"... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    # 3. 若有mask.应用mask
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    # 4. 计算Att矩阵,即注意力矩阵
    Att = softmax(scores,dim=-1)
    # 5. 注意力输出
    result = Att @ V
    return result

def CrossEntropy(logits, targets):
    # 1.数值稳定, 减去最大值
    logits_stable = logits - logits.max(dim=-1, keepdim=True).values
    # 2.计算logsumexp
    logsumexp = torch.logsumexp(logits_stable,dim=-1)
    # 3.取出target位置的logit
    target_logits = torch.gather(logits_stable, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # 4.计算loss并且返回平均值
    loss = -target_logits + logsumexp
    return loss.mean()