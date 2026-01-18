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

def lr_cos_scheduler(
    t,
    a_max,
    a_min,
    T_w,
    T_c
):
    """
        预热阶段: 学习率从0增加到a_max
        余弦退火: 按余弦曲线从a_max平滑下降到a_min
        恒定阶段: 学习率保持在a_min
    """
    if t < T_w:
        a_t = t * a_max / T_w
    elif t > T_c:
        a_t = a_min
    else:
        a_t = a_min + 0.5 * (a_max - a_min) * (1 + math.cos(math.pi * ((t-T_w)/(T_c-T_w))))

    return a_t     

def gradient_clipping(
    params,
    max_l2_norm,
    eps = 1e-6
):
    
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            para_norm = p.grad.data.norm(2)
            total_norm += para_norm ** 2
    total_norm = total_norm ** 0.5
    clip_coef = max_l2_norm / (total_norm + eps)
    # 若需要裁剪,则裁剪所有梯度
    if total_norm >= max_l2_norm:
        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    return True

def get_batch(
        x:  np.ndarray,
        batch_size: int,
        context_length: int,
        device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # 1. 随机采样位置
    n = len(x)
    # 生成batch_size个随机位置,范围是(0,n-context__length)
    start_indices = np.random.randint(0, n - context_length, size=batch_size)

    # 2. 提取输入
    inputs = []
    for i in start_indices:
        inputs.append(x[i:i+context_length])
    inputs = np.array(inputs) 
    inputs = torch.from_numpy(inputs).to(device=device,dtype=torch.long)
    
    # 3. 获取targets
    targets = []
    for i in start_indices:
        targets.append(x[i+1:i+1+context_length])
    targets = np.array(targets)
    targets = torch.from_numpy(targets).to(device=device,dtype=torch.long)
    return inputs,targets

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str
):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(
        obj = checkpoint,
        f = out
    )


def load_checkpoint(
    src: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['iteration']
