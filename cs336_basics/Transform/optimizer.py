import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, 
        params, 
        lr=1e-3, 
        betas=(0.9, 0.999), 
        eps=1e-8, 
        weight_decay=0.0
    ):
        # 1. 检查参数有效性
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        # 2. 设置默认参数
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None if closure is None else closure()
        
        # 遍历每个参数组
        for group in self.param_groups:
            # 提取超参数
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            # 遍历这个组的每个参数
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # 1. 获取梯度 g
                g = p.grad.data
                # 2. 初始化或获取 state
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                # 3. 更新 m (exp_avg)
                m = state["exp_avg"]
                state["exp_avg"] = beta1 * m + (1-beta1) * g
                # 4. 更新 v (exp_avg_sq)
                v = state["exp_avg_sq"]
                state["exp_avg_sq"] = beta2 * v + (1-beta2) * g * g
                # 5. 计算 α_t (bias-corrected lr)
                t = state["step"]
                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                # 6. 更新参数 θ
                p.data -= alpha_t * state["exp_avg"] / (torch.sqrt(state["exp_avg_sq"]) + eps)
                # 7. 应用 weight decay
                p.data -= weight_decay * lr * p.data

        return loss
