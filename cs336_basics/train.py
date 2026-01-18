import argparse
import torch
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

from Transform import function as F
from Transform import optimizer as op
from Transform import Module as Mo

def parse_args():
    """
        解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Train a Transformer Model')

    # 模型参数
    parser.add_argument("--vocab_size", type=int, default=10000, help="词表大小")
    parser.add_argument("--d_model", type=int, default=768, help="模型的维度")
    parser.add_argument("--num_layers", type=int, default=12, help="Transformer层数")
    parser.add_argument("--num_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--d_ff", type=int, default=3072, help="FFN层的隐藏层维度")
    parser.add_argument("--context_length", type=int, default=256, help="上下文长度")
    parser.add_argument("--theta", type=float, default=10000.0, help="RoPE的theta参数")

    # 训练参数
    parser.add_argument("--device", type=str, default="cuda", help="训练设备(cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=8, help="训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="最大学习率")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="最小学习率")
    parser.add_argument("--max_steps", type=int, default=10000, help="总训练步数")
    parser.add_argument("--warmup_steps", type=int, default=100, help="学习率预热步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔(步)")
    parser.add_argument("--val_interval", type=int, default=100, help="验证间隔(步)")

    # 路径参数
    parser.add_argument("--train_path", type=str, default=None, help="训练数据路径")
    parser.add_argument("--val_path", type=str, default=None, help="验证数据路径")
    parser.add_argument("--save_checkpoint_path", type=str, default=None, help="检查点保存路径")
    parser.add_argument("--load_checkpoint_path", type=str, default=None, help="加载检查点路径")
    parser.add_argument("--log_dir", type=str, default="runs", help="TensorBoard日志目录")

    return parser.parse_args()

def load_data(path: str) -> np.ndarray:
    """
        训练数据应该是已经 tokenize 好的 .npy 或 .bin 文件（整数数组）
        使用 np.memmap 可以不把整个文件加载到内存
    """
    data = np.memmap(path, dtype=np.uint16, mode='r')
    return data

def estimate_training_time(model, train_data, batch_size, context_length, device, max_steps, warmup_runs=5, benchmark_runs=10):
    """
        估算训练时间
        通过运行几个 warmup step 和 benchmark step 来估算每步耗时
    """
    print("Estimating training time...")

    # Warmup: 让 GPU 预热
    for _ in range(warmup_runs):
        inputs, targets = F.get_batch(train_data, batch_size, context_length, device)
        logits = model(inputs)
        loss = F.CrossEntropy(logits, targets)
        loss.backward()
        model.zero_grad()

    # 同步 GPU（确保 warmup 完成）
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark: 计时
    start_time = time.time()
    for _ in range(benchmark_runs):
        inputs, targets = F.get_batch(train_data, batch_size, context_length, device)
        logits = model(inputs)
        loss = F.CrossEntropy(logits, targets)
        loss.backward()
        model.zero_grad()

    # 同步 GPU（确保 benchmark 完成）
    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    # 计算每步平均耗时
    time_per_step = (end_time - start_time) / benchmark_runs
    total_time_seconds = time_per_step * max_steps

    # 转换为可读格式
    hours = int(total_time_seconds // 3600)
    minutes = int((total_time_seconds % 3600) // 60)
    seconds = int(total_time_seconds % 60)

    print(f"Estimated time per step: {time_per_step*1000:.2f} ms")
    print(f"Estimated total training time: {hours}h {minutes}m {seconds}s ({max_steps} steps)")
    print("-" * 50)

    return time_per_step

def main():
    """
        主训练函数
    """
    # 解析参数
    args = parse_args()
    # 模型参数
    vocab_size = args.vocab_size
    d_model = args.d_model
    num_layers = args.num_layers
    num_heads = args.num_heads
    d_ff = args.d_ff
    context_length = args.context_length
    theta = args.theta
    
    # 训练参数
    device = args.device
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    min_lr = args.min_lr
    max_steps = args.max_steps
    warmup_steps = args.warmup_steps
    grad_clip = args.grad_clip
    weight_decay = args.weight_decay
    log_interval = args.log_interval
    val_interval = args.val_interval
    
    # 路径参数
    train_path = args.train_path
    val_path = args.val_path
    save_checkpoint_path = args.save_checkpoint_path
    load_checkpoint_path = args.load_checkpoint_path
    log_dir = args.log_dir

    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # 加载数据
    train_data = load_data(train_path)
    val_data = load_data(val_path) if val_path is not None else None
   
    # 创建模型
    model = Mo.Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        context_length=context_length,
        num_heads=num_heads,
        num_layers=num_layers,
        rope_theta=theta,
        device=device
    )

    # 创建优化器
    optimizer = op.AdamW(
        params=model.parameters(),
        lr = learning_rate,
        weight_decay=weight_decay
    )

    # 加载检查点：可选
    start_step = 0
    if load_checkpoint_path is not None:
        start_step = F.load_checkpoint(
            src = load_checkpoint_path,
            model=model,
            optimizer=optimizer
        )

    # 估算训练时间
    estimate_training_time(
        model=model,
        train_data=train_data,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
        max_steps=max_steps - start_step
    )

    # 循环训练
    for step in range(start_step, max_steps):
        # 梯度清0
        optimizer.zero_grad()
        # 采样batch
        inputs, targets = F.get_batch(
            x = train_data,
            batch_size =  batch_size,
            context_length = context_length,
            device = device
        )
        # 前向传播
        logits = model(inputs)
        # 计算loss,
        loss = F.CrossEntropy(logits,targets)
        # 反向传播
        loss.backward()
        # 梯度裁剪
        F.gradient_clipping(
            params = model.parameters(),
            max_l2_norm = grad_clip,
        )
        # 更新学习率
        lr = F.lr_cos_scheduler(
            t = step,
            a_max = learning_rate,
            a_min = min_lr,
            T_w = warmup_steps,
            T_c = max_steps
        )
        for para_group in optimizer.param_groups:
            para_group['lr'] = lr

        # 更新参数(优化器)
        optimizer.step()

        # 日志打印
        if step % log_interval == 0:
            print(f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | LR: {lr:.6f}")
            # TensorBoard 记录训练指标
            writer.add_scalar('Train/Loss', loss.item(), step)
            writer.add_scalar('Train/LearningRate', lr, step)

        # 验证
        if val_data is not None and step % val_interval == 0 and step > 0:
            model.eval()
            with torch.no_grad():
                val_inputs, val_targets = F.get_batch(
                    x=val_data,
                    batch_size=batch_size,
                    context_length=context_length,
                    device=device
                )
                val_logits = model(val_inputs)
                val_loss = F.CrossEntropy(val_logits, val_targets)
            model.train()
            print(f"Step {step}/{max_steps} | Val Loss: {val_loss.item():.4f}")
            # TensorBoard 记录验证指标
            writer.add_scalar('Val/Loss', val_loss.item(), step)

        # 保存checkpoint
        if save_checkpoint_path is not None and step % val_interval == 0 and step > 0:
            F.save_checkpoint(
                model = model,
                optimizer = optimizer,
                iteration = step,
                out = save_checkpoint_path
            )

    # 保存训练最终权重
    if save_checkpoint_path is not None:
        F.save_checkpoint(
            model = model,
            optimizer = optimizer,
            iteration = max_steps,
            out = save_checkpoint_path
        )

    # 关闭 TensorBoard writer
    writer.close()

    print(f"Training finished. Checkpoint saved to {save_checkpoint_path}")

if __name__ == "__main__":
    main()