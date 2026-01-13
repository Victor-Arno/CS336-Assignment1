"""
GPU Support Test File for CS336 Assignment 1

This file tests all implemented modules to ensure they work correctly
on both CPU and GPU (if available).
"""

import sys
from pathlib import Path

# Add parent directory to path to import cs336_basics
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from cs336_basics.Transform.Module import (
    Linear,
    Embedding,
    RMSNorm,
    SwiGLU,
    RotaryPositionalEmbedding,
    MultiHeadSelfAttention,
    TransformerBlock
)
from cs336_basics.Transform import function as F


def test_device_support(device):
    """Test all modules on the specified device"""
    print(f"\n{'='*60}")
    print(f"Testing on device: {device}")
    print(f"{'='*60}\n")

    # Test 1: Linear Layer
    print("1. Testing Linear Layer...")
    linear = Linear(64, 128, device=device, dtype=torch.float32)
    x = torch.randn(2, 10, 64, device=device)
    y = linear(x)
    assert y.device == x.device, "Linear output device mismatch!"
    assert y.shape == (2, 10, 128), f"Linear output shape wrong: {y.shape}"
    print(f"   ✓ Linear: Input device={x.device}, Output device={y.device}, Shape={y.shape}")

    # Test 2: Embedding Layer
    print("\n2. Testing Embedding Layer...")
    embedding = Embedding(1000, 64, device=device, dtype=torch.float32)
    token_ids = torch.randint(0, 1000, (2, 10), device=device)
    embeddings = embedding(token_ids)
    assert embeddings.device == token_ids.device, "Embedding output device mismatch!"
    assert embeddings.shape == (2, 10, 64), f"Embedding output shape wrong: {embeddings.shape}"
    print(f"   ✓ Embedding: Token device={token_ids.device}, Output device={embeddings.device}, Shape={embeddings.shape}")

    # Test 3: RMSNorm
    print("\n3. Testing RMSNorm...")
    rmsnorm = RMSNorm(64, device=device, dtype=torch.float32)
    x = torch.randn(2, 10, 64, device=device)
    y = rmsnorm(x)
    assert y.device == x.device, "RMSNorm output device mismatch!"
    assert y.shape == x.shape, f"RMSNorm output shape wrong: {y.shape}"
    print(f"   ✓ RMSNorm: Input device={x.device}, Output device={y.device}, Shape={y.shape}")
    print(f"   ✓ G_matrix device={rmsnorm.G_matrix.device}")

    # Test 4: SwiGLU
    print("\n4. Testing SwiGLU...")
    swiglu = SwiGLU(64, 256, device=device, dtype=torch.float32)
    x = torch.randn(2, 10, 64, device=device)
    y = swiglu(x)
    assert y.device == x.device, "SwiGLU output device mismatch!"
    assert y.shape == x.shape, f"SwiGLU output shape wrong: {y.shape}"
    print(f"   ✓ SwiGLU: Input device={x.device}, Output device={y.device}, Shape={y.shape}")

    # Test 5: RotaryPositionalEmbedding
    print("\n5. Testing RotaryPositionalEmbedding...")
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=32, max_seq_len=512, device=device)
    x = torch.randn(2, 8, 10, 32, device=device)  # (batch, heads, seq, d_k)
    y = rope(x)
    assert y.device == x.device, "RoPE output device mismatch!"
    assert y.shape == x.shape, f"RoPE output shape wrong: {y.shape}"
    print(f"   ✓ RoPE: Input device={x.device}, Output device={y.device}, Shape={y.shape}")
    print(f"   ✓ cos_cache device={rope.cos_cache.device}")
    print(f"   ✓ sin_cache device={rope.sin_cache.device}")

    # Test 6: Scaled Dot-Product Attention
    print("\n6. Testing Scaled Dot-Product Attention...")
    Q = torch.randn(2, 8, 10, 32, device=device)  # (batch, heads, queries, d_k)
    K = torch.randn(2, 8, 10, 32, device=device)  # (batch, heads, keys, d_k)
    V = torch.randn(2, 8, 10, 32, device=device)  # (batch, heads, keys, d_v)
    mask = torch.tril(torch.ones(10, 10, device=device)).bool()
    output = F.scaled_dot_product_attention(Q, K, V, mask)
    assert output.device == Q.device, "Attention output device mismatch!"
    assert output.shape == (2, 8, 10, 32), f"Attention output shape wrong: {output.shape}"
    print(f"   ✓ Attention: Q device={Q.device}, Output device={output.device}, Shape={output.shape}")

    # Test 7: MultiHeadSelfAttention (without RoPE)
    print("\n7. Testing MultiHeadSelfAttention (without RoPE)...")
    mha = MultiHeadSelfAttention(d_model=64, num_heads=8, device=device, dtype=torch.float32)
    x = torch.randn(2, 10, 64, device=device)
    y = mha(x)
    assert y.device == x.device, "MHA output device mismatch!"
    assert y.shape == x.shape, f"MHA output shape wrong: {y.shape}"
    print(f"   ✓ MHA: Input device={x.device}, Output device={y.device}, Shape={y.shape}")

    # Test 8: MultiHeadSelfAttention (with RoPE)
    print("\n8. Testing MultiHeadSelfAttention (with RoPE)...")
    mha_rope = MultiHeadSelfAttention(
        d_model=64,
        num_heads=8,
        max_seq_len=512,
        theta=10000.0,
        device=device,
        dtype=torch.float32
    )
    x = torch.randn(2, 10, 64, device=device)
    y = mha_rope(x)
    assert y.device == x.device, "MHA+RoPE output device mismatch!"
    assert y.shape == x.shape, f"MHA+RoPE output shape wrong: {y.shape}"
    print(f"   ✓ MHA+RoPE: Input device={x.device}, Output device={y.device}, Shape={y.shape}")
    print(f"   ✓ RoPE cos_cache device={mha_rope.rope.cos_cache.device}")

    # Test 9: TransformerBlock
    print("\n9. Testing TransformerBlock...")
    block = TransformerBlock(
        d_model=64,
        num_heads=8,
        d_ff=256,
        theta=10000.0,
        max_seq_len=512,
        device=device,
        dtype=torch.float32
    )
    x = torch.randn(2, 10, 64, device=device)
    y = block(x)
    assert y.device == x.device, "TransformerBlock output device mismatch!"
    assert y.shape == x.shape, f"TransformerBlock output shape wrong: {y.shape}"
    print(f"   ✓ TransformerBlock: Input device={x.device}, Output device={y.device}, Shape={y.shape}")
    print(f"   ✓ norm1.G_matrix device={block.norm1.G_matrix.device}")
    print(f"   ✓ MHA.W_Q.W device={block.MHA.W_Q.W.device}")
    print(f"   ✓ ffn.w1.W device={block.ffn.w1.W.device}")

    print(f"\n{'='*60}")
    print(f"All tests passed on {device}! ✅")
    print(f"{'='*60}\n")


def main():
    """Main test function"""
    print("\n" + "="*60)
    print("CS336 Assignment 1 - GPU Support Test")
    print("="*60)

    # Test on CPU
    print("\n🖥️  Testing on CPU...")
    test_device_support(torch.device('cpu'))

    # Test on GPU if available
    if torch.cuda.is_available():
        print("\n🚀 GPU detected! Testing on CUDA...")
        test_device_support(torch.device('cuda'))

        # Additional GPU-specific test: verify memory transfer
        print("\n" + "="*60)
        print("Testing CPU-to-GPU transfer...")
        print("="*60 + "\n")

        # Create on CPU
        block_cpu = TransformerBlock(
            d_model=64, num_heads=8, d_ff=256,
            theta=10000.0, max_seq_len=512,
            device=torch.device('cpu')
        )
        x_cpu = torch.randn(2, 10, 64)

        # Move to GPU
        print("Moving model to GPU...")
        block_gpu = TransformerBlock(
            d_model=64, num_heads=8, d_ff=256,
            theta=10000.0, max_seq_len=512,
            device=torch.device('cuda')
        )
        x_gpu = x_cpu.cuda()

        # Test
        y_gpu = block_gpu(x_gpu)
        print(f"✓ Input moved to GPU: {x_gpu.device}")
        print(f"✓ Output on GPU: {y_gpu.device}")
        print(f"✓ Model weights on GPU: {block_gpu.MHA.W_Q.W.device}")

        print("\n" + "="*60)
        print("GPU tests passed! 🎉")
        print("="*60)
    else:
        print("\n⚠️  No GPU available. Skipping CUDA tests.")

    print("\n" + "="*60)
    print("All tests completed successfully! ✅")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
