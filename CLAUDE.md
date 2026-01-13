# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Stanford CS336 Spring 2025 Assignment 1: Basics. The assignment involves implementing core components of a transformer-based language model from scratch, including BPE tokenization, neural network modules, and training utilities.

## Commands

```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_tokenizer.py

# Run a specific test
uv run pytest tests/test_model.py::test_rmsnorm

# Run any Python file
uv run <python_file_path>
```

Environment is managed with `uv` - dependencies are automatically resolved and activated when using `uv run`.

## Architecture

### Module Structure

- **cs336_basics/Transform/Module.py**: Neural network building blocks (Linear, Embedding, RMSNorm, SwiGLU, RoPE, attention, etc.)
- **cs336_basics/Tokenizer/**: BPE tokenizer package
  - `BPE_tokenizer.py`: Main entry point, exports trainers and tokenizer
  - `base.py`: Shared utilities (GPT2_TOKENIZER_REGEX, chunk processing, `Tokenizer` class for encode/decode)
  - `trainer_linear.py` / `trainer_heap.py`: Two BPE training implementations (linear vs heap-based)
- **tests/adapters.py**: Bridge between test suite and implementations - **implement adapter functions here to wire up your code to tests**

### Implementation Pattern

The assignment uses an adapter pattern:
1. Implement neural network modules in `cs336_basics/Transform/Module.py`
2. Implement tokenizer logic in `cs336_basics/Tokenizer/` (base.py for shared code, trainers for BPE training)
3. Wire up implementations in `tests/adapters.py` to make tests pass

### Key Components to Implement

**Neural Network Modules** (in Module.py):
- Linear layer (no bias, truncated normal initialization with std=sqrt(2/(in+out)))
- Embedding layer
- RMSNorm (upcast to float32 internally to prevent overflow)
- SiLU activation
- SwiGLU feed-forward network: `W2 * (SiLU(W1 * x) ⊙ W3 * x)`
- RotaryPositionalEmbedding (RoPE) - uses interleaved (x0,x1), (x2,x3) pairing
- Scaled dot-product attention
- Multi-head self-attention (with and without RoPE)
- Transformer block (pre-norm architecture)
- Full Transformer language model

**Tokenizer** (in Tokenizer/):
- BPE training with parallel chunk processing
- GPT-2 style pre-tokenization regex: `'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
- Encode/decode with special token handling
- Merge tie-breaking: higher frequency wins, then lexicographically larger pair

**Training Utilities** (in adapters.py):
- Cross-entropy loss, Softmax
- AdamW optimizer
- Cosine learning rate schedule with linear warmup
- Gradient clipping (by global L2 norm)
- Checkpointing (save/load model, optimizer, iteration)
- Batch sampling from dataset

### Data

Download training data to `data/` directory:
```bash
mkdir -p data && cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz && gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz && gunzip owt_valid.txt.gz
```

### Testing

Tests use snapshot testing (stored in `tests/_snapshots/`) to verify numerical correctness. Tests initially fail with `NotImplementedError` - implementations make them pass.

### Weight Loading Convention

In adapters.py, reference weights use specific naming conventions. For the custom Linear layer (which stores weights as `W` not `weight`), access via `.W.data` (e.g., `swiglu.w1.W.data = w1_weight`). For RMSNorm, the gain parameter is named `G_matrix`.

## Code Style

- Uses `jaxtyping` for tensor shape annotations (e.g., `Float[Tensor, "batch seq d_model"]`)
- Uses `einops` for tensor operations
- Uses `regex` module (not `re`) for GPT-2 pre-tokenization (supports Unicode properties like `\p{L}`)
- Line length: 120 (configured in pyproject.toml via ruff)
