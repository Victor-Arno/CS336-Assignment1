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

- **cs336_basics/Transform/Module.py**: Neural network building blocks (Linear, Embedding, RMSNorm, etc.)
- **cs336_basics/Tokenizer/BPE_tokenizer.py**: BPE tokenizer training and encoding/decoding implementation
- **tests/adapters.py**: Bridge between test suite and student implementations - **complete functions here to connect your implementations to tests**

### Implementation Pattern

The assignment uses an adapter pattern:
1. Implement neural network modules in `cs336_basics/Transform/Module.py`
2. Implement tokenizer logic in `cs336_basics/Tokenizer/BPE_tokenizer.py`
3. Wire up implementations in `tests/adapters.py` to make tests pass

### Key Components to Implement

**Neural Network Modules** (in Module.py):
- Linear layer (no bias, custom initialization)
- Embedding layer
- RMSNorm
- SwiGLU feed-forward network
- Scaled dot-product attention
- Multi-head self-attention (with and without RoPE)
- Transformer block
- Full Transformer language model

**Tokenizer** (in BPE_tokenizer.py):
- BPE training with parallel processing
- GPT-2 style pre-tokenization regex
- Encode/decode functionality
- Special token handling

**Training Utilities** (adapters reference these):
- Cross-entropy loss
- Softmax
- SiLU activation
- AdamW optimizer
- Cosine learning rate schedule with warmup
- Gradient clipping
- Checkpointing (save/load)
- Batch sampling

### Data

Download training data to `data/` directory:
- TinyStories: TinyStoriesV2-GPT4-train.txt, TinyStoriesV2-GPT4-valid.txt
- OpenWebText sample: owt_train.txt, owt_valid.txt

### Testing

Tests use snapshot testing (stored in `tests/_snapshots/`) to verify numerical correctness. Tests initially fail with `NotImplementedError` - implementations make them pass.

## Code Style

- Uses `jaxtyping` for tensor shape annotations (e.g., `Float[Tensor, "batch seq d_model"]`)
- Uses `einops` for tensor operations
- Line length: 120 (configured in pyproject.toml via ruff)
