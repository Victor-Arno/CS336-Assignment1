# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

## Setup

### Environment

We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.

### Run Unit Tests

**Linux/macOS:**
```bash
uv run pytest
```

**Windows (PowerShell):**

```powershell
$env:PYTHONUTF8=1
uv run pytest
```

> **Windows Notes:** You need to comment out `import resource` in `tests/test_tokenizer.py`, as the `resource` module is only supported on Unix systems.

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the functions in [./tests/adapters.py](./tests/adapters.py).

## Quick Start

### 1. Download Data

```bash
mkdir -p data && cd data

# TinyStories
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText (optional)
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz && gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz && gunzip owt_valid.txt.gz

cd ..
```

### 2. Train BPE Tokenizer

```bash
uv run python demo/train_bpe_tinystories.py
```

Output: `data/tinystories_vocab.json`, `data/tinystories_merges.txt`

### 3. Preprocess Data

```bash
uv run python demo/preprocess_tinystories.py
```

Output: `data/tinystories_train.bin`, `data/tinystories_val.bin`

### 4. Train Model

```bash
cd cs336_basics && uv run python train.py \
    --train_path ../data/tinystories_train.bin \
    --val_path ../data/tinystories_val.bin \
    --save_checkpoint_path ../checkpoints/tinystories.pt \
    --log_dir ../runs/tinystories
```

Training time: ~2 hours on 2080Ti, ~30-40 min on H100

### 5. Monitor Training

```bash
tensorboard --logdir runs/tinystories
```

### 6. Generate Text

```bash
cd cs336_basics && uv run python generate.py \
    --checkpoint ../checkpoints/tinystories_best.pt \
    --prompt "Once upon a time," \
    --temperature 0.8 --top_p 0.9
```

### 7. Interactive Chat

```bash
cd cs336_basics && uv run python chat.py
```

Example:
```
[temp=0.8, top_p=0.9, top_k=50, max=256]
>>> Once upon a time,
Once upon a time, there was a little girl named Lily...

>>> /temp 0.7 /top_p 0.85 The brave knight
The brave knight went on an adventure...
```

Commands: `/temp`, `/top_p`, `/top_k`, `/max`, `/settings`, `/help`, `/quit`

## Project Structure

```
├── cs336_basics/
│   ├── Tokenizer/       # BPE tokenizer implementation
│   ├── Transform/       # Transformer modules (attention, FFN, etc.)
│   └── train.py         # Training script
├── demo/
│   ├── train_bpe_tinystories.py    # Train BPE tokenizer
│   └── preprocess_tinystories.py   # Preprocess data
├── tests/
│   ├── adapters.py      # Connect implementation to tests
│   └── test_*.py        # Unit tests
└── data/                # Training data (download required)
```

