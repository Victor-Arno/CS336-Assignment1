#!/usr/bin/env python3
"""Train BPE tokenizer on TinyStories dataset"""

import json
import time
import psutil
import os
import cProfile
import pstats
from io import StringIO

from cs336_basics.Tokenizer.BPE_tokenizer import BPETrainerHeap, Tokenizer

# Get project root directory (parent of demo/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def get_memory_usage_gb():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def train_bpe():
    # Configuration
    input_path = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt")
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    output_vocab_path = os.path.join(DATA_DIR, "tinystories_vocab.json")
    output_merges_path = os.path.join(DATA_DIR, "tinystories_merges.txt")

    print(f"Training BPE tokenizer on {input_path}")
    print(f"Vocab size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    print()

    # Record start time and memory
    start_time = time.time()
    start_memory = get_memory_usage_gb()

    # Train BPE
    trainer = BPETrainerHeap()
    vocab, merges = trainer.train_BPE(input_path, vocab_size, special_tokens)

    # Record end time and memory
    end_time = time.time()
    end_memory = get_memory_usage_gb()

    training_time = end_time - start_time
    memory_used = end_memory - start_memory

    print()
    print("=" * 50)
    print("Training Results:")
    print("=" * 50)
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Peak memory usage: ~{end_memory:.2f} GB")
    print(f"Vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

    # Save vocab and merges
    vocab_json = {v.decode("latin1"): k for k, v in vocab.items()}
    with open(output_vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)
    print(f"Saved vocab to {output_vocab_path}")

    with open(output_merges_path, "w", encoding="utf-8") as f:
        for m1, m2 in merges:
            f.write(f"{m1.decode('latin1')} {m2.decode('latin1')}\n")
    print(f"Saved merges to {output_merges_path}")

    # Find longest token
    print()
    print("=" * 50)
    print("Vocabulary Analysis:")
    print("=" * 50)

    longest_token = max(vocab.values(), key=len)
    longest_token_id = [k for k, v in vocab.items() if v == longest_token][0]

    print(f"Longest token ID: {longest_token_id}")
    print(f"Longest token length: {len(longest_token)} bytes")

    try:
        decoded = longest_token.decode("utf-8")
        print(f"Longest token (decoded): '{decoded}'")
    except UnicodeDecodeError:
        print(f"Longest token (latin1): '{longest_token.decode('latin1')}'")

    # Show top 10 longest tokens
    print()
    print("Top 10 longest tokens:")
    sorted_by_len = sorted(vocab.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for token_id, token_bytes in sorted_by_len:
        try:
            decoded = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            decoded = token_bytes.decode("latin1")
        print(f"  ID {token_id}: len={len(token_bytes):2d}, '{repr(decoded)}'")

    return vocab, merges, training_time, end_memory


def profile_training():
    """Profile the training process"""
    print()
    print("=" * 50)
    print("Profiling Training Process:")
    print("=" * 50)

    input_path = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt")
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    profiler = cProfile.Profile()
    profiler.enable()

    trainer = BPETrainerHeap()
    vocab, merges = trainer.train_BPE(input_path, vocab_size, special_tokens)

    profiler.disable()

    # Print profiling results
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    stats.print_stats(30)
    print(s.getvalue())


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--profile":
        profile_training()
    else:
        train_bpe()
