#!/usr/bin/env python3
"""Preprocess TinyStories dataset: tokenize text and save as binary file for training"""

import json
import time
import psutil
import os
import numpy as np

from cs336_basics.Tokenizer.BPE_tokenizer import Tokenizer

# Get project root directory (parent of demo/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def get_memory_usage_gb():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def preprocess_data():
    # Configuration
    vocab_path = os.path.join(DATA_DIR, "tinystories_vocab.json")
    merges_path = os.path.join(DATA_DIR, "tinystories_merges.txt")
    special_tokens = ["<|endoftext|>"]

    # Input and output paths
    train_input_path = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt")
    val_input_path = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-valid.txt")
    train_output_path = os.path.join(DATA_DIR, "tinystories_train.bin")
    val_output_path = os.path.join(DATA_DIR, "tinystories_val.bin")

    # Check if vocab and merges exist
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        print("Error: vocab or merges file not found!")
        print(f"Expected vocab path: {vocab_path}")
        print(f"Expected merges path: {merges_path}")
        print()
        print("Please run train_bpe_tinystories.py first to generate vocab and merges.")
        return

    print("=" * 50)
    print("Preprocessing TinyStories Dataset")
    print("=" * 50)
    print(f"Vocab path: {vocab_path}")
    print(f"Merges path: {merges_path}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    start_time = time.time()
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    load_time = time.time() - start_time
    print(f"Tokenizer loaded in {load_time:.2f} seconds")
    print()

    # Process training data
    if os.path.exists(train_input_path):
        process_file(tokenizer, train_input_path, train_output_path, "Training")
    else:
        print(f"Warning: Training file not found: {train_input_path}")

    # Process validation data
    if os.path.exists(val_input_path):
        process_file(tokenizer, val_input_path, val_output_path, "Validation")
    else:
        print(f"Warning: Validation file not found: {val_input_path}")

    print()
    print("=" * 50)
    print("Preprocessing Complete!")
    print("=" * 50)
    print()
    print("Output files:")
    if os.path.exists(train_output_path):
        size_mb = os.path.getsize(train_output_path) / (1024 * 1024)
        print(f"  Train: {train_output_path} ({size_mb:.2f} MB)")
    if os.path.exists(val_output_path):
        size_mb = os.path.getsize(val_output_path) / (1024 * 1024)
        print(f"  Val:   {val_output_path} ({size_mb:.2f} MB)")


def process_file(tokenizer, input_path, output_path, name):
    """Tokenize a text file and save as binary"""
    print(f"Processing {name} data: {input_path}")

    start_time = time.time()
    start_memory = get_memory_usage_gb()

    # Read and tokenize
    print(f"  Reading file...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    file_size_mb = len(text.encode("utf-8")) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Tokenizing...")

    token_ids = tokenizer.encode(text)

    tokenize_time = time.time() - start_time
    print(f"  Tokenization time: {tokenize_time:.2f} seconds")
    print(f"  Number of tokens: {len(token_ids):,}")

    # Check if all token IDs fit in uint16
    max_token_id = max(token_ids)
    if max_token_id > 65535:
        print(f"  Warning: max token ID ({max_token_id}) exceeds uint16 range, using uint32")
        dtype = np.uint32
    else:
        dtype = np.uint16

    # Save as binary
    print(f"  Saving to {output_path}...")
    token_array = np.array(token_ids, dtype=dtype)
    token_array.tofile(output_path)

    end_time = time.time()
    end_memory = get_memory_usage_gb()

    total_time = end_time - start_time
    output_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"  Output size: {output_size_mb:.2f} MB")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Memory usage: {end_memory:.2f} GB")
    print(f"  Compression ratio: {file_size_mb / output_size_mb:.2f}x")
    print()


if __name__ == "__main__":
    preprocess_data()