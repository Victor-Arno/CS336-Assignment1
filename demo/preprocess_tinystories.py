#!/usr/bin/env python3
"""Preprocess TinyStories dataset: tokenize text and save as binary file for training"""

import json
import time
import psutil
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from cs336_basics.Tokenizer.BPE_tokenizer import Tokenizer

# Get project root directory (parent of demo/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Global tokenizer config (for multiprocessing)
_tokenizer_config = {}


def get_memory_usage_gb():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def init_worker(vocab, merges, special_tokens):
    """Initialize tokenizer in worker process"""
    global _worker_tokenizer
    _worker_tokenizer = Tokenizer(vocab, merges, special_tokens)


def tokenize_chunk(lines):
    """Tokenize a chunk of lines (called in worker process)"""
    global _worker_tokenizer
    result = []
    for line in lines:
        result.extend(_worker_tokenizer.encode(line))
    return result


def load_tokenizer_config(vocab_path, merges_path, special_tokens):
    """Load vocab and merges for tokenizer"""
    # Load vocab
    with open(vocab_path, "r") as vf:
        vocab_data = json.load(vf)
        vocab = {int(i): bytes(v, "latin1") for v, i in vocab_data.items()}

    # Load merges with correct parsing (rsplit to handle space tokens)
    merges = []
    with open(merges_path, "r") as mf:
        for line in mf:
            line = line.rstrip('\n')
            if line and not line.startswith("#"):
                parts = line.rsplit(' ', 1)
                if len(parts) == 2:
                    merges.append((bytes(parts[0], "latin1"), bytes(parts[1], "latin1")))

    return vocab, merges, special_tokens


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

    # Load tokenizer config
    print("Loading tokenizer...")
    start_time = time.time()
    vocab, merges, special_tokens = load_tokenizer_config(vocab_path, merges_path, special_tokens)
    load_time = time.time() - start_time
    print(f"Tokenizer loaded in {load_time:.2f} seconds")
    print(f"Vocab size: {len(vocab)}, Merges: {len(merges)}")
    print()

    # Process training data
    if os.path.exists(train_input_path):
        process_file_parallel(vocab, merges, special_tokens, train_input_path, train_output_path, "Training")
    else:
        print(f"Warning: Training file not found: {train_input_path}")

    # Process validation data
    if os.path.exists(val_input_path):
        process_file_parallel(vocab, merges, special_tokens, val_input_path, val_output_path, "Validation")
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


def process_file_parallel(vocab, merges, special_tokens, input_path, output_path, name):
    """Tokenize a text file using multiple processes"""
    print(f"Processing {name} data: {input_path}")

    start_time = time.time()
    start_memory = get_memory_usage_gb()

    # Read file
    print(f"  Reading file...")
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    file_size_mb = sum(len(line.encode("utf-8")) for line in lines) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Total lines: {len(lines):,}")

    # Split lines into chunks for parallel processing
    num_workers = cpu_count()
    chunk_size = max(1, len(lines) // (num_workers * 10))  # More chunks for better progress tracking
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    print(f"  Using {num_workers} workers, {len(chunks)} chunks")
    print(f"  Tokenizing...")

    # Process in parallel with progress bar
    token_ids = []
    with Pool(num_workers, initializer=init_worker, initargs=(vocab, merges, special_tokens)) as pool:
        for result in tqdm(pool.imap(tokenize_chunk, chunks), total=len(chunks), desc=f"  {name}", unit="chunks"):
            token_ids.extend(result)

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
