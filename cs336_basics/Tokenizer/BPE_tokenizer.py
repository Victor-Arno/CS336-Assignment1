"""BPE Tokenizer Module

This module provides BPE tokenizer implementations with two training methods:
- Linear: Simple linear search for max pair (default, good for small-medium datasets)
- Heap: Heap-optimized search (better for large datasets with many unique pairs)

Usage:
    from cs336_basics.Tokenizer.BPE_tokenizer import BPETokenizer, BPETrainerLinear, BPETrainerHeap, Tokenizer

    # Using default (linear) trainer
    trainer = BPETokenizer()
    vocab, merges = trainer.train_BPE(input_path, vocab_size, special_tokens)

    # Using specific trainer
    trainer = BPETrainerLinear()  # or BPETrainerHeap()
    vocab, merges = trainer.train_BPE(input_path, vocab_size, special_tokens)

    # Encoding/Decoding
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    ids = tokenizer.encode("Hello world")
    text = tokenizer.decode(ids)
"""

# Import from base module
from .base import (
    BPETokenizerParams,
    BPETokenizerBase,
    Tokenizer,
    GPT2_TOKENIZER_REGEX,
    find_chunk_boundaries,
    get_chunk,
)

# Import trainers
from .trainer_linear import BPETrainerLinear
from .trainer_heap import BPETrainerHeap

# Default BPETokenizer uses linear method (for backward compatibility)
BPETokenizer = BPETrainerHeap

__all__ = [
    # Parameters
    "BPETokenizerParams",
    # Base class
    "BPETokenizerBase",
    # Trainers
    "BPETrainerLinear",
    "BPETrainerHeap",
    "BPETokenizer",  # Default alias for BPETrainerLinear
    # Encoder/Decoder
    "Tokenizer",
    # Utilities
    "GPT2_TOKENIZER_REGEX",
    "find_chunk_boundaries",
    "get_chunk",
]
