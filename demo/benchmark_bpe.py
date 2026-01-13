"""BPE Training Performance Comparison: Linear vs Heap Optimization"""
import sys
import time
from pathlib import Path

# Add parent directory to path to import cs336_basics
sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.Tokenizer.BPE_tokenizer import BPETrainerLinear, BPETrainerHeap

FIXTURES_PATH = Path(__file__).parent.parent / "tests" / "fixtures"


def benchmark():
    # Test files
    test_files = [
        ("corpus.en (small)", FIXTURES_PATH / "corpus.en", 500),
        ("tinystories_sample_5M (medium)", FIXTURES_PATH / "tinystories_sample_5M.txt", 1000),
    ]

    special_tokens = ["<|endoftext|>"]

    for name, input_path, vocab_size in test_files:
        if not input_path.exists():
            print(f"Skipping {name}: file does not exist")
            continue

        print(f"\n{'='*60}")
        print(f"Test: {name}")
        print(f"Vocab size: {vocab_size}")
        print(f"{'='*60}")

        # Test linear version
        trainer_linear = BPETrainerLinear()
        start = time.time()
        vocab_linear, merges_linear = trainer_linear.train_BPE(
            input_path=str(input_path),
            vocab_size=vocab_size,
            special_tokens=special_tokens
        )
        linear_time = time.time() - start
        print(f"\nLinear method: {linear_time:.3f}s")

        # Test heap version
        trainer_heap = BPETrainerHeap()
        start = time.time()
        vocab_heap, merges_heap = trainer_heap.train_BPE(
            input_path=str(input_path),
            vocab_size=vocab_size,
            special_tokens=special_tokens
        )
        heap_time = time.time() - start
        print(f"Heap method:   {heap_time:.3f}s")

        # Comparison
        if linear_time > 0 and heap_time > 0:
            if heap_time < linear_time:
                ratio = linear_time / heap_time
                print(f"\nHeap is {ratio:.2f}x faster than Linear")
            else:
                ratio = heap_time / linear_time
                print(f"\nHeap is {ratio:.2f}x slower than Linear")

        # Verify result consistency
        if merges_linear == merges_heap:
            print("✓ Results match")
        else:
            print("✗ Results do not match!")
            # Find first difference
            for i, (l, h) in enumerate(zip(merges_linear, merges_heap)):
                if l != h:
                    print(f"  First difference at merge {i}: linear={l}, heap={h}")
                    break


if __name__ == "__main__":
    benchmark()
