"""BPE Trainer - Heap Optimized Method"""
from collections import Counter
import heapq
from tqdm import tqdm

from .base import BPETokenizerBase


class BPETrainerHeap(BPETokenizerBase):
    """BPE Trainer using heap optimization for finding max pair"""

    def _build_heap(self, pair_counts: dict[tuple[bytes, bytes], int]) -> list:
        """Build max heap from pair_counts (using negative values)"""
        # Heap element format: (-freq, neg_pair, pair)
        # neg_pair is used for tie-breaking by lexicographic order
        heap = []
        for pair, freq in pair_counts.items():
            neg_pair = self._negate_pair_for_heap(pair)
            heap.append((-freq, neg_pair, pair))
        heapq.heapify(heap)
        return heap

    def _negate_pair_for_heap(self, pair: tuple[bytes, bytes]) -> tuple:
        """
        Convert pair to comparable form for heap ordering.

        For equal frequencies, we want lexicographically larger pairs first.
        We negate each byte value and add a terminator to handle prefix comparisons.
        """
        def negate_bytes(b: bytes) -> tuple:
            return tuple(-x for x in b) + (1,)

        return (negate_bytes(pair[0]), negate_bytes(pair[1]))

    def _push_to_heap(self, heap: list, pair: tuple[bytes, bytes], freq: int):
        """Push new pair to heap"""
        if freq > 0:
            neg_pair = self._negate_pair_for_heap(pair)
            heapq.heappush(heap, (-freq, neg_pair, pair))

    def _pop_max_pair(self, heap: list, pair_counts: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes] | None:
        """Pop highest frequency valid pair from heap (lazy deletion)"""
        while heap:
            neg_freq, neg_pair, pair = heapq.heappop(heap)
            freq = -neg_freq
            # Check if pair is still valid (frequency matches)
            if pair in pair_counts and pair_counts[pair] == freq:
                return pair
            # If frequency doesn't match or pair deleted, continue popping
        return None

    def merge_tokens_with_heap(
        self,
        tokens_counter: Counter[tuple[bytes]],
        match1: bytes,
        match2: bytes,
        pair_counts: dict[tuple[bytes, bytes], int],
        heap: list
    ) -> Counter[tuple[bytes]]:
        """Merge tokens and update heap"""
        merged_token = match1 + match2
        tokens_to_remove = []
        tokens_to_add = []

        # Track all affected pairs and their changes
        affected_pairs = set()

        for word, freq in list(tokens_counter.items()):
            new_word = []
            word_len = len(word)
            i = 0
            has_merge = False

            while i < word_len:
                if i < word_len - 1 and word[i] == match1 and word[i + 1] == match2:
                    new_word.append(merged_token)
                    i += 2
                    has_merge = True
                else:
                    new_word.append(word[i])
                    i += 1

            if has_merge:
                new_word_tuple = tuple(new_word)
                tokens_to_remove.append(word)
                tokens_to_add.append((new_word_tuple, freq))

                # Update pair_counts: subtract old word's contribution
                for j in range(word_len - 1):
                    pair = (word[j], word[j + 1])
                    affected_pairs.add(pair)
                    pair_counts[pair] -= freq
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]

                # Update pair_counts: add new word's contribution
                new_word_len = len(new_word_tuple)
                for j in range(new_word_len - 1):
                    pair = (new_word_tuple[j], new_word_tuple[j + 1])
                    affected_pairs.add(pair)
                    pair_counts[pair] += freq

        # Update tokens_counter
        for word in tokens_to_remove:
            del tokens_counter[word]
        for word, freq in tokens_to_add:
            tokens_counter[word] += freq

        # Push all affected pairs back to heap (if still exist)
        for pair in affected_pairs:
            if pair in pair_counts:
                self._push_to_heap(heap, pair, pair_counts[pair])

        return tokens_counter

    def train_BPE(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """Train BPE tokenizer using heap optimization (for large-scale data)"""
        next_index = self._initialize_vocab(special_tokens)
        tokens_counter = self._load_and_process_corpus(input_path, special_tokens)

        target_vocab_size = vocab_size
        pair_counts = self.count_pair_frequencies(tokens_counter)
        heap = self._build_heap(pair_counts)

        with tqdm(total=target_vocab_size - len(self.vocab), desc="Training BPE (Heap)") as pbar:
            while len(self.vocab) < target_vocab_size:
                if not pair_counts:
                    break

                # Use heap to find max frequency pair
                best_pair = self._pop_max_pair(heap, pair_counts)
                if best_pair is None:
                    break

                match1, match2 = best_pair
                # Merge and update heap
                tokens_counter = self.merge_tokens_with_heap(
                    tokens_counter, match1, match2, pair_counts, heap
                )
                self.vocab[next_index] = match1 + match2
                self.merges.append((match1, match2))
                next_index += 1
                pbar.update(1)

        return self.vocab, self.merges
