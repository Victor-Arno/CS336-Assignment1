"""BPE Trainer - Heap Optimized Method with Inverted Index"""
from collections import Counter, defaultdict
import heapq
from tqdm import tqdm

from .base import BPETokenizerBase


class BPETrainerHeap(BPETokenizerBase):
    """BPE Trainer using heap optimization and pair_to_words inverted index"""

    def _build_initial_structures(
        self, tokens_counter: Counter[tuple[bytes, ...]]
    ) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]], list]:
        """
        Build initial pair_counts, pair_to_words index, and heap.

        Returns:
            pair_counts: pair -> frequency
            pair_to_words: pair -> set of words containing this pair
            heap: max heap for finding highest frequency pair
        """
        pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
        pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

        for word, freq in tokens_counter.items():
            word_len = len(word)
            for i in range(word_len - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += freq
                pair_to_words[pair].add(word)

        # Build heap
        heap = []
        for pair, freq in pair_counts.items():
            neg_pair = self._negate_pair_for_heap(pair)
            heap.append((-freq, neg_pair, pair))
        heapq.heapify(heap)

        return dict(pair_counts), pair_to_words, heap

    def _negate_pair_for_heap(self, pair: tuple[bytes, bytes]) -> tuple:
        """
        Convert pair to comparable form for heap ordering.
        For equal frequencies, we want lexicographically larger pairs first.
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
        return None

    def merge_tokens_with_index(
        self,
        tokens_counter: Counter[tuple[bytes, ...]],
        match1: bytes,
        match2: bytes,
        pair_counts: dict[tuple[bytes, bytes], int],
        pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
        heap: list
    ) -> None:
        """
        Merge tokens using inverted index - only process words containing the target pair.
        Updates tokens_counter, pair_counts, pair_to_words, and heap in-place.
        """
        merged_token = match1 + match2
        target_pair = (match1, match2)

        # Get words containing this pair (and remove from index)
        words_to_process = pair_to_words.pop(target_pair, set())

        # Remove pair from counts
        if target_pair in pair_counts:
            del pair_counts[target_pair]

        # Track pairs that need heap updates
        affected_pairs: set[tuple[bytes, bytes]] = set()

        for word in words_to_process:
            if word not in tokens_counter:
                continue

            freq = tokens_counter[word]
            word_len = len(word)

            # Build new word with merges
            new_word = []
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

            if not has_merge:
                continue

            new_word_tuple = tuple(new_word)

            # Update tokens_counter
            del tokens_counter[word]
            tokens_counter[new_word_tuple] += freq

            # Update pair_counts and pair_to_words for old word's pairs
            for j in range(word_len - 1):
                old_pair = (word[j], word[j + 1])
                if old_pair == target_pair:
                    continue  # Already handled

                # Decrease count
                if old_pair in pair_counts:
                    pair_counts[old_pair] -= freq
                    affected_pairs.add(old_pair)
                    if pair_counts[old_pair] <= 0:
                        del pair_counts[old_pair]
                        pair_to_words.pop(old_pair, None)
                    else:
                        # Remove old word from index
                        if old_pair in pair_to_words:
                            pair_to_words[old_pair].discard(word)

            # Update pair_counts and pair_to_words for new word's pairs
            new_word_len = len(new_word_tuple)
            for j in range(new_word_len - 1):
                new_pair = (new_word_tuple[j], new_word_tuple[j + 1])
                if new_pair not in pair_counts:
                    pair_counts[new_pair] = 0
                pair_counts[new_pair] += freq
                pair_to_words[new_pair].add(new_word_tuple)
                affected_pairs.add(new_pair)

        # Push affected pairs to heap
        for pair in affected_pairs:
            if pair in pair_counts and pair_counts[pair] > 0:
                self._push_to_heap(heap, pair, pair_counts[pair])

    def train_BPE(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """Train BPE tokenizer using heap optimization with inverted index"""
        next_index = self._initialize_vocab(special_tokens)
        tokens_counter = self._load_and_process_corpus(input_path, special_tokens)

        target_vocab_size = vocab_size

        # Build all structures at once
        pair_counts, pair_to_words, heap = self._build_initial_structures(tokens_counter)

        with tqdm(total=target_vocab_size - len(self.vocab), desc="Training BPE (Heap+Index)") as pbar:
            while len(self.vocab) < target_vocab_size:
                if not pair_counts:
                    break

                # Use heap to find max frequency pair
                best_pair = self._pop_max_pair(heap, pair_counts)
                if best_pair is None:
                    break

                match1, match2 = best_pair

                # Merge using inverted index
                self.merge_tokens_with_index(
                    tokens_counter, match1, match2,
                    pair_counts, pair_to_words, heap
                )

                self.vocab[next_index] = match1 + match2
                self.merges.append((match1, match2))
                next_index += 1
                pbar.update(1)

        return self.vocab, self.merges
