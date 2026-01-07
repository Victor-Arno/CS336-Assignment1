"""BPE Trainer - Linear Search Method"""
from collections import Counter
from tqdm import tqdm

from .base import BPETokenizerBase


class BPETrainerLinear(BPETokenizerBase):
    """BPE Trainer using linear search for finding max pair"""

    def merge_tokens(
        self,
        tokens_counter: Counter[tuple[bytes]],
        match1: bytes,
        match2: bytes,
        pair_counts: dict[tuple[bytes, bytes], int]
    ) -> Counter[tuple[bytes]]:
        """Merge tokens: combine matching byte pairs into new token and update pair_counts"""
        merged_token = match1 + match2
        tokens_to_remove = []
        tokens_to_add = []

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

                for j in range(word_len - 1):
                    pair = (word[j], word[j + 1])
                    pair_counts[pair] -= freq
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]

                new_word_len = len(new_word_tuple)
                for j in range(new_word_len - 1):
                    pair = (new_word_tuple[j], new_word_tuple[j + 1])
                    pair_counts[pair] += freq

        for word in tokens_to_remove:
            del tokens_counter[word]
        for word, freq in tokens_to_add:
            tokens_counter[word] += freq

        return tokens_counter

    def train_BPE(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """Train BPE tokenizer using linear search"""
        next_index = self._initialize_vocab(special_tokens)
        tokens_counter = self._load_and_process_corpus(input_path, special_tokens)

        target_vocab_size = vocab_size
        pair_counts = self.count_pair_frequencies(tokens_counter)

        with tqdm(total=target_vocab_size - len(self.vocab), desc="Training BPE (Linear)") as pbar:
            while len(self.vocab) < target_vocab_size:
                if not pair_counts:
                    break

                # Linear search for max pair
                best_pair = self.find_max_pair(pair_counts)

                match1, match2 = best_pair
                tokens_counter = self.merge_tokens(tokens_counter, match1, match2, pair_counts)
                self.vocab[next_index] = match1 + match2
                self.merges.append((match1, match2))
                next_index += 1
                pbar.update(1)

        return self.vocab, self.merges
