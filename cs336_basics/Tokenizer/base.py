"""BPE Tokenizer Base Module - Shared utilities and classes"""
from typing import Optional, overload, BinaryIO, Iterable, Iterator
from dataclasses import dataclass
from collections import defaultdict, Counter
import regex
import re
import os
import multiprocessing
from functools import partial


# === GPT-2 Pre-tokenization Regex ===
GPT2_TOKENIZER_REGEX = (
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


# === Chunk Utilities ===
def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """Find file chunk boundaries based on special token splits"""
    assert isinstance(split_special_token, bytes), "Special token must be bytes"
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def get_chunk(input_path: str, desired_num_chunks: int) -> list[str]:
    """Get file chunks"""
    chunks = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, b"\n")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Normalize newlines
            chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
            chunks.append(chunk)
    return chunks


# === BPE Tokenizer Parameters ===
@dataclass
class BPETokenizerParams:
    """BPE tokenizer parameters"""
    vocab: dict[int, bytes]  # Vocabulary: index to bytes mapping
    merges: list[tuple[bytes, bytes]]  # Merge rules list


# === Base BPE Tokenizer ===
class BPETokenizerBase:
    """Base BPE Tokenizer with shared functionality"""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, params: BPETokenizerParams) -> None: ...

    def __init__(self, params: Optional[BPETokenizerParams] = None) -> None:
        """Initialize BPE tokenizer"""
        if params:
            self.vocab = params.vocab
            self.merges = params.merges
        else:
            self.vocab = {}
            self.merges = []

    def process_text_with_pre_tokenize(self, text: str) -> Counter[tuple[bytes, ...]]:
        """
        Pre-tokenize using GPT-2 regex, encode to UTF-8, and return
        Counter of token byte tuples (e.g., (b't', b'h', b'e')) with frequencies
        """
        PAT = GPT2_TOKENIZER_REGEX
        tokens_counter = Counter()

        for match in regex.finditer(PAT, text):
            token = match.group()
            token_bytes = token.encode("utf-8")
            byte_tuple = tuple(bytes([b]) for b in token_bytes)
            tokens_counter[byte_tuple] += 1

        return tokens_counter

    @staticmethod
    def _static_process_chunk(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
        """Static method: process text chunk considering special tokens"""
        split_pattern = re.compile("|".join(re.escape(tok) for tok in special_tokens))
        chunks = split_pattern.split(text)
        tokenizer = BPETokenizerBase()
        tokens_counter = Counter()
        for chunk in chunks:
            chunk_counter = tokenizer.process_text_with_pre_tokenize(chunk)
            tokens_counter.update(chunk_counter)
        return tokens_counter

    def count_pair_frequencies(self, tokens_counter: Counter[tuple[bytes]]) -> dict[tuple[bytes, bytes], int]:
        """Count byte pair frequencies"""
        counts = defaultdict(int)
        for word, freq in tokens_counter.items():
            word_len = len(word)
            for i in range(word_len - 1):
                pair = (word[i], word[i + 1])
                counts[pair] += freq
        return counts

    def find_max_pair(self, counts: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
        """Find byte pair with highest frequency, break ties by lexicographic order"""
        max_freq = -1
        max_pair = None
        for pair, freq in counts.items():
            if freq > max_freq:
                max_freq = freq
                max_pair = pair
            elif freq == max_freq:
                if pair > max_pair:
                    max_pair = pair
        return max_pair

    def get_params(self) -> BPETokenizerParams:
        """Get tokenizer parameters"""
        return BPETokenizerParams(vocab=self.vocab, merges=self.merges)

    def _initialize_vocab(self, special_tokens: list[str]) -> int:
        """Initialize vocabulary with special tokens and byte values"""
        self.vocab = {}
        self.merges = []

        # First add special tokens
        for i, token in enumerate(special_tokens):
            self.vocab[i] = token.encode("utf-8")

        offset = len(special_tokens)
        for i in range(256):
            self.vocab[offset + i] = bytes([i])

        return offset + 256

    def _load_and_process_corpus(self, input_path: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
        """Load corpus and process with parallel tokenization"""
        num_processes = min(multiprocessing.cpu_count(), 8)
        chunks = get_chunk(input_path, num_processes)

        partial_func = partial(BPETokenizerBase._static_process_chunk, special_tokens=special_tokens)

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(pool.imap(partial_func, chunks, chunksize=max(1, len(chunks) // num_processes)))

        tokens_counter = Counter()
        for counter in results:
            tokens_counter.update(counter)

        return tokens_counter


# === Tokenizer (Encoder/Decoder) ===
class Tokenizer:
    """Tokenizer class for encoding and decoding text"""

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """Initialize tokenizer"""
        self.byte_vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Build byte sequence to token ID lookup
        self.byte_to_id = {v: k for k, v in self.byte_vocab.items()}

        # Encode special tokens to bytes
        self.special_tokens_bytes = [s.encode("utf-8") for s in self.special_tokens]
        self.special_tokens_set = set(self.special_tokens_bytes)

        # Ensure special tokens are in vocabulary
        for token in self.special_tokens_bytes:
            if token not in self.byte_to_id:
                new_id = len(self.byte_vocab)
                self.byte_vocab[new_id] = token
                self.byte_to_id[token] = new_id

        # Prepare efficient merge operations
        self.merges = [(a, b) for a, b in merges]
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        """Load tokenizer from files"""
        import json

        with open(vocab_filepath, "r") as vf:
            vocab_data = json.load(vf)
            vocab = {int(i): bytes(v, "latin1") for v, i in vocab_data.items()}

        merges = []
        with open(merges_filepath, "r") as mf:
            for line in mf:
                if line.strip() and not line.startswith("#"):
                    parts = line.strip().split()
                    if len(parts) == 2:
                        merges.append((bytes(parts[0], "latin1"), bytes(parts[1], "latin1")))

        return cls(vocab, merges, special_tokens)

    def _pre_tokenize(self, text: str) -> list[str]:
        """Pre-tokenize using GPT-2 regex"""
        PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        return regex.findall(PAT, text)

    def _byte_pair_merge(self, token: bytes) -> list[bytes]:
        """Apply BPE merge rules"""
        word = [bytes([b]) for b in token]
        pairs = lambda w: set((w[i], w[i + 1]) for i in range(len(w) - 1))

        while True:
            candidate_pairs = pairs(word)
            ranked_pairs = [(self.merge_ranks[p], p) for p in candidate_pairs if p in self.merge_ranks]
            if not ranked_pairs:
                break

            _, best_pair = min(ranked_pairs)
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word

    def encode(self, text: str) -> list[int]:
        """Encode text to token ID list"""
        result = []
        special_pattern = "|".join(re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True))
        split_pattern = re.compile(f"({special_pattern})") if special_pattern else None

        segments = re.split(split_pattern, text) if split_pattern else [text]

        for segment in segments:
            if segment == "":
                continue
            b = segment.encode("utf-8")
            if b in self.special_tokens_set:
                result.append(self.byte_to_id[b])
            else:
                for token in self._pre_tokenize(segment):
                    for merged in self._byte_pair_merge(token.encode("utf-8")):
                        result.append(self.byte_to_id[merged])
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode text from iterable"""
        for line in iterable:
            yield from self.encode(line)

    def decode(self, ids: list[int]) -> str:
        """Decode token ID list to text"""
        byte_seq = b"".join(self.byte_vocab[i] for i in ids)
        return byte_seq.decode("utf-8", errors="replace")
