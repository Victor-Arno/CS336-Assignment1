from typing import Optional, overload, BinaryIO
from dataclasses import dataclass
from collections import defaultdict, Counter
import regex
import re
import os
import multiprocessing
from tqdm import tqdm
from functools import partial


# === 分块工具函数 ===
def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """找到文件的分块边界，基于特殊标记进行分割"""
    assert isinstance(split_special_token, bytes), "特殊标记必须表示为字节字符串"
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
    """获取文件分块"""
    chunks = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, b"\n")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # 规范化换行符，将所有换行符转换为\n
            chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
            chunks.append(chunk)
    return chunks

# === 分词器 ===
GPT2_TOKENIZER_REGEX = (
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)

@dataclass
class BPETokenizerParams:
    """BPE分词器参数"""
    vocab: dict[int, bytes]  # 词汇表：索引到字节的映射
    merges: list[tuple[bytes, bytes]]  # 合并规则列表

class BPETokenizer:
    """BPE分词器实现"""
    
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, params: BPETokenizerParams) -> None: ...

    def __init__(self, params: Optional[BPETokenizerParams] = None) -> None:
        """初始化BPE分词器"""
        if params:
            self.vocab = params.vocab
            self.merges = params.merges
        else:
            self.vocab = {}
            self.merges = []

    def process_text_with_pre_tokenize(self, text: str) -> Counter[tuple[bytes, ...]]:
        """
        使用GPT-2正则表达式进行预分词，将标记编码为UTF-8，并返回
        标记字节元组（例如，(b't', b'h', b'e')）及其频率的计数器
        """
        PAT = GPT2_TOKENIZER_REGEX
        tokens_counter = Counter()

        for match in regex.finditer(PAT, text):
            token = match.group()
            token_bytes = token.encode("utf-8")
            byte_tuple = tuple(bytes([b]) for b in token_bytes)  # 字节元组
            tokens_counter[byte_tuple] += 1

        return tokens_counter

    @staticmethod
    def _static_process_chunk(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
        """静态方法：处理文本分块，考虑特殊标记"""
        split_pattern = re.compile("|".join(re.escape(tok) for tok in special_tokens))
        chunks = split_pattern.split(text)  # 保留特殊标记
        tokenizer = BPETokenizer()
        tokens_counter = Counter()
        for chunk in chunks:
            chunk_counter = tokenizer.process_text_with_pre_tokenize(chunk)
            tokens_counter.update(chunk_counter)
        return tokens_counter
    
    def count_pair_frequencies(self, tokens_counter: Counter[tuple[bytes]]) -> dict[tuple[bytes, bytes], int]:
        """计算字节对频率"""
        counts = defaultdict(int)
        for word, freq in tokens_counter.items():
            # 优化：避免重复计算长度
            word_len = len(word)
            for i in range(word_len - 1):
                pair = (word[i], word[i + 1])
                counts[pair] += freq
        return counts

    def find_max_pair(self, counts: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
        """找到频率最高的字节对"""
        # 优化：直接在循环中比较，避免lambda函数开销
        max_freq = -1
        max_pair = None
        for pair, freq in counts.items():
            if freq > max_freq:
                max_freq = freq
                max_pair = pair
            elif freq == max_freq:
                # 修复：当频率相同时，选择字典序更大的pair（与原始代码一致）
                if pair > max_pair:
                    max_pair = pair
        return max_pair

    def merge_tokens(self, tokens_counter: Counter[tuple[bytes]], match1: bytes, match2: bytes, pair_counts: dict[tuple[bytes, bytes], int]) -> Counter[tuple[bytes]]:
        """合并标记：将匹配的字节对合并为新标记"""
        merged_token = match1 + match2
        tokens_to_remove = []  # 要移除的标记
        tokens_to_add = []     # 要添加的标记
        
        for word, freq in list(tokens_counter.items()):
            new_word = []
            word_len = len(word)
            i = 0
            has_merge = False
            
            # 遍历单词中的字节对，寻找匹配的合并对
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
                
                # 更新pair_counts：先减去旧词的贡献
                for i in range(word_len - 1):
                    pair = (word[i], word[i + 1])
                    pair_counts[pair] -= freq
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                
                # 更新pair_counts：再加上新词的贡献
                new_word_len = len(new_word_tuple)
                for i in range(new_word_len - 1):
                    pair = (new_word_tuple[i], new_word_tuple[i + 1])
                    pair_counts[pair] += freq
        
        # 更新tokens_counter
        for word in tokens_to_remove:
            del tokens_counter[word]
        for word, freq in tokens_to_add:
            tokens_counter[word] += freq
        
        return tokens_counter

    def train_BPE(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """训练BPE分词器"""
        self.vocab = {}
        self.merges = []

        # 首先用特殊标记初始化词汇表
        for i, token in enumerate(special_tokens):
            self.vocab[i] = token.encode("utf-8")

        offset = len(special_tokens)
        for i in range(256):
            self.vocab[offset + i] = bytes([i])

        next_index = offset + 256

        # 步骤1：获取文件分块
        num_processes = min(multiprocessing.cpu_count(), 8)  # 限制进程数，避免过度并行化
        chunks = get_chunk(input_path, num_processes)

        # 步骤2：并行处理分词
        partial_func = partial(BPETokenizer._static_process_chunk, special_tokens=special_tokens)

        # 优化：使用chunksize参数提高并行效率
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(pool.imap(partial_func, chunks, chunksize=max(1, len(chunks) // num_processes)))

        tokens_counter = Counter()
        for counter in results:
            tokens_counter.update(counter)

        # 优化：提前计算目标词汇量
        target_vocab_size = vocab_size
        current_vocab_size = len(self.vocab)
        
        # 初始化pair_counts，只计算一次
        pair_counts = self.count_pair_frequencies(tokens_counter)
        
        with tqdm(total=target_vocab_size - current_vocab_size, desc="训练BPE") as pbar:
            while len(self.vocab) < target_vocab_size:
                if not pair_counts:
                    break
                match1, match2 = self.find_max_pair(pair_counts)
                # 使用优化后的merge_tokens方法，传入pair_counts并让它更新
                tokens_counter = self.merge_tokens(tokens_counter, match1, match2, pair_counts)
                self.vocab[next_index] = match1 + match2
                self.merges.append((match1, match2))
                next_index += 1
                pbar.update(1)
        return self.vocab, self.merges

    def get_params(self) -> BPETokenizerParams:
        """获取分词器参数"""
        return BPETokenizerParams(vocab=self.vocab, merges=self.merges)



from typing import Iterable, Iterator
import re
import regex

class Tokenizer:
    """分词器类，用于编码和解码文本"""
    
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """初始化分词器"""
        self.byte_vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens or []

        # 构建从字节序列到标记ID的查找表
        self.byte_to_id = {v: k for k, v in self.byte_vocab.items()}

        # 将特殊标记编码为字节
        self.special_tokens_bytes = [s.encode("utf-8") for s in self.special_tokens]
        self.special_tokens_set = set(self.special_tokens_bytes)

        # 确保特殊标记在词汇表中
        for token in self.special_tokens_bytes:
            if token not in self.byte_to_id:
                new_id = len(self.byte_vocab)
                self.byte_vocab[new_id] = token
                self.byte_to_id[token] = new_id

        # 准备高效的合并操作
        self.merges = [(a, b) for a, b in merges]
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        """从文件加载分词器"""
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
        """预分词：使用GPT-2正则表达式分割文本"""
        PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        return regex.findall(PAT, text)

    def _byte_pair_merge(self, token: bytes) -> list[bytes]:
        """字节对合并：应用BPE合并规则"""
        # 将字节转换为单字节元素的元组
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
        """编码文本为标记ID列表"""
        result = []
        special_pattern = "|".join(re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True))
        split_pattern = re.compile(f"({special_pattern})") if special_pattern else None

        segments = re.split(split_pattern, text) if split_pattern else [text]

        for segment in segments:  # 移除tqdm进度条，提高性能
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
        """编码可迭代对象中的文本"""
        for line in iterable:
            yield from self.encode(line)

    def decode(self, ids: list[int]) -> str:
        """解码标记ID列表为文本"""
        byte_seq = b"".join(self.byte_vocab[i] for i in ids)
        return byte_seq.decode("utf-8", errors="replace")