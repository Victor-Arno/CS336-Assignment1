import heapq

def build_pair_heap(self, pair_counts: dict) -> list:
    """构建最大堆（使用负频率实现）"""
    heap = []
    for pair, freq in pair_counts.items():
        heapq.heappush(heap, (-freq, pair))
    return heap

def find_max_pair_using_heap(self, heap: list) -> tuple[bytes, bytes]:
    """使用堆找到频率最高的词对"""
    if not heap:
        return None, None
    # 弹出频率最高的词对
    neg_freq, pair = heapq.heappop(heap)
    return pair, -neg_freq

def update_heap_after_merge(self, heap: list, pair_counts: dict, affected_pairs: set):
    """合并后更新堆"""
    # 重新构建受影响的词对在堆中的位置
    new_heap = []
    for neg_freq, pair in heap:
        if pair in pair_counts:
            heapq.heappush(new_heap, (-pair_counts[pair], pair))
    return new_heap


def train_BPE_optimized(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    self.vocab = {}
    self.merges = []

    # Initialize vocab with special tokens first
    for i, token in enumerate(special_tokens):
        self.vocab[i] = token.encode("utf-8")

    offset = len(special_tokens)
    for i in range(256):
        self.vocab[offset + i] = bytes([i])

    next_index = offset + 256

    # Step 1: Get file chunks
    num_processes = min(multiprocessing.cpu_count(), 4)  # 进一步减少进程数
    chunks = get_chunk(input_path, num_processes)

    # Step 2: Parallel processing
    partial_func = partial(BPETokenizer._static_process_chunk, special_tokens=special_tokens)
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(pool.imap(partial_func, chunks, chunksize=max(1, len(chunks) // num_processes)))

    tokens_counter = Counter()
    for counter in results:
        tokens_counter.update(counter)

    # 初始计算词对频率并构建堆
    pair_counts = self.count_pair_frequencies(tokens_counter)
    pair_heap = self.build_pair_heap(pair_counts)
    
    target_vocab_size = vocab_size
    current_vocab_size = len(self.vocab)
    
    with tqdm(total=target_vocab_size - current_vocab_size, desc="Training BPE (Optimized)") as pbar:
        while len(self.vocab) < target_vocab_size and pair_heap:
            # 使用堆找到最大频率词对
            match1, match2 = self.find_max_pair_using_heap(pair_heap)
            if match1 is None:
                break
                
            # 增量合并和更新
            tokens_counter, pair_counts = self.merge_tokens_with_incremental_update(
                tokens_counter, match1, match2, pair_counts)
            
            # 更新堆
            pair_heap = self.update_heap_after_merge(pair_heap, pair_counts, set())
            
            self.vocab[next_index] = match1 + match2
            self.merges.append((match1, match2))
            next_index += 1
            pbar.update(1)
    
    return self.vocab, self.merges