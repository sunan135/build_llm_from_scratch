import os
from dataclasses import dataclass
from typing import BinaryIO
import regex as re 
import heapq
from collections import Counter
from typing import Set, List, Tuple

Pair = Tuple[bytes, bytes]

@dataclass(order=False)
class PairWithFreq:
    freq: int
    pair: Pair
    def __lt__(self, other):
        if self.freq == other.freq:
            return self.pair > other.pair # NOTE: This is a lexicographical comparison
        return self.freq > other.freq
    def __repr__(self):
        return f"({self.freq}, ({self.pair[0]}, {self.pair[1]}))"

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))



def train_bpe_algo(tokens_cnt: Counter[str], vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer based on the given tokens.
    
    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab = {i: bytes([i]) for i in range(256)}
    # NOTE: We add the special tokens to the vocabulary
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    sequences: List[List[bytes]] = []
    count: List[int] = []
    merged_pairs: List[Tuple[bytes, bytes]] = []
    pair_to_seq_id: dict[Pair, Set[int]] = {}  # key: pair, value: set of the sequences' indices that contain the pair
    pair_cnt: Counter[Pair] = Counter()  # key: pair, value: frequency
    seq_to_pair_cnt: dict[int, Counter[Pair]] = {}  # key: sequence index, value: counter of pairs in the sequence

    def count_pair_in_seq(seq: List[bytes]) -> Counter[Pair]:
        cnt = Counter()
        for i in range(len(seq) - 1):
            cnt[(seq[i], seq[i+1])] += 1
        return cnt
                
    idx = 0
    # Initialize the sequences and counts, and the pair_to_seq_id and pair_cnt
    for seq, cnt in tokens_cnt.items():
        b = seq.encode("utf-8") 
        sequences.append([bytes([bb]) for bb in b])
        count.append(cnt)
        current_pair_cnt = count_pair_in_seq(sequences[idx])
        seq_to_pair_cnt[idx] = current_pair_cnt
        for pair, pair_freq in current_pair_cnt.items():
            if pair not in pair_to_seq_id:
                pair_to_seq_id[pair] = set()
            pair_to_seq_id[pair].add(idx)
            pair_cnt[pair] += cnt * pair_freq
        idx += 1
    

    heap: List[PairWithFreq] = [PairWithFreq(c, p) for p, c in pair_cnt.items()]
    heapq.heapify(heap)
    def get_valid_most_frequent_pair(heap: List[PairWithFreq], pair_cnt: Counter[Pair]) -> Pair:
        while True and len(heap) > 0:
            pair_with_freq = heapq.heappop(heap)
            if pair_cnt[pair_with_freq.pair] == pair_with_freq.freq:
                # print(pair_with_freq)
                return pair_with_freq.pair
            
    def apply_merged_pair(pair: Pair, seq: List[bytes]) -> List[bytes]:
        new_seq = []
        i = 0
        while i< len(seq):
            if i+1 < len(seq) and seq[i] == pair[0] and seq[i+1] == pair[1]:
                new_seq.append(pair[0] + pair[1])
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        return new_seq

    while len(vocab) < vocab_size:
        cur_merge_pair = get_valid_most_frequent_pair(heap, pair_cnt)
        if cur_merge_pair is None:
            break
        vocab[len(vocab)] = cur_merge_pair[0] + cur_merge_pair[1]
        merged_pairs.append((cur_merge_pair[0], cur_merge_pair[1]))
        impacted_seq_ids = pair_to_seq_id[cur_merge_pair]
        # print("impacted_seq_ids:", impacted_seq_ids)
        for seq_id in impacted_seq_ids.copy():
            new_seq = apply_merged_pair(cur_merge_pair, sequences[seq_id])
            # print("new_seq:", b"".join(tok for tok in new_seq).decode("utf-8"))
            new_seq_pair_cnt = count_pair_in_seq(new_seq)
            for old_pair, old_cnt in seq_to_pair_cnt[seq_id].items():
                pair_cnt[old_pair] -= old_cnt * count[seq_id]
                heapq.heappush(heap, PairWithFreq(pair_cnt[old_pair], old_pair))
                pair_to_seq_id[old_pair].remove(seq_id)
            for new_pair, new_cnt in new_seq_pair_cnt.items():
                pair_cnt[new_pair] += new_cnt * count[seq_id]
                heapq.heappush(heap, PairWithFreq(pair_cnt[new_pair], new_pair))
                if new_pair not in pair_to_seq_id:
                    pair_to_seq_id[new_pair] = set()
                pair_to_seq_id[new_pair].add(seq_id)
            seq_to_pair_cnt[seq_id] = new_seq_pair_cnt
            sequences[seq_id] = new_seq

        # print(b"".join(tok for seq in sequences for tok in seq).decode("utf-8"))
    return vocab, merged_pairs


def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        token_cnt: Counter[str] = Counter()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            special_pattern = re.compile("|".join(re.escape(t) for t in special_tokens))
            parts = special_pattern.split(chunk)
            seps = special_pattern.findall(chunk)
            for p, s in zip(parts, seps + [""]):
                 # Run pre-tokenization on your chunk and store the counts for each pre-token
                pre_tokens = re.finditer(PAT, p)
                for match in pre_tokens:
                    # We should not count the special tokens
                    token_cnt[match.group(0)] += 1

        
        return train_bpe_algo(token_cnt, vocab_size, special_tokens)

           
def train_bpe_with_str(input_str: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    token_cnt: Counter[str] = Counter()
    special_pattern = re.compile("|".join(re.escape(t) for t in special_tokens))
    parts = special_pattern.split(input_str)
    seps = special_pattern.findall(input_str)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    for p, s in zip(parts, seps + [""]):
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        pre_tokens = re.finditer(PAT, p)
        for match in pre_tokens:
            token_cnt[match.group(0)] += 1

        token_cnt[s] += 1
    # print(token_cnt)
    return train_bpe_algo(token_cnt, vocab_size)

# vocab, merges = train_bpe_with_str("Once upon a time there was a little boy named Ben. Ben loved to explore the world around him. He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben was walking through the store when he came across a very special vase.", 300, ["<|endoftext|>"])
# vocab, merges = train_bpe_algo({" lower": 1, " low": 5, " lowest": 1}, 259)
# print(vocab)
# print(merges)
# print(os.getcwd())
# vocab, merges = train_bpe("tests/fixtures/corpus.en", 500, ["<|endoftext|>"])
# print(vocab)
# print(merges)