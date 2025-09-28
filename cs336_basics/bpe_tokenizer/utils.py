from collections import Counter
import regex as re
from typing import List, Tuple


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

Pair = Tuple[bytes, bytes]


def count_tokens_shard(args):
    start, end, input_path, special_tokens = args
    token_cnt: Counter[str] = Counter()
    special_pattern = re.compile(
        "|".join(re.escape(t) for t in special_tokens)
    )
    PAT_pattern = re.compile(PAT)

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    parts = special_pattern.split(chunk)
    for p in parts:
        pre_tokens = PAT_pattern.finditer(p)
        for match in pre_tokens:
            # We should not count the special tokens
            token_cnt[match.group(0)] += 1

    return token_cnt


def apply_merged_pair(pair: Pair, seq: List[bytes]) -> List[bytes]:
    new_seq = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            new_seq.append(pair[0] + pair[1])
            i += 2
        else:
            new_seq.append(seq[i])
            i += 1
    return new_seq
