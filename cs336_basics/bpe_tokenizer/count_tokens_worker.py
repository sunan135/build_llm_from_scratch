from collections import Counter
import regex as re 

def _count_tokens_shard(args):
    start, end, input_path, special_tokens, PAT = args
    token_cnt: Counter[str] = Counter()
    special_pattern = re.compile("|".join(re.escape(t) for t in special_tokens))
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