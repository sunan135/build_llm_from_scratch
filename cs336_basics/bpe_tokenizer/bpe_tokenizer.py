from cs336_basics.bpe_tokenizer.utils import (
    apply_merged_pair,
    PAT,
)
import regex as re
from typing import Iterable, Iterator


class BPETokenizer:
    """
    A BPE tokenizer that uses the provided vocab, merges, and special tokens
    to tokenize and detokenize text.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        # self.special_tokens = special_tokens
        self.vocab_reverse: dict[bytes, int] = {
            tok: idx for idx, tok in self.vocab.items()
        }
        if special_tokens:
            # Sort the special tokens by length prefer longer special tokens to
            # not split the longer special tokens that would lead to semantic
            # distinction and inefficiency
            self.special_pattern = re.compile(
                "|".join(
                    re.escape(t)
                    for t in sorted(special_tokens, key=len, reverse=True)
                )
            )
            self.special_tokens_vocab_ids = {
                self.vocab_reverse[special.encode("utf-8")]: special
                for special in special_tokens
            }
            print("special_tokens_vocab_ids: ")
            print(self.special_tokens_vocab_ids)
        else:
            self.special_pattern = None
            self.special_tokens_vocab_ids = {}

    def encode(self, text: str) -> list[int]:
        # Split by special tokens if we have a pattern
        if self.special_pattern:
            parts = self.special_pattern.split(text)
            specials = self.special_pattern.findall(text)
        else:
            parts = [text]
            specials = []

        encoded_text: list[int] = []

        for p, special in zip(parts, specials + [""]):
            # pre tokenize the text
            pre_tokens = re.finditer(PAT, p)
            for match in pre_tokens:
                print(match.group(0))
                merged_seq_bytes = self._apply_merges(match.group(0))
                print(merged_seq_bytes)
                for b in merged_seq_bytes:
                    encoded_text.append(self.vocab_reverse[b])
            # append special tokens
            if special:
                print("special: " + special)
                print(self.vocab_reverse[special.encode("utf-8")])
                encoded_text.append(
                    self.vocab_reverse[special.encode("utf-8")]
                )
        print(encoded_text)
        return encoded_text

    def decode(self, ids: list[int]) -> str:
        decoded_str_bytes: list[bytes] = []
        decoded_strs: list[str] = []
        for token_id in ids:
            print(self.vocab[token_id])
            print(token_id)
            if token_id in self.special_tokens_vocab_ids:
                print("special token id: " + str(token_id))
                decoded_strs.append(
                    b"".join(decoded_str_bytes).decode("utf-8")
                )
                print(decoded_strs)
                decoded_strs.append(self.special_tokens_vocab_ids[token_id])
                print(decoded_strs)
                decoded_str_bytes = []
            else:
                decoded_str_bytes.append(self.vocab[token_id])

        print("decoded_str_bytes: " + str(decoded_str_bytes))
        if len(decoded_str_bytes) > 0:
            last_str_bytes = b"".join(decoded_str_bytes)
            try:
                last_str = last_str_bytes.decode("utf-8")
                decoded_strs.append(last_str)
            except UnicodeDecodeError:
                decoded_strs.append("\ufffd")

        return "".join(decoded_strs)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (i.e., a file handle), yield a generator
        that lazily yields token IDs.
        This is required for memory-efficienttokenization of large files that
        we cannot directly load into memory.
        """
        pass

    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """
        Class method that constructs and returns a BPE tokenizer from a serialized vocabulary and list of merged
        and (optionally) a list of special tokens.
        """
        pass

    def _apply_merges(self, seq: str) -> list[bytes]:
        str_bytes = seq.encode("UTF-8")
        seq_bytes = [str_bytes[i : i + 1] for i in range(len(str_bytes))]
        for [l, r] in self.merges:
            seq_bytes = apply_merged_pair((l, r), seq_bytes)
        return seq_bytes
