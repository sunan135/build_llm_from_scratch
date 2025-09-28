class BPETokenizer:
    """
    A BPE tokenizer that uses the provided vocab, merges, and special tokens to tokenize and detokenize text.
    """

def __init__(
    self,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
):
    self.vocab = vocab
    self.merges = merges
    self.special_tokens = special_tokens


def encode(self, text: str) -> list[int]:
    pass


def decode(self, ids: list[int]) -> str:
    pass


def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    """
    Given an iterable of strings (i.e., a file handle), yield a generator taht lazily yields token IDs. This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
    """
    pass

def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
    """
    Class method that constructs and returns a BPE tokenizer from a serialized vocabulary and list of merged and (optionally) a list of special tokens. 
    """
    pass