"""Tokenizer tests — cover the documented contract of MythosTokenizer."""

import pytest

from open_mythos.tokenizer import MythosTokenizer


@pytest.fixture(scope="module")
def tokenizer() -> MythosTokenizer:
    return MythosTokenizer()


def test_loads(tokenizer: MythosTokenizer) -> None:
    assert tokenizer is not None


def test_vocab_size_positive(tokenizer: MythosTokenizer) -> None:
    assert tokenizer.vocab_size > 0


def test_vocab_size_covers_len(tokenizer: MythosTokenizer) -> None:
    """
    ``vocab_size`` must be >= ``len(tokenizer.tokenizer)``.

    This is the core invariant: an ``nn.Embedding(vocab_size, dim)`` sized
    from this property cannot index out of range for any token the
    tokenizer can emit, including added special tokens that HF's
    ``tokenizer.vocab_size`` silently excludes.
    """
    assert tokenizer.vocab_size >= len(tokenizer.tokenizer)


def test_vocab_size_rounded_to_multiple(tokenizer: MythosTokenizer) -> None:
    """Default ``vocab_multiple_of=128`` → vocab_size is a multiple of 128."""
    assert tokenizer.vocab_size % 128 == 0


def test_vocab_multiple_of_zero_disables_rounding() -> None:
    """Passing ``vocab_multiple_of=0`` returns the raw ``len(tokenizer)``."""
    tok = MythosTokenizer(vocab_multiple_of=0)
    assert tok.vocab_size == len(tok.tokenizer)


def test_encode_returns_list_of_ints(tokenizer: MythosTokenizer) -> None:
    ids = tokenizer.encode("Hello, world!")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0


def test_encode_ids_within_vocab(tokenizer: MythosTokenizer) -> None:
    """Every emitted id must be < vocab_size (otherwise embeddings crash)."""
    ids = tokenizer.encode("Any reasonable text with punctuation, 123.")
    vocab = tokenizer.vocab_size
    assert all(0 <= i < vocab for i in ids)


def test_encode_empty_string(tokenizer: MythosTokenizer) -> None:
    assert tokenizer.encode("") == []


def test_encode_none_returns_empty(tokenizer: MythosTokenizer) -> None:
    # Defense against a corrupt dataset sample killing the loader worker.
    assert tokenizer.encode(None) == []  # type: ignore[arg-type]


def test_encode_non_str_returns_empty(tokenizer: MythosTokenizer) -> None:
    assert tokenizer.encode(12345) == []  # type: ignore[arg-type]
    assert tokenizer.encode(["list"]) == []  # type: ignore[arg-type]


def test_encode_truncates_oversized_input(tokenizer: MythosTokenizer) -> None:
    """Hard character cap keeps the tokenizer from eating giant docs."""
    huge = "a" * 10_000_000
    ids = tokenizer.encode(huge, max_chars=1000)
    # 1000 chars of 'a' tokenize to something bounded — certainly much less
    # than if we had tokenized the full 10M-char input.
    assert len(ids) < 5000


def test_decode_returns_string(tokenizer: MythosTokenizer) -> None:
    ids = tokenizer.encode("Hello, world!")
    text = tokenizer.decode(ids)
    assert isinstance(text, str)


def test_roundtrip(tokenizer: MythosTokenizer) -> None:
    original = "The quick brown fox jumps over the lazy dog."
    ids = tokenizer.encode(original)
    recovered = tokenizer.decode(ids)
    assert original in recovered or recovered in original


def test_eos_token_id_is_int_or_none(tokenizer: MythosTokenizer) -> None:
    eos = tokenizer.eos_token_id
    assert eos is None or isinstance(eos, int)


def test_encode_with_eos_appends_eos_token(tokenizer: MythosTokenizer) -> None:
    """``encode_with_eos`` appends EOS when one is defined, else plain encode."""
    ids = tokenizer.encode_with_eos("Hello, world!")
    base = tokenizer.encode("Hello, world!")
    eos = tokenizer.eos_token_id
    if eos is not None:
        assert ids == base + [eos]
    else:
        assert ids == base


def test_encode_with_eos_on_empty_is_empty(tokenizer: MythosTokenizer) -> None:
    """No EOS appended on empty input — packer would otherwise emit a lone EOS."""
    assert tokenizer.encode_with_eos("") == []


if __name__ == "__main__":
    pytest.main([__file__, "--verbose", "-s"])
