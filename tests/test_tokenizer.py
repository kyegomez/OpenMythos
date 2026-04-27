import pytest
from open_mythos.tokenizer import (
    MythosTokenizer,
    get_vocab_size,
    load_tokenizer,
)


@pytest.fixture(scope="module")
def tokenizer():
    tok = MythosTokenizer()
    print(f"\nLoaded tokenizer: {tok.tokenizer.name_or_path}")
    return tok


def test_loads(tokenizer):
    assert tokenizer is not None
    print(f"Tokenizer: {tokenizer}")


def test_vocab_size(tokenizer):
    size = tokenizer.vocab_size
    print(f"Vocab size: {size:,}")
    assert size > 0


def test_encode_returns_list_of_ints(tokenizer):
    ids = tokenizer.encode("Hello, world!")
    print(f"encode('Hello, world!') → {ids}")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0


def test_encode_empty_string(tokenizer):
    ids = tokenizer.encode("")
    print(f"encode('') → {ids}")
    assert isinstance(ids, list)


def test_decode_returns_string(tokenizer):
    ids = tokenizer.encode("Hello, world!")
    text = tokenizer.decode(ids)
    print(f"decode({ids}) → '{text}'")
    assert isinstance(text, str)


def test_roundtrip(tokenizer):
    original = "The quick brown fox jumps over the lazy dog."
    ids = tokenizer.encode(original)
    recovered = tokenizer.decode(ids)
    print(f"original:  '{original}'")
    print(f"token ids: {ids}")
    print(f"recovered: '{recovered}'")
    assert original in recovered or recovered in original


def test_encode_long_text(tokenizer):
    text = "OpenMythos is a recurrent depth transformer. " * 100
    ids = tokenizer.encode(text)
    print(f"Long text ({len(text)} chars) → {len(ids)} tokens")
    assert len(ids) > 100


def test_custom_model_id():
    tok = MythosTokenizer(model_id="openai/gpt-oss-20b")
    print(f"Custom model_id vocab size: {tok.vocab_size:,}")
    assert tok.vocab_size > 0


def test_load_tokenizer_helper():
    tok = load_tokenizer()
    assert isinstance(tok, MythosTokenizer)


def test_get_vocab_size_helper(tokenizer):
    assert get_vocab_size() == tokenizer.vocab_size


def test_vocab_size_consistent(tokenizer):
    outer = tokenizer.vocab_size
    inner = tokenizer.tokenizer.vocab_size
    print(f"vocab_size property: {outer:,}  |  inner tokenizer.vocab_size: {inner:,}")
    assert outer == inner


if __name__ == "__main__":
    pytest.main([__file__, "--verbose", "-s"])
