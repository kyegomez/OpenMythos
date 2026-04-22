DEFAULT_MODEL_ID = "openai/gpt-oss-20b"


class MythosTokenizer:
    """
    HuggingFace tokenizer wrapper for OpenMythos.

    The underlying transformers import is deferred into __init__ so that
    `import open_mythos` does not pay the transformers import cost unless the
    tokenizer is actually constructed.

    Args:
        model_id (str): The HuggingFace model ID or path to use with AutoTokenizer.
            Defaults to "openai/gpt-oss-20b".

    Attributes:
        tokenizer: An instance of HuggingFace's AutoTokenizer.

    Example:
        >>> tok = MythosTokenizer()
        >>> ids = tok.encode("Hello world")
        >>> s = tok.decode(ids)
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        """
        Initialize the MythosTokenizer.

        Args:
            model_id (str): HuggingFace model identifier or path to tokenizer files.
        """
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    @property
    def eos_token_id(self) -> int | None:
        """End-of-sequence token id, or None if the tokenizer does not define one."""
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        """Pad token id, or None if the tokenizer does not define one."""
        return self.tokenizer.pad_token_id

    @property
    def vocab_size(self) -> int:
        """
        Return the size of the tokenizer vocabulary.

        Returns:
            int: The number of unique tokens in the tokenizer vocabulary.
        """
        return self.tokenizer.vocab_size

    def encode(self, text: str) -> list[int]:
        """
        Encode input text into a list of token IDs.

        Args:
            text (str): The input text string to tokenize.

        Returns:
            list[int]: List of integer token IDs representing the input text.
        """
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a list of token IDs back into a text string.

        Args:
            token_ids (list[int]): A list of integer token IDs to decode.

        Returns:
            str: Decoded string representation of the token IDs.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
