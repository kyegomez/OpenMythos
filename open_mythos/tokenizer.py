from transformers import AutoTokenizer

DEFAULT_MODEL_ID = "openai/gpt-oss-20b"


class MythosTokenizer:
    """
    HuggingFace tokenizer wrapper for OpenMythos.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def encode(self, text: str):
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    # ✅ New methods added
    def token_count(self, text: str) -> int:
        """Return number of tokens in text."""
        return len(self.tokenizer.encode(text))

    def batch_encode(self, texts: list[str], padding: bool = True, truncation: bool = True):
        """Encode multiple texts at once."""
        return self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )

    def get_special_tokens(self):
        """Return special tokens used by tokenizer."""
        return self.tokenizer.special_tokens_map

    def is_within_limit(self, text: str, max_tokens: int) -> bool:
        """Check if text fits within a token limit."""
        return self.token_count(text) <= max_tokens        Returns:
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
