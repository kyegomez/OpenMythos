"""HuggingFace tokenizer wrapper with vocab sizing + EOS-aware encoding."""

from typing import Optional

from transformers import AutoTokenizer

DEFAULT_MODEL_ID = "openai/gpt-oss-20b"

# Upper bound on a single document before we truncate. FineWeb-Edu has
# pathological outliers (single "documents" that are gigabytes of concatenated
# dumps); those stall DataLoader workers and occasionally OOM the tokenizer.
MAX_CHARS_PER_DOC = 4_000_000  # ~1M BPE tokens — well past any legitimate doc


class MythosTokenizer:
    """
    HuggingFace tokenizer wrapper for OpenMythos.

    Key behavior:
    - ``vocab_size`` returns ``len(self.tokenizer)`` (base vocab + added
      specials), NOT the base model's nominal vocab size. This matches the ID
      space reachable via ``encode``/``decode``, so ``nn.Embedding(vocab_size, ...)``
      sized from this property cannot index out of range.
    - Optionally rounds up to ``vocab_multiple_of`` (default 128) so embedding
      matrices align to tensor-core-friendly widths.
    - ``encode`` defends against None / non-str / huge inputs so a single bad
      sample does not crash the DataLoader worker.
    - ``encode_with_eos`` appends the EOS token id so the document packer can
      inject a boundary between concatenated docs, preventing the model from
      learning spurious cross-document attention.

    Args:
        model_id          -- HF model id or local tokenizer path.
        vocab_multiple_of -- round vocab_size up to this multiple (0 = off).

    Example:
        >>> tok = MythosTokenizer()
        >>> ids = tok.encode("Hello world")
        >>> text = tok.decode(ids)
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        vocab_multiple_of: int = 128,
    ):
        """
        Initialize the MythosTokenizer.

        Args:
            model_id          -- HF model id or path to a tokenizer directory.
            vocab_multiple_of -- if >0, round ``vocab_size`` up to this multiple
                                  for tensor-core-friendly embedding widths.
                                  128 matches the Llama / DeepSeek convention.
        """
        # Explicit trust_remote_code=False pins safe behavior across future
        # transformers upgrades even if defaults shift.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=False,
        )
        self.vocab_multiple_of = vocab_multiple_of

    @property
    def vocab_size(self) -> int:
        """
        Number of token IDs the model embedding must cover.

        Returns ``len(self.tokenizer)`` (base vocab + added special tokens),
        rounded up to ``vocab_multiple_of`` when enabled. ``tokenizer.vocab_size``
        on HF returns the base vocab and silently excludes added specials,
        which is a common source of CUDA ``device-side assert`` on long runs.
        """
        true_size = len(self.tokenizer)
        m = self.vocab_multiple_of
        if m and m > 1:
            return ((true_size + m - 1) // m) * m
        return true_size

    @property
    def eos_token_id(self) -> Optional[int]:
        """
        Token id used as a document boundary. Prefers ``eos_token_id``, falls
        back to ``bos_token_id``, then to the first defined special id, then
        to None. The trainer's doc packer injects this between concatenated
        samples so the model never sees a cross-document boundary without a
        marker at train time.
        """
        tid = self.tokenizer.eos_token_id
        if tid is not None:
            return int(tid)
        tid = self.tokenizer.bos_token_id
        if tid is not None:
            return int(tid)
        added = self.tokenizer.all_special_ids
        if added:
            return int(added[0])
        return None

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        max_chars: int = MAX_CHARS_PER_DOC,
    ) -> list[int]:
        """
        Encode text to a list of token ids.

        Rejects None / non-str silently (returns ``[]``) so a single malformed
        sample does not kill the DataLoader worker. Oversized documents are
        truncated at ``max_chars`` characters before tokenization.

        Args:
            text               -- source text; non-strings are treated as empty.
            add_special_tokens -- defaults to False for pretraining packing
                                  (the packer injects EOS explicitly).
            max_chars          -- hard character cap; pass a very large value
                                  if you genuinely have legitimate huge docs.

        Returns:
            List of integer token ids.
        """
        if not isinstance(text, str) or not text:
            return []
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def encode_with_eos(self, text: str) -> list[int]:
        """
        Encode ``text`` and append the EOS token id (when defined).

        Used by the document-packer path so concatenated documents are
        delimited by a boundary token rather than flowing into each other.
        Falls back to plain ``encode`` when the tokenizer has no EOS.
        """
        ids = self.encode(text)
        if not ids:
            return ids
        eos = self.eos_token_id
        if eos is not None:
            ids.append(eos)
        return ids

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode ``token_ids`` back to a string, stripping special tokens.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
