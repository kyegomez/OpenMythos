"""
Static pre-allocated KV cache for inference.

The dict-of-growing-tensors approach in early OpenMythos reallocates the KV
cache tensor on every decode step via `torch.cat`, which copies the entire
prefix each time (O(S^2) total writes over S decode steps). This module
implements a pre-allocated buffer that:

- Allocates once, on first write per cache key, at size `max_seq_len`.
- Writes new chunks at the current position and advances an internal counter.
- Returns the cumulative prefix as a cheap view (no copy).

The cache is a drop-in replacement for the legacy dict-based cache in
attention modules: both GQAttention.forward and MLAttention.forward accept
either a plain dict (legacy path, preserved for backward compatibility) or
a StaticKVCache instance (fast path). The public API of OpenMythos.generate
uses StaticKVCache by default.
"""

from __future__ import annotations

from typing import Optional

import torch


class StaticKVCache:
    """Pre-allocated KV cache keyed by string.

    Each unique key maps to an internal buffer of shape `(B, max_seq_len, *rest)`
    and a position counter. `write_and_read(key, tensor)` writes the incoming
    chunk at the current position for that key and returns the cumulative
    prefix up to and including this write.

    The attention modules use composite keys of the form `"<cache_key>:k"`,
    `"<cache_key>:v"` (GQA) or `"<cache_key>:c_kv"`, `"<cache_key>:k_rope"`
    (MLA) so that k and v advance independently; `__contains__` matches on
    the `cache_key` prefix so legacy `"layer0" in cache` checks still work.
    """

    def __init__(self, max_seq_len: int):
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        self.max_seq_len = max_seq_len
        self._buffers: dict[str, torch.Tensor] = {}
        self._positions: dict[str, int] = {}

    def write_and_read(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        """Append `tensor` at the current position for `key`; return the full prefix.

        Lazily allocates the buffer on first call using the tensor's shape,
        dtype, and device. Subsequent calls for the same key must use a
        tensor with the same batch size and trailing dimensions.
        """
        if key not in self._buffers:
            B = tensor.shape[0]
            rest = tensor.shape[2:]
            self._buffers[key] = torch.zeros(
                B, self.max_seq_len, *rest,
                dtype=tensor.dtype, device=tensor.device,
            )
            self._positions[key] = 0

        buf = self._buffers[key]
        pos = self._positions[key]
        T = tensor.shape[1]
        end = pos + T
        if end > self.max_seq_len:
            raise RuntimeError(
                f"StaticKVCache overflow for key {key!r}: "
                f"pos {pos} + T {T} > max_seq_len {self.max_seq_len}"
            )
        buf[:, pos:end] = tensor.detach()
        self._positions[key] = end
        return buf[:, :end]

    def current_length(self, key: str) -> int:
        """Number of tokens written for `key`, or 0 if unused."""
        return self._positions.get(key, 0)

    def reset(self) -> None:
        """Reset all position counters to 0 without releasing buffers."""
        for k in list(self._positions):
            self._positions[k] = 0

    def __contains__(self, cache_key: str) -> bool:
        # Matches bare `cache_key` or any `cache_key:<subkey>` form so tests
        # written against the legacy dict API (`"layer0" in cache`) behave
        # transparently.
        return any(
            k == cache_key or k.startswith(f"{cache_key}:")
            for k in self._buffers
        )

    def buffer_id(self, key: str) -> Optional[int]:
        """`id()` of the underlying buffer for diagnostic use in tests."""
        buf = self._buffers.get(key)
        return None if buf is None else id(buf)
