"""AttentionProtocol — structural type for swappable attention backends.

Stage 4.3 (P4.3) introduces this to replace hardcoded `if cfg.attn_type == "mla"`
dispatch. Any class implementing the __call__ signature and exposing the
`freqs_cis_dim` attribute satisfies the Protocol.

`freqs_cis_dim` is the number of complex frequency pairs the attention
layer needs from a RoPE buffer, i.e. rope_dim // 2. GQAttention uses
(head_dim // 2), MLAttention uses (qk_rope_head_dim // 2). Exposing it
lets the model precompute a single freqs_cis buffer sized correctly for
the chosen attention type, instead of precomputing both and discarding
one.
"""
from __future__ import annotations

from typing import Optional, Protocol, Union, runtime_checkable

import torch

from open_mythos.kv_cache import StaticKVCache


@runtime_checkable
class AttentionProtocol(Protocol):
    freqs_cis_dim: int

    def __call__(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Union[dict, StaticKVCache]] = None,
        cache_key: str = "default",
    ) -> torch.Tensor: ...
