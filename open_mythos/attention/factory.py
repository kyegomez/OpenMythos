"""Attention factory. Stage 4.3 (P4.3).

`build_attention(cfg)` replaces the pre-4.3 `if cfg.attn_type == "mla"`
ternary that had been inlined at both TransformerBlock.__init__ and
OpenMythos.__init__ call sites. Adding a new attention backend now only
requires extending this factory — TransformerBlock and the model stay
closed for modification.
"""
from __future__ import annotations

from open_mythos.attention.gqa import GQAttention
from open_mythos.attention.mla import MLAttention
from open_mythos.attention.protocol import AttentionProtocol
from open_mythos.config import MythosConfig


def build_attention(cfg: MythosConfig) -> AttentionProtocol:
    """Return an attention module matching cfg.attention.attn_type."""
    attn_type = cfg.attention.attn_type
    if attn_type == "mla":
        return MLAttention(cfg)
    if attn_type == "gqa":
        return GQAttention(cfg)
    # MythosConfig.__post_init__ already rejects unknown attn_type, but we
    # defend the factory anyway for direct callers bypassing construction.
    raise ValueError(
        f"build_attention: unknown attn_type {attn_type!r}; "
        "expected 'gqa' or 'mla'"
    )
