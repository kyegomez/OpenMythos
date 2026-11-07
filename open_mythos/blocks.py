from typing import Optional, Union

import torch
import torch.nn as nn

from open_mythos.attention.factory import build_attention
from open_mythos.config import MythosConfig
from open_mythos.kv_cache import StaticKVCache
from open_mythos.moe import MoEFFN
from open_mythos.primitives.expert import Expert
from open_mythos.primitives.norm import RMSNorm


# ---------------------------------------------------------------------------
# Single Transformer Block (shared across recurrent loops)
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """
    Standard pre-norm transformer block with swappable attention and optional MoE FFN.

    Attention is selected by cfg.attn_type:
        "gqa" → GQAttention  (Grouped Query Attention, fewer KV heads)
        "mla" → MLAttention  (Multi-Latent Attention, compressed KV cache)

    FFN is selected by use_moe:
        True  → MoEFFN  (fine-grained routed + shared experts; used in RecurrentBlock)
        False → Expert  (dense SwiGLU FFN; used in Prelude and Coda)
    """

    def __init__(self, cfg: MythosConfig, use_moe: bool = False):
        """
        Args:
            cfg     -- MythosConfig; attn_type selects the attention class
            use_moe -- if True, use MoEFFN; otherwise use a dense Expert FFN
        """
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)
        # Stage 4.3 (P4.3): factory replaces hardcoded `if cfg.attn_type == "mla"`.
        self.attn = build_attention(cfg)
        self.ffn = MoEFFN(cfg) if use_moe else Expert(cfg.dim, cfg.dim * 4 // 3, cfg=cfg)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Union[dict, StaticKVCache]] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- precomputed RoPE frequencies
            mask      -- additive causal mask or None
            kv_cache  -- cache dict mutated in-place by the attention layer
            cache_key -- key identifying this layer in the cache

        Returns:
            Output tensor of shape (B, T, dim)
        """
        x = x + self.resid_drop(
            self.attn(self.attn_norm(x), freqs_cis, mask, kv_cache, cache_key)
        )
        x = x + self.resid_drop(self.ffn(self.ffn_norm(x)))
        return x
