from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from open_mythos.config import MythosConfig
from open_mythos.kv_cache import StaticKVCache
from open_mythos.primitives.quant import build_linear
from open_mythos.primitives.rope import apply_rope


# ---------------------------------------------------------------------------
# Grouped Query Attention with KV cache
# ---------------------------------------------------------------------------


class GQAttention(nn.Module):
    """
    Grouped Query Attention (Ainslie et al., 2023).

    Uses fewer KV heads than Q heads (n_kv_heads < n_heads). Each KV head is
    shared across n_heads // n_kv_heads query heads, reducing the KV cache size
    by that factor while keeping full query expressiveness.

    RoPE is applied to both Q and K. K and V are stored in kv_cache after
    RoPE application so that cached values are already positionally encoded and
    do not need to be re-rotated on retrieval.
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses dim, n_heads, n_kv_heads
        """
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.groups = cfg.n_heads // cfg.n_kv_heads

        self.wq = build_linear(cfg, cfg.dim, cfg.n_heads * self.head_dim)
        self.wk = build_linear(cfg, cfg.dim, cfg.n_kv_heads * self.head_dim)
        self.wv = build_linear(cfg, cfg.dim, cfg.n_kv_heads * self.head_dim)
        self.wo = build_linear(cfg, cfg.n_heads * self.head_dim, cfg.dim)
        self.attn_drop = nn.Dropout(cfg.dropout)

        # Stage 4.3 (P4.3): expose rope-dim for single-buffer precomputation.
        # GQA rotates the full per-head dim.
        self.freqs_cis_dim = (cfg.dim // cfg.attention.n_heads) // 2

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
            freqs_cis -- RoPE frequencies for head_dim, shape (T, head_dim//2)
            mask      -- additive causal mask of shape (1, 1, T, S) or None
            kv_cache  -- dict mutated in-place; stores {"k": ..., "v": ...} per cache_key
            cache_key -- unique key identifying this layer in the cache dict

        Returns:
            Output tensor of shape (B, T, dim)
        """
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # P3.3: StaticKVCache writes the new chunk at the current position
        # and returns the cumulative prefix (no torch.cat, no reallocation).
        # The legacy dict path is preserved for tests written against the
        # pre-P3.3 API.
        if kv_cache is not None:
            if isinstance(kv_cache, StaticKVCache):
                k = kv_cache.write_and_read(f"{cache_key}:k", k)
                v = kv_cache.write_and_read(f"{cache_key}:v", v)
            else:
                if cache_key in kv_cache:
                    k = torch.cat([kv_cache[cache_key]["k"], k], dim=1)
                    v = torch.cat([kv_cache[cache_key]["v"], v], dim=1)
                kv_cache[cache_key] = {"k": k.detach(), "v": v.detach()}

        # P3.1: use fused scaled-dot-product attention (SDPA / FlashAttention).
        # SDPA expects (B, H, T, D). We still carry GQA's fewer KV heads; SDPA's
        # enable_gqa kwarg (PyTorch >= 2.5) handles the head broadcast internally.
        q = q.transpose(1, 2)  # (B, Hq, T, head_dim)
        k = k.transpose(1, 2)  # (B, Hkv, S, head_dim)
        v = v.transpose(1, 2)  # (B, Hkv, S, head_dim)

        # Causal mask only applies during prefill (T_q == T_kv). During decode
        # with a cached KV the new query (T_q=1) attends over everything, so
        # no mask is required.
        is_causal = (mask is not None) and (q.shape[-2] == k.shape[-2]) and (q.shape[-2] > 1)
        dropout_p = self.attn_drop.p if self.training else 0.0

        try:
            # enable_gqa is PyTorch ≥ 2.5 only; older torch raises TypeError at
            # call time and falls through to the repeat_interleave branch below.
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=is_causal,
                dropout_p=dropout_p,
                enable_gqa=(q.shape[-3] != k.shape[-3]),
            )
        except TypeError:
            # PyTorch < 2.5: no enable_gqa kwarg. Manually expand KV heads.
            if q.shape[-3] != k.shape[-3]:
                k = k.repeat_interleave(self.groups, dim=1)
                v = v.repeat_interleave(self.groups, dim=1)
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=is_causal,
                dropout_p=dropout_p,
            )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)
