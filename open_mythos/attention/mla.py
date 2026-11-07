from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from open_mythos.config import MythosConfig
from open_mythos.kv_cache import StaticKVCache
from open_mythos.primitives.norm import RMSNorm
from open_mythos.primitives.quant import build_linear
from open_mythos.primitives.rope import apply_rope


# ---------------------------------------------------------------------------
# Multi-Latent Attention (DeepSeek-V2 style)
# ---------------------------------------------------------------------------


class MLAttention(nn.Module):
    """
    Multi-Latent Attention (DeepSeek-V2, 2024).

    The key insight: instead of caching full K and V tensors (each of size
    n_heads × head_dim per token), MLA compresses the KV path through a
    low-rank latent c_kv and only caches that plus the RoPE keys. K_nope and
    V are reconstructed from c_kv at each decoding step, trading a cheap
    linear projection for dramatically smaller cache memory.

    Q path:
        x → q_down (dim→q_lora_rank) → q_norm
          → q_up_nope (q_lora_rank → n_heads×qk_nope_head_dim)  [no RoPE]
          → q_up_rope (q_lora_rank → n_heads×qk_rope_head_dim)  [RoPE applied]
        q = cat(q_nope, q_rope)  per head

    KV path:
        x → kv_down (dim → kv_lora_rank + qk_rope_head_dim)
          splits into c_kv (latent, cached) and k_rope_raw (shared across heads)
        k_rope = RoPE(expand(k_rope_raw))  — applied before caching
        c_kv → kv_norm → kv_up → [k_nope | v]  — reconstructed each step
        k = cat(k_nope, k_rope)  per head

    Cache stores: c_kv (kv_lora_rank) + k_rope (n_heads × qk_rope_head_dim),
    versus full GQA cache: n_kv_heads × head_dim × 2.  At production scale this
    is roughly a 10–20× memory reduction.
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses dim, n_heads, kv_lora_rank, q_lora_rank,
                   qk_rope_head_dim, qk_nope_head_dim, v_head_dim
        """
        super().__init__()
        self.n_heads = cfg.n_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_rope_dim = cfg.qk_rope_head_dim
        self.qk_nope_dim = cfg.qk_nope_head_dim
        self.v_dim = cfg.v_head_dim
        self.q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim

        # Q compression
        self.q_down = build_linear(cfg, cfg.dim, cfg.q_lora_rank)
        self.q_norm = RMSNorm(cfg.q_lora_rank)
        self.q_up_nope = build_linear(
            cfg, cfg.q_lora_rank, cfg.n_heads * cfg.qk_nope_head_dim
        )
        self.q_up_rope = build_linear(
            cfg, cfg.q_lora_rank, cfg.n_heads * cfg.qk_rope_head_dim
        )

        # KV compression: output is [c_kv | k_rope_raw] concatenated
        self.kv_down = build_linear(
            cfg, cfg.dim, cfg.kv_lora_rank + cfg.qk_rope_head_dim
        )
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.kv_up = build_linear(
            cfg,
            cfg.kv_lora_rank,
            cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
        )

        self.wo = build_linear(cfg, cfg.n_heads * cfg.v_head_dim, cfg.dim)
        self.attn_drop = nn.Dropout(cfg.dropout)

        # Stage 4.3 (P4.3): MLA applies RoPE only on qk_rope_head_dim channels.
        self.freqs_cis_dim = cfg.attention.qk_rope_head_dim // 2

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
            freqs_cis -- RoPE frequencies sized for qk_rope_head_dim, shape (T, rope_dim//2)
            mask      -- additive causal mask of shape (1, 1, T, S) or None
            kv_cache  -- dict mutated in-place; stores {"c_kv": ..., "k_rope": ...}
            cache_key -- unique key identifying this layer in the cache dict

        Returns:
            Output tensor of shape (B, T, dim)
        """
        B, T, _ = x.shape

        # Q
        c_q = self.q_norm(self.q_down(x))
        q_nope = self.q_up_nope(c_q).view(B, T, self.n_heads, self.qk_nope_dim)
        q_rope = self.q_up_rope(c_q).view(B, T, self.n_heads, self.qk_rope_dim)
        q_rope = apply_rope(q_rope, freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1)  # (B, T, H, nope+rope)

        # KV compress
        kv_raw = self.kv_down(x)
        c_kv = kv_raw[..., : self.kv_lora_rank]  # (B, T, lora_rank)  ← cached
        k_rope = kv_raw[..., self.kv_lora_rank :]  # (B, T, rope_dim)
        # expand rope keys across heads and apply RoPE before caching so
        # retrieved keys are already positionally encoded
        k_rope = (
            k_rope.unsqueeze(2)
            .expand(B, T, self.n_heads, self.qk_rope_dim)
            .contiguous()
        )
        k_rope = apply_rope(k_rope, freqs_cis)  # (B, T, H, rope_dim) ← cached

        # P3.3: StaticKVCache fast path (no torch.cat reallocation).
        if kv_cache is not None:
            if isinstance(kv_cache, StaticKVCache):
                c_kv = kv_cache.write_and_read(f"{cache_key}:c_kv", c_kv)
                k_rope = kv_cache.write_and_read(f"{cache_key}:k_rope", k_rope)
            else:
                if cache_key in kv_cache:
                    c_kv = torch.cat([kv_cache[cache_key]["c_kv"], c_kv], dim=1)
                    k_rope = torch.cat([kv_cache[cache_key]["k_rope"], k_rope], dim=1)
                kv_cache[cache_key] = {"c_kv": c_kv.detach(), "k_rope": k_rope.detach()}

        S = c_kv.shape[1]  # full sequence length including cache

        # reconstruct K_nope and V from latent (not cached, recomputed each step)
        kv = self.kv_up(self.kv_norm(c_kv))  # (B, S, H*(nope+v))
        kv = kv.view(B, S, self.n_heads, self.qk_nope_dim + self.v_dim)
        k_nope = kv[..., : self.qk_nope_dim]  # (B, S, H, nope)
        v = kv[..., self.qk_nope_dim :]  # (B, S, H, v_dim)
        k = torch.cat([k_nope, k_rope], dim=-1)  # (B, S, H, nope+rope)

        # attention — P3.1: fused SDPA. MLA uses the same number of Q and K
        # heads (n_heads) with differing K-dim (nope+rope) and V-dim (v_head_dim);
        # SDPA supports K/V dim asymmetry so no repeat is needed.
        q = q.transpose(1, 2)  # (B, H, T, q_head_dim)
        k = k.transpose(1, 2)  # (B, H, S, q_head_dim)
        v = v.transpose(1, 2)  # (B, H, S, v_dim)

        is_causal = (mask is not None) and (q.shape[-2] == k.shape[-2]) and (q.shape[-2] > 1)
        dropout_p = self.attn_drop.p if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            dropout_p=dropout_p,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)
