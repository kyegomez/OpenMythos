"""
Shape and correctness tests for GQAttention and MLAttention.

Covers:
- GQA output shape with fallback (no flash-attn on CPU)
- GQA KV-head grouping: n_heads // n_kv_heads > 1
- MLA output shape
- MLA kv_cache population and cache-extended sequence length
- Both modules are nn.Module subclasses with parameters
"""

import torch
import pytest

from open_mythos.main import (
    GQAttention,
    MLAttention,
    MythosConfig,
    precompute_rope_freqs,
)


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _gqa_cfg(n_heads: int = 4, n_kv_heads: int = 2) -> MythosConfig:
    return MythosConfig(
        vocab_size=100,
        dim=64,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=32,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=2,
        n_shared_experts=1,
        n_experts_per_tok=1,
        expert_dim=16,
        lora_rank=4,
        dropout=0.0,
    )


def _mla_cfg() -> MythosConfig:
    return MythosConfig(
        vocab_size=100,
        dim=64,
        n_heads=4,
        n_kv_heads=4,
        max_seq_len=32,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        attn_type="mla",
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        n_experts=2,
        n_shared_experts=1,
        n_experts_per_tok=1,
        expert_dim=16,
        lora_rank=4,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# GQAttention tests
# ---------------------------------------------------------------------------


class TestGQAttention:
    def test_output_shape(self) -> None:
        cfg = _gqa_cfg()
        attn = GQAttention(cfg)
        attn.eval()

        B, T = 2, 8
        x = torch.randn(B, T, cfg.dim)
        freqs = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len)[:T]

        with torch.no_grad():
            out = attn(x, freqs)

        assert out.shape == (B, T, cfg.dim), f"Expected ({B}, {T}, {cfg.dim}), got {out.shape}"

    def test_grouped_kv_heads(self) -> None:
        """n_heads // n_kv_heads = 4 tests the repeat_interleave path."""
        cfg = _gqa_cfg(n_heads=8, n_kv_heads=2)
        attn = GQAttention(cfg)
        attn.eval()

        B, T = 1, 4
        x = torch.randn(B, T, cfg.dim)
        freqs = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len)[:T]

        with torch.no_grad():
            out = attn(x, freqs)

        assert out.shape == (B, T, cfg.dim)

    def test_is_nn_module_with_params(self) -> None:
        cfg = _gqa_cfg()
        attn = GQAttention(cfg)
        assert isinstance(attn, torch.nn.Module)
        assert sum(p.numel() for p in attn.parameters()) > 0

    def test_kv_cache_populated(self) -> None:
        cfg = _gqa_cfg()
        attn = GQAttention(cfg)
        attn.eval()

        B, T = 1, 5
        x = torch.randn(B, T, cfg.dim)
        freqs = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len)[:T]
        cache: dict = {}

        with torch.no_grad():
            attn(x, freqs, kv_cache=cache, cache_key="layer0")

        assert "layer0" in cache
        assert "k" in cache["layer0"] and "v" in cache["layer0"]
        # Cache should hold T tokens
        assert cache["layer0"]["k"].shape[1] == T


# ---------------------------------------------------------------------------
# MLAttention tests
# ---------------------------------------------------------------------------


class TestMLAttention:
    def test_output_shape(self) -> None:
        cfg = _mla_cfg()
        attn = MLAttention(cfg)
        attn.eval()

        B, T = 2, 8
        x = torch.randn(B, T, cfg.dim)
        freqs = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len)[:T]

        with torch.no_grad():
            out = attn(x, freqs)

        assert out.shape == (B, T, cfg.dim)

    def test_kv_cache_extends_sequence(self) -> None:
        """Second call with cache should produce output for T=1 given S=T_prev+1."""
        cfg = _mla_cfg()
        attn = MLAttention(cfg)
        attn.eval()

        T_prefill = 6
        x_pre = torch.randn(1, T_prefill, cfg.dim)
        freqs_pre = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len)[:T_prefill]
        cache: dict = {}

        with torch.no_grad():
            attn(x_pre, freqs_pre, kv_cache=cache, cache_key="mla_l0")
            # Now decode one token
            x_dec = torch.randn(1, 1, cfg.dim)
            freqs_dec = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len)[T_prefill : T_prefill + 1]
            out = attn(x_dec, freqs_dec, kv_cache=cache, cache_key="mla_l0")

        # Output for single decode token
        assert out.shape == (1, 1, cfg.dim)
        # Cache now holds prefill + decode tokens
        assert cache["mla_l0"]["c_kv"].shape[1] == T_prefill + 1

    def test_is_nn_module_with_params(self) -> None:
        cfg = _mla_cfg()
        attn = MLAttention(cfg)
        assert isinstance(attn, torch.nn.Module)
        assert sum(p.numel() for p in attn.parameters()) > 0
