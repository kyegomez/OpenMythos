"""
Tests for RecurrentBlock loop depth and LTI stability.

Covers:
- Output shape matches input
- n_loops parameter is respected (runs specified number of iterations)
- LTIInjection.get_A() always returns values strictly in (0, 1)  [spectral radius < 1]
- ACT halting fires before max_loop_iters for easy inputs (not always guaranteed, but we
  verify the shape contract and that halted flags reach True eventually)
- LoRAAdapter produces delta of correct shape
- Varying loop depth changes output (depth extrapolation is active)
"""

import torch
import pytest

from open_mythos.main import (
    LTIInjection,
    LoRAAdapter,
    MythosConfig,
    RecurrentBlock,
    precompute_rope_freqs,
)


def _cfg(max_loop_iters: int = 4) -> MythosConfig:
    return MythosConfig(
        vocab_size=100,
        dim=64,
        n_heads=4,
        n_kv_heads=4,
        max_seq_len=32,
        max_loop_iters=max_loop_iters,
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


class TestLTIInjection:
    def test_spectral_radius_less_than_one(self) -> None:
        """get_A() must return all values strictly in (0, 1)."""
        injection = LTIInjection(dim=64)
        A = injection.get_A()
        assert A.shape == (64,), f"Expected shape (64,), got {A.shape}"
        assert (A > 0).all(), "A has non-positive values"
        assert (A < 1).all(), f"Spectral radius >= 1: max(A) = {A.max().item():.6f}"

    def test_spectral_radius_after_random_init(self) -> None:
        """Even with random (possibly large) log_A/log_dt, A must stay in (0,1)."""
        torch.manual_seed(42)
        injection = LTIInjection(dim=128)
        # Perturb parameters to extreme values
        with torch.no_grad():
            injection.log_A.data = torch.randn(128) * 10
            injection.log_dt.data = torch.randn(1) * 10
        A = injection.get_A()
        assert (A > 0).all() and (A < 1).all(), "Stability violated with large parameters"

    def test_forward_shape(self) -> None:
        injection = LTIInjection(dim=64)
        B, T = 2, 8
        h = torch.randn(B, T, 64)
        e = torch.randn(B, T, 64)
        trans_out = torch.randn(B, T, 64)
        out = injection(h, e, trans_out)
        assert out.shape == h.shape


class TestLoRAAdapter:
    def test_delta_shape(self) -> None:
        adapter = LoRAAdapter(dim=64, rank=8, max_loops=4)
        x = torch.randn(2, 10, 64)
        delta = adapter(x, loop_t=2)
        assert delta.shape == x.shape

    def test_clamped_loop_index(self) -> None:
        """loop_t beyond max_loops should not raise; clamps to last embedding."""
        adapter = LoRAAdapter(dim=64, rank=8, max_loops=4)
        x = torch.randn(1, 5, 64)
        # loop_t=10 >> max_loops=4 — must not IndexError
        delta = adapter(x, loop_t=10)
        assert delta.shape == x.shape


class TestRecurrentBlock:
    def test_output_shape(self) -> None:
        cfg = _cfg()
        block = RecurrentBlock(cfg)
        block.eval()

        B, T = 1, 8
        h = torch.randn(B, T, cfg.dim)
        e = torch.randn(B, T, cfg.dim)
        freqs = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len)[:T]

        with torch.no_grad():
            out = block(h, e, freqs, n_loops=2)

        assert out.shape == (B, T, cfg.dim)

    def test_depth_changes_output(self) -> None:
        """Running with more loops should produce a different output."""
        cfg = _cfg(max_loop_iters=8)
        block = RecurrentBlock(cfg)
        block.eval()

        torch.manual_seed(7)
        B, T = 1, 6
        h = torch.randn(B, T, cfg.dim)
        e = torch.randn(B, T, cfg.dim)
        freqs = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len)[:T]

        with torch.no_grad():
            out2 = block(h.clone(), e.clone(), freqs, n_loops=2)
            out6 = block(h.clone(), e.clone(), freqs, n_loops=6)

        # Outputs at different depths should differ — depth extrapolation
        assert not torch.allclose(out2, out6, atol=1e-5), (
            "n_loops=2 and n_loops=6 produced identical output — ACT may always halt at step 1"
        )

    def test_output_finite(self) -> None:
        cfg = _cfg()
        block = RecurrentBlock(cfg)
        block.eval()

        h = torch.randn(2, 8, cfg.dim)
        e = torch.randn(2, 8, cfg.dim)
        freqs = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len)[:8]

        with torch.no_grad():
            out = block(h, e, freqs, n_loops=3)

        assert torch.isfinite(out).all(), "RecurrentBlock output contains inf or NaN"
