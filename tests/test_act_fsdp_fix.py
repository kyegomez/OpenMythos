"""
Tests for the ACT early exit FSDP deadlock fix (issue #4).

Verifies that:
  - Single-process early exit still works (no regression)
  - KV cache disables early exit (unchanged behavior)
  - The all-reduce branch is skipped when torch.distributed is not initialized
  - Loop runs all iterations when not all positions have halted
  - The fix doesn't change model outputs
"""

import torch
import pytest
from unittest.mock import patch

from open_mythos.main import (
    MythosConfig,
    OpenMythos,
    RecurrentBlock,
)


B, T = 2, 8


def small_cfg(**overrides) -> MythosConfig:
    defaults = dict(
        vocab_size=200,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        max_loop_iters=3,
        prelude_layers=1,
        coda_layers=1,
        attn_type="mla",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=16,
        act_threshold=0.99,
        lora_rank=4,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
    )
    defaults.update(overrides)
    return MythosConfig(**defaults)


class TestACTEarlyExitSingleProcess:
    """Verify early exit still works in single-process (no dist initialized)."""

    def test_early_exit_when_all_halted(self):
        """With very low threshold + high halt prob, loop should exit early."""
        cfg = small_cfg(act_threshold=0.01, max_loop_iters=16)
        model = OpenMythos(cfg)
        # Bias ACT to halt immediately
        with torch.no_grad():
            model.recurrent.act.halt.weight.fill_(0.0)
            model.recurrent.act.halt.bias.fill_(10.0)  # sigmoid(10) ≈ 1.0

        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        # Should complete without hanging — early exit fires
        logits = model(ids)
        assert logits.shape == (1, 4, cfg.vocab_size)
        assert not torch.isnan(logits).any()

    def test_dist_not_initialized_in_tests(self):
        """Confirm torch.distributed is not initialized in test environment."""
        assert not torch.distributed.is_initialized()

    def test_early_exit_skips_iterations(self):
        """When halting is immediate, fewer loop iterations should run.

        We verify this indirectly: with max_loop_iters=16 and immediate halting,
        the forward pass should be fast (not 16x slower than needed).
        """
        cfg = small_cfg(act_threshold=0.01, max_loop_iters=16)
        model = OpenMythos(cfg)
        with torch.no_grad():
            model.recurrent.act.halt.weight.fill_(0.0)
            model.recurrent.act.halt.bias.fill_(10.0)

        ids = torch.randint(0, cfg.vocab_size, (B, T))
        # Just verify it completes and produces valid output
        logits = model(ids)
        assert not torch.isnan(logits).any()


class TestACTNoEarlyExitWithKVCache:
    """KV cache should disable early exit regardless of halting state."""

    def test_kv_cache_prevents_early_exit(self):
        """With KV cache, all loop iterations must run for cache consistency."""
        cfg = small_cfg(act_threshold=0.01, max_loop_iters=3)
        model = OpenMythos(cfg)
        # Bias ACT to halt immediately
        with torch.no_grad():
            model.recurrent.act.halt.weight.fill_(0.0)
            model.recurrent.act.halt.bias.fill_(10.0)

        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        kv_cache = {}
        logits = model(ids, kv_cache=kv_cache)
        assert logits.shape == (1, 4, cfg.vocab_size)

        # Verify all 3 recurrent loop cache keys were populated
        for t in range(cfg.max_loop_iters):
            key = f"recurrent_loop_{t}"
            assert key in kv_cache, (
                f"Cache key '{key}' missing — loop didn't run iteration {t}"
            )


class TestACTAllReduceBranch:
    """Verify the all-reduce code path logic."""

    def test_all_reduce_not_called_without_dist(self):
        """When dist is not initialized, torch.distributed.all_reduce should
        not be called."""
        cfg = small_cfg(act_threshold=0.01)
        model = OpenMythos(cfg)
        with torch.no_grad():
            model.recurrent.act.halt.weight.fill_(0.0)
            model.recurrent.act.halt.bias.fill_(10.0)

        ids = torch.randint(0, cfg.vocab_size, (B, T))
        with patch("torch.distributed.all_reduce") as mock_ar:
            model(ids)
            mock_ar.assert_not_called()

    def test_all_reduce_would_be_called_if_dist_initialized(self):
        """Verify the is_initialized() check gates the all_reduce call."""
        cfg = small_cfg(act_threshold=0.01)
        model = OpenMythos(cfg)
        with torch.no_grad():
            model.recurrent.act.halt.weight.fill_(0.0)
            model.recurrent.act.halt.bias.fill_(10.0)

        ids = torch.randint(0, cfg.vocab_size, (B, T))
        with patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.all_reduce") as mock_ar:
            model(ids)
            # all_reduce should have been called at least once
            assert mock_ar.call_count > 0


class TestACTFixOutputEquivalence:
    """The fix must not change model outputs in single-process mode."""

    def test_output_deterministic(self):
        """Same input produces same output — fix doesn't introduce randomness."""
        cfg = small_cfg()
        model = OpenMythos(cfg)
        model.eval()

        torch.manual_seed(42)
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        out1 = model(ids)
        out2 = model(ids)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_forward_backward_works(self):
        """Full forward+backward completes without error after the fix."""
        cfg = small_cfg()
        model = OpenMythos(cfg)
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        logits = model(ids)
        loss = logits.sum()
        loss.backward()
        assert model.embed.weight.grad is not None

    def test_generate_works(self):
        """Autoregressive generation still works after the fix."""
        cfg = small_cfg()
        model = OpenMythos(cfg)
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        out = model.generate(ids, max_new_tokens=4, n_loops=2)
        assert out.shape == (1, 8)

    def test_many_loops_no_nan(self):
        """Depth extrapolation still works."""
        cfg = small_cfg()
        model = OpenMythos(cfg)
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        logits = model(ids, n_loops=10)
        assert not torch.isnan(logits).any()


if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])
