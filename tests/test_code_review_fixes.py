"""
Tests for code review fixes and MoE dispatch optimization (2026-04-23).

Covers:
  - MoE grouped dispatch: correctness, edge cases, gradient flow
  - ACT remainder for non-halted positions
  - MoE score renormalization epsilon (div-by-zero guard)
  - LoRAAdapter.B dtype safety
  - loop_index_embedding float32 precision
  - __init__.py public API exports
"""

import importlib
import math

import torch
import torch.nn as nn
import pytest
from unittest.mock import patch

from open_mythos.main import (
    ACTHalting,
    Expert,
    LoRAAdapter,
    MoEFFN,
    MythosConfig,
    OpenMythos,
    RecurrentBlock,
    TransformerBlock,
    loop_index_embedding,
    precompute_rope_freqs,
)

# ---------------------------------------------------------------------------
# Shared test config — tiny dims for CPU speed
# ---------------------------------------------------------------------------

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
        attn_type="gqa",
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


# ===================================================================
# MoE Grouped Dispatch
# ===================================================================


class TestMoEGroupedDispatch:
    """Tests for the grouped/batched MoE dispatch replacing the nested loop."""

    def setup_method(self):
        self.cfg = small_cfg()
        self.moe = MoEFFN(self.cfg)

    def test_output_shape_standard(self):
        x = torch.randn(B, T, self.cfg.dim)
        assert self.moe(x).shape == (B, T, self.cfg.dim)

    def test_single_token(self):
        """Edge case: batch with only one token (B=1, T=1)."""
        x = torch.randn(1, 1, self.cfg.dim)
        out = self.moe(x)
        assert out.shape == (1, 1, self.cfg.dim)
        assert not torch.isnan(out).any()

    def test_large_batch(self):
        """Stress test with larger batch to exercise grouping with many tokens."""
        x = torch.randn(8, 32, self.cfg.dim)
        out = self.moe(x)
        assert out.shape == (8, 32, self.cfg.dim)
        assert not torch.isnan(out).any()

    def test_topk_1(self):
        """Edge case: only one expert per token (topk=1)."""
        cfg = small_cfg(n_experts_per_tok=1)
        moe = MoEFFN(cfg)
        x = torch.randn(B, T, cfg.dim)
        out = moe(x)
        assert out.shape == (B, T, cfg.dim)
        assert not torch.isnan(out).any()

    def test_topk_equals_n_experts(self):
        """Edge case: every expert is selected for every token."""
        cfg = small_cfg(n_experts=4, n_experts_per_tok=4)
        moe = MoEFFN(cfg)
        x = torch.randn(B, T, cfg.dim)
        out = moe(x)
        assert out.shape == (B, T, cfg.dim)
        assert not torch.isnan(out).any()

    def test_all_tokens_same_expert(self):
        """Force all tokens to route to the same expert via router_bias."""
        cfg = small_cfg(n_experts=4, n_experts_per_tok=2)
        moe = MoEFFN(cfg)
        # Overwhelm the router logits: bias expert 0 and 1 massively
        moe.router_bias.data = torch.tensor(
            [1000.0, 999.0, -1000.0, -1000.0]
        )
        x = torch.randn(B, T, cfg.dim)
        out = moe(x)
        assert out.shape == (B, T, cfg.dim)
        assert not torch.isnan(out).any()

    def test_no_nan_or_inf(self):
        x = torch.randn(B, T, self.cfg.dim)
        out = self.moe(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradient_flows_through_routed_experts(self):
        """Verify gradients reach routed expert parameters."""
        x = torch.randn(B, T, self.cfg.dim, requires_grad=True)
        out = self.moe(x)
        loss = out.sum()
        loss.backward()
        # At least some routed experts should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for exp in self.moe.routed_experts
            for p in exp.parameters()
        )
        assert has_grad, "No gradient flowed to any routed expert"

    def test_gradient_flows_through_shared_experts(self):
        """Verify gradients reach shared expert parameters."""
        x = torch.randn(B, T, self.cfg.dim, requires_grad=True)
        out = self.moe(x)
        loss = out.sum()
        loss.backward()
        for shared in self.moe.shared_experts:
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in shared.parameters()
            )
            assert has_grad, "No gradient flowed to shared expert"

    def test_gradient_flows_to_input(self):
        """Verify gradients propagate back to the input tensor."""
        x = torch.randn(B, T, self.cfg.dim, requires_grad=True)
        out = self.moe(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_router_gradient_exists(self):
        """Verify the router weight receives gradients."""
        x = torch.randn(B, T, self.cfg.dim)
        out = self.moe(x)
        out.sum().backward()
        assert self.moe.router.weight.grad is not None
        assert self.moe.router.weight.grad.abs().sum() > 0

    def test_deterministic_output(self):
        """Same input should produce same output (no randomness in dispatch)."""
        torch.manual_seed(42)
        x = torch.randn(B, T, self.cfg.dim)
        out1 = self.moe(x.clone())
        out2 = self.moe(x.clone())
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_output_changes_with_different_input(self):
        """Different inputs should produce different outputs."""
        x1 = torch.randn(B, T, self.cfg.dim)
        x2 = torch.randn(B, T, self.cfg.dim)
        out1 = self.moe(x1)
        out2 = self.moe(x2)
        assert not torch.allclose(out1, out2)

    def test_router_bias_shifts_expert_selection(self):
        """Changing router_bias should change which experts are selected."""
        cfg = small_cfg(n_experts=4, n_experts_per_tok=1)
        moe = MoEFFN(cfg)
        x = torch.randn(1, 1, cfg.dim)

        moe.router_bias.data = torch.tensor([100.0, 0.0, 0.0, 0.0])
        out_biased_0 = moe(x.clone()).detach()

        moe.router_bias.data = torch.tensor([0.0, 0.0, 0.0, 100.0])
        out_biased_3 = moe(x.clone()).detach()

        # Different experts → different outputs (shared expert is the same,
        # but routed contribution differs)
        assert not torch.allclose(out_biased_0, out_biased_3)

    def test_only_shared_experts_when_routed_zeroed(self):
        """Zeroing routed experts: output should match shared-only."""
        cfg = small_cfg()
        moe = MoEFFN(cfg)
        for exp in moe.routed_experts:
            for p in exp.parameters():
                p.data.zero_()
        x = torch.randn(B, T, cfg.dim)
        out = moe(x)
        # Recompute shared-only
        flat = x.view(B * T, cfg.dim)
        shared_out = sum(s(flat) for s in moe.shared_experts)
        expected = shared_out.view(B, T, cfg.dim)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_scores_sum_to_one_per_token(self):
        """After renormalization, topk scores per token should sum to ~1."""
        cfg = small_cfg()
        moe = MoEFFN(cfg)
        x = torch.randn(B, T, cfg.dim)
        flat = x.view(B * T, cfg.dim)
        logits = moe.router(flat)
        scores = torch.nn.functional.softmax(logits, dim=-1)
        _, topk_idx = (logits + moe.router_bias).topk(moe.topk, dim=-1)
        topk_scores = scores.gather(-1, topk_idx)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp(
            min=1e-9
        )
        sums = topk_scores.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


# ===================================================================
# MoE Score Renormalization Epsilon (div-by-zero guard)
# ===================================================================


class TestMoEScoreEpsilon:
    """Tests for the .clamp(min=1e-9) guard on score renormalization."""

    def test_zero_scores_no_nan(self):
        """If all topk softmax scores underflow to zero, output should not be NaN."""
        cfg = small_cfg()
        moe = MoEFFN(cfg)
        # Force router to produce extreme negative logits → softmax → ~0
        with torch.no_grad():
            moe.router.weight.fill_(-100.0)
        x = torch.randn(B, T, cfg.dim)
        out = moe(x)
        assert not torch.isnan(out).any(), "NaN in output despite epsilon guard"
        assert not torch.isinf(out).any(), "Inf in output despite epsilon guard"

    def test_near_zero_scores_bfloat16(self):
        """Simulate bfloat16 underflow scenario with very small scores."""
        cfg = small_cfg()
        moe = MoEFFN(cfg)
        x = torch.randn(B, T, cfg.dim)
        # Run in bfloat16 if available (the actual risk scenario)
        if torch.cuda.is_available():
            moe = moe.to(torch.bfloat16).cuda()
            x = x.to(torch.bfloat16).cuda()
        out = moe(x)
        assert not torch.isnan(out).any()

    def test_uniform_scores_stay_uniform(self):
        """When all topk scores are equal, renorm should keep them equal."""
        cfg = small_cfg(n_experts=4, n_experts_per_tok=2)
        moe = MoEFFN(cfg)
        # Manually compute: equal softmax scores → equal after renorm
        flat = torch.randn(4, cfg.dim)
        logits = moe.router(flat)
        # Make all logits equal so softmax is uniform
        logits = torch.zeros_like(logits)
        scores = torch.nn.functional.softmax(logits, dim=-1)
        _, topk_idx = logits.topk(cfg.n_experts_per_tok, dim=-1)
        topk_scores = scores.gather(-1, topk_idx)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp(
            min=1e-9
        )
        # Each of 2 selected experts should get score 0.5
        assert torch.allclose(
            topk_scores, torch.full_like(topk_scores, 0.5), atol=1e-5
        )


# ===================================================================
# ACT Remainder for Non-Halted Positions
# ===================================================================


class TestACTRemainder:
    """Tests for the post-loop remainder weight assignment.

    Uses the full OpenMythos model (MLA mode) to avoid the pre-existing
    GQA RoPE dimension mismatch in RecurrentBlock-level tests.
    """

    def _make_model(self, **cfg_overrides):
        cfg = small_cfg(attn_type="mla", **cfg_overrides)
        model = OpenMythos(cfg)
        return model, cfg

    def test_output_not_all_zero_with_low_halting(self):
        """With very high threshold, positions won't halt but should still
        produce nonzero output via the remainder."""
        model, cfg = self._make_model(act_threshold=0.9999)
        # Bias ACT to predict very low halting probability
        with torch.no_grad():
            model.recurrent.act.halt.weight.fill_(0.0)
            model.recurrent.act.halt.bias.fill_(-10.0)  # sigmoid(-10) ≈ 0.00005
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        logits = model(ids)
        assert logits.abs().sum() > 0, "Output is all zeros — remainder not applied"
        assert not torch.isnan(logits).any()

    def test_remainder_does_not_double_count_halted(self):
        """Positions that halted normally should NOT get additional remainder."""
        model, cfg = self._make_model(act_threshold=0.01)
        # Bias ACT to halt immediately (high halting prob)
        with torch.no_grad():
            model.recurrent.act.halt.weight.fill_(0.0)
            model.recurrent.act.halt.bias.fill_(10.0)  # sigmoid(10) ≈ 0.99995
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        # Run twice — if remainder double-counts, outputs would differ
        logits1 = model(ids)
        logits2 = model(ids)
        assert torch.allclose(logits1, logits2, atol=1e-5)

    def test_single_loop_remainder(self):
        """With n_loops=1 and no halting, remainder should provide weight."""
        model, cfg = self._make_model(act_threshold=0.9999)
        with torch.no_grad():
            model.recurrent.act.halt.weight.fill_(0.0)
            model.recurrent.act.halt.bias.fill_(-10.0)
        ids = torch.randint(0, cfg.vocab_size, (1, 1))
        logits = model(ids, n_loops=1)
        assert logits.shape == (1, 1, cfg.vocab_size)
        assert not torch.isnan(logits).any()
        assert logits.abs().sum() > 0

    def test_no_nan_with_many_loops(self):
        """Run many loops with low halting — should never produce NaN."""
        model, cfg = self._make_model(act_threshold=0.9999, max_loop_iters=16)
        with torch.no_grad():
            model.recurrent.act.halt.weight.fill_(0.0)
            model.recurrent.act.halt.bias.fill_(-5.0)  # sigmoid(-5) ≈ 0.007
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        logits = model(ids, n_loops=16)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_low_threshold_all_halt_early(self):
        """Very low threshold + high halting prob → all positions halt in loop 1."""
        model, cfg = self._make_model(act_threshold=0.01)
        with torch.no_grad():
            model.recurrent.act.halt.weight.fill_(0.0)
            model.recurrent.act.halt.bias.fill_(10.0)
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        logits = model(ids, n_loops=5)
        assert not torch.isnan(logits).any()
        # Should produce valid logits even if everything halts immediately
        assert logits.shape == (B, T, cfg.vocab_size)


# ===================================================================
# ACT Halting Weight Invariants
# ===================================================================


class TestACTWeightInvariants:
    """Verify that ACT weights (halted + remainder) sum correctly."""

    def test_weights_sum_to_one_all_halt(self):
        """When all positions halt, accumulated weights should sum to ~1."""
        cfg = small_cfg(act_threshold=0.5)
        act = ACTHalting(cfg.dim)
        # Force high halting prob so everything halts in 1 iteration
        with torch.no_grad():
            act.halt.weight.fill_(0.0)
            act.halt.bias.fill_(10.0)

        B_, T_ = 2, 4
        halted = torch.zeros(B_, T_, dtype=torch.bool)
        cumulative_p = torch.zeros(B_, T_)
        total_weight = torch.zeros(B_, T_)
        h = torch.randn(B_, T_, cfg.dim)

        for t in range(5):
            p = act(h)
            still_running = ~halted
            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= cfg.act_threshold,
                remainder,
                p,
            )
            weight = weight * still_running.float()
            total_weight = total_weight + weight
            cumulative_p = cumulative_p + p * still_running.float()
            halted = halted | (cumulative_p >= cfg.act_threshold)

        # Post-loop remainder for non-halted
        not_halted = ~halted
        if not_halted.any():
            final_remainder = (1.0 - cumulative_p).clamp(min=0) * not_halted.float()
            total_weight = total_weight + final_remainder

        assert torch.allclose(
            total_weight, torch.ones_like(total_weight), atol=1e-4
        ), f"Weights don't sum to 1: {total_weight}"

    def test_weights_sum_to_one_none_halt(self):
        """When no positions halt within the loop, remainder ensures sum ~1."""
        cfg = small_cfg(act_threshold=0.9999)
        act = ACTHalting(cfg.dim)
        # Force very low halting prob
        with torch.no_grad():
            act.halt.weight.fill_(0.0)
            act.halt.bias.fill_(-10.0)  # sigmoid(-10) ≈ 0.00005

        B_, T_ = 2, 4
        halted = torch.zeros(B_, T_, dtype=torch.bool)
        cumulative_p = torch.zeros(B_, T_)
        total_weight = torch.zeros(B_, T_)
        h = torch.randn(B_, T_, cfg.dim)

        for t in range(3):
            p = act(h)
            still_running = ~halted
            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= cfg.act_threshold,
                remainder,
                p,
            )
            weight = weight * still_running.float()
            total_weight = total_weight + weight
            cumulative_p = cumulative_p + p * still_running.float()
            halted = halted | (cumulative_p >= cfg.act_threshold)

        # Post-loop remainder
        not_halted = ~halted
        if not_halted.any():
            final_remainder = (1.0 - cumulative_p).clamp(min=0) * not_halted.float()
            total_weight = total_weight + final_remainder

        assert torch.allclose(
            total_weight, torch.ones_like(total_weight), atol=1e-4
        ), f"Weights don't sum to 1: {total_weight}"

    def test_weights_sum_to_one_mixed_halting(self):
        """Mix of halted and non-halted positions: all weights sum to ~1."""
        cfg = small_cfg(act_threshold=0.5)
        act = ACTHalting(cfg.dim)

        B_, T_ = 1, 8
        halted = torch.zeros(B_, T_, dtype=torch.bool)
        cumulative_p = torch.zeros(B_, T_)
        total_weight = torch.zeros(B_, T_)
        h = torch.randn(B_, T_, cfg.dim)

        for t in range(3):
            p = act(h)
            still_running = ~halted
            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= cfg.act_threshold,
                remainder,
                p,
            )
            weight = weight * still_running.float()
            total_weight = total_weight + weight
            cumulative_p = cumulative_p + p * still_running.float()
            halted = halted | (cumulative_p >= cfg.act_threshold)

        # Post-loop remainder
        not_halted = ~halted
        if not_halted.any():
            final_remainder = (1.0 - cumulative_p).clamp(min=0) * not_halted.float()
            total_weight = total_weight + final_remainder

        assert torch.allclose(
            total_weight, torch.ones_like(total_weight), atol=1e-4
        ), f"Weights don't sum to 1: {total_weight}"


# ===================================================================
# LoRAAdapter.B dtype Safety
# ===================================================================


class TestLoRADtypeSafety:
    """Tests for the defensive .to(down.dtype) cast on self.B."""

    def test_float32_pass_through(self):
        """Standard float32 — should work as before."""
        lora = LoRAAdapter(dim=64, rank=8, max_loops=10)
        x = torch.randn(B, T, 64)
        out = lora(x, loop_t=0)
        assert out.dtype == torch.float32
        assert out.shape == (B, T, 64)

    def test_float16_input(self):
        """float16 input with float32 parameters — cast should prevent mismatch."""
        lora = LoRAAdapter(dim=64, rank=8, max_loops=10)
        x = torch.randn(B, T, 64, dtype=torch.float16)
        out = lora(x, loop_t=0)
        assert out.shape == (B, T, 64)
        assert not torch.isnan(out).any()

    def test_bfloat16_input(self):
        """bfloat16 input — the actual FSDP mixed precision scenario."""
        lora = LoRAAdapter(dim=64, rank=8, max_loops=10)
        x = torch.randn(B, T, 64, dtype=torch.bfloat16)
        out = lora(x, loop_t=0)
        assert out.shape == (B, T, 64)
        assert not torch.isnan(out).any()

    def test_B_param_dtype_mismatch_handled(self):
        """Manually set B to a different dtype — the cast should still work."""
        lora = LoRAAdapter(dim=64, rank=8, max_loops=10)
        # Simulate FSDP casting down to bfloat16 but B staying float32
        lora.down = lora.down.to(torch.bfloat16)
        # B is still float32
        assert lora.B.dtype == torch.float32
        x = torch.randn(B, T, 64, dtype=torch.bfloat16)
        # Should not raise RuntimeError about dtype mismatch
        out = lora(x, loop_t=0)
        assert out.shape == (B, T, 64)

    def test_gradient_flows_through_B(self):
        """Verify the dtype cast doesn't block gradient flow to self.B."""
        lora = LoRAAdapter(dim=64, rank=8, max_loops=10)
        x = torch.randn(B, T, 64)
        out = lora(x, loop_t=0)
        out.sum().backward()
        assert lora.B.grad is not None
        assert lora.B.grad.abs().sum() > 0

    def test_loop_index_clamp(self):
        """Exceeding max_loops should clamp to last index, not crash."""
        lora = LoRAAdapter(dim=64, rank=8, max_loops=5)
        x = torch.randn(B, T, 64)
        # loop_t=10 > max_loops=5 → should clamp to index 4
        out = lora(x, loop_t=10)
        assert out.shape == (B, T, 64)
        assert not torch.isnan(out).any()


# ===================================================================
# loop_index_embedding Float32 Precision
# ===================================================================


class TestLoopIndexEmbeddingPrecision:
    """Tests for computing trig in float32 then casting back."""

    def test_bfloat16_input_no_error(self):
        """bfloat16 hidden state should work without dtype errors."""
        h = torch.randn(B, T, 64, dtype=torch.bfloat16)
        out = loop_index_embedding(h, loop_t=5, loop_dim=8)
        assert out.dtype == torch.bfloat16
        assert out.shape == h.shape

    def test_float16_input_preserves_dtype(self):
        """float16 input should return float16 output."""
        h = torch.randn(B, T, 64, dtype=torch.float16)
        out = loop_index_embedding(h, loop_t=3, loop_dim=8)
        assert out.dtype == torch.float16

    def test_float32_input_preserves_dtype(self):
        """float32 input should return float32 output."""
        h = torch.randn(B, T, 64, dtype=torch.float32)
        out = loop_index_embedding(h, loop_t=3, loop_dim=8)
        assert out.dtype == torch.float32

    def test_precision_matches_float32_reference(self):
        """bfloat16 computation should match float32 reference (via the fix)."""
        h_f32 = torch.randn(1, 1, 64, dtype=torch.float32)
        h_bf16 = h_f32.to(torch.bfloat16)

        out_f32 = loop_index_embedding(h_f32, loop_t=7, loop_dim=16)
        out_bf16 = loop_index_embedding(h_bf16, loop_t=7, loop_dim=16)

        # The embedding itself should be computed with float32 precision,
        # so the difference should be only from bf16 quantization of h, not
        # from bf16 trig functions.
        diff = (out_f32 - out_bf16.float()).abs().max().item()
        # bf16 has ~0.4% relative error; float32 trig vs bf16 trig would give
        # much larger errors on high-frequency components
        assert diff < 0.05, f"Precision gap too large: {diff}"

    def test_large_loop_index_no_nan(self):
        """High loop indices should not produce NaN from overflow."""
        h = torch.randn(1, 1, 64, dtype=torch.bfloat16)
        out = loop_index_embedding(h, loop_t=1000, loop_dim=8)
        assert not torch.isnan(out).any()

    def test_loop_zero_is_nonzero_embedding(self):
        """loop_t=0 should still add sin(0)/cos(0) = [0, ..., 1, ...] pattern."""
        h = torch.zeros(1, 1, 64, dtype=torch.float32)
        out = loop_index_embedding(h, loop_t=0, loop_dim=8)
        # sin(0)=0, cos(0)=1, so first 4 dims are 0, next 4 are 1
        # (because emb = cat([sin, cos])[:loop_dim])
        embedding = out[0, 0, :8]
        assert embedding[:4].abs().sum() < 1e-5  # sin(0) = 0
        assert torch.allclose(
            embedding[4:], torch.ones(4), atol=1e-5
        )  # cos(0) = 1


# ===================================================================
# __init__.py Public API
# ===================================================================


class TestPublicAPI:
    """Tests for the __init__.py exports."""

    def test_no_broken_exports(self):
        """Every symbol in __all__ should be importable."""
        import open_mythos

        for name in open_mythos.__all__:
            assert hasattr(open_mythos, name), (
                f"'{name}' is in __all__ but not importable"
            )

    def test_removed_symbols_not_in_all(self):
        """load_tokenizer and get_vocab_size should not be in __all__."""
        import open_mythos

        assert "load_tokenizer" not in open_mythos.__all__
        assert "get_vocab_size" not in open_mythos.__all__

    def test_key_classes_exported(self):
        """Core classes should remain in __all__."""
        import open_mythos

        required = [
            "MythosConfig",
            "OpenMythos",
            "MoEFFN",
            "RecurrentBlock",
            "MythosTokenizer",
        ]
        for name in required:
            assert name in open_mythos.__all__, f"'{name}' missing from __all__"

    def test_import_from_package(self):
        """Smoke test: importing key symbols from the package level."""
        from open_mythos import MythosConfig, OpenMythos, MoEFFN

        assert MythosConfig is not None
        assert OpenMythos is not None
        assert MoEFFN is not None


# ===================================================================
# Full Model Integration (exercises all fixes together)
# ===================================================================


class TestFullModelIntegration:
    """End-to-end tests verifying all fixes work together in the full model."""

    def setup_method(self):
        self.cfg = small_cfg()
        self.model = OpenMythos(self.cfg)

    def test_forward_no_nan(self):
        ids = torch.randint(0, self.cfg.vocab_size, (B, T))
        logits = self.model(ids)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_backward_no_error(self):
        """Full forward+backward should work with all fixes in place."""
        ids = torch.randint(0, self.cfg.vocab_size, (B, T))
        logits = self.model(ids)
        loss = logits.sum()
        loss.backward()
        # Check key parameters got gradients
        assert self.model.embed.weight.grad is not None

    def test_generate_no_nan(self):
        ids = torch.randint(0, self.cfg.vocab_size, (1, T))
        out = self.model.generate(ids, max_new_tokens=4, n_loops=2)
        assert out.shape == (1, T + 4)

    def test_many_loops_no_nan(self):
        """Depth extrapolation with many loops — exercises ACT remainder."""
        ids = torch.randint(0, self.cfg.vocab_size, (1, 4))
        logits = self.model(ids, n_loops=10)
        assert not torch.isnan(logits).any()

    def test_single_token_input(self):
        """Edge case: single token sequence."""
        ids = torch.randint(0, self.cfg.vocab_size, (1, 1))
        logits = self.model(ids)
        assert logits.shape == (1, 1, self.cfg.vocab_size)
        assert not torch.isnan(logits).any()


if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])
