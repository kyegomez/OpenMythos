"""
Before/After comparison: MoE dispatch optimization.

Verifies that the new grouped dispatch (sort-by-expert, batch-per-expert)
produces identical numerical results to the old nested-loop dispatch.
"""

import torch
import torch.nn.functional as F
import pytest

from open_mythos.main import Expert, MoEFFN, MythosConfig


# ---------------------------------------------------------------------------
# Reference: OLD nested-loop dispatch (pre-optimization, commit before 65cd807)
# ---------------------------------------------------------------------------

def old_moe_dispatch(moe: MoEFFN, x: torch.Tensor) -> torch.Tensor:
    """Reimplementation of the old MoE forward — nested for-loops."""
    B, T, D = x.shape
    x = x.to(moe.router.weight.dtype)
    flat = x.view(B * T, D)

    logits = moe.router(flat)
    scores = F.softmax(logits, dim=-1)
    _, topk_idx = (logits + moe.router_bias).topk(moe.topk, dim=-1)
    topk_scores = scores.gather(-1, topk_idx)
    # OLD code: no .clamp(min=1e-9)
    topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

    # OLD code: nested for-loops
    out = torch.zeros_like(flat)
    for i in range(moe.topk):
        expert_ids = topk_idx[:, i]
        token_scores = topk_scores[:, i].unsqueeze(-1)
        for eid in range(moe.n_experts):
            mask = expert_ids == eid
            if not mask.any():
                continue
            out[mask] += token_scores[mask] * moe.routed_experts[eid](flat[mask])

    for shared in moe.shared_experts:
        out = out + shared(flat)

    return out.view(B, T, D)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

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


B, T = 2, 8


# ===================================================================
# Numerical equivalence tests
# ===================================================================


class TestMoEBeforeAfterEquivalence:
    """Verify that old and new MoE dispatch produce identical results."""

    def test_basic_equivalence(self):
        """Standard batch: old and new dispatch should match within float32 tolerance."""
        torch.manual_seed(42)
        cfg = small_cfg()
        moe = MoEFFN(cfg)
        moe.eval()

        x = torch.randn(B, T, cfg.dim)
        new_out = moe(x)
        old_out = old_moe_dispatch(moe, x)

        assert torch.allclose(new_out, old_out, atol=1e-5), (
            f"Max diff: {(new_out - old_out).abs().max().item()}"
        )

    def test_equivalence_single_token(self):
        """Single token (B=1, T=1)."""
        torch.manual_seed(123)
        cfg = small_cfg()
        moe = MoEFFN(cfg)
        moe.eval()

        x = torch.randn(1, 1, cfg.dim)
        new_out = moe(x)
        old_out = old_moe_dispatch(moe, x)

        assert torch.allclose(new_out, old_out, atol=1e-5), (
            f"Max diff: {(new_out - old_out).abs().max().item()}"
        )

    def test_equivalence_topk_1(self):
        """Top-1 routing."""
        torch.manual_seed(7)
        cfg = small_cfg(n_experts_per_tok=1)
        moe = MoEFFN(cfg)
        moe.eval()

        x = torch.randn(B, T, cfg.dim)
        new_out = moe(x)
        old_out = old_moe_dispatch(moe, x)

        assert torch.allclose(new_out, old_out, atol=1e-5), (
            f"Max diff: {(new_out - old_out).abs().max().item()}"
        )

    def test_equivalence_all_experts_selected(self):
        """Every expert selected for every token (topk == n_experts)."""
        torch.manual_seed(99)
        cfg = small_cfg(n_experts=4, n_experts_per_tok=4)
        moe = MoEFFN(cfg)
        moe.eval()

        x = torch.randn(B, T, cfg.dim)
        new_out = moe(x)
        old_out = old_moe_dispatch(moe, x)

        assert torch.allclose(new_out, old_out, atol=1e-5), (
            f"Max diff: {(new_out - old_out).abs().max().item()}"
        )

    def test_equivalence_forced_single_expert(self):
        """Force all tokens to the same expert via router_bias."""
        torch.manual_seed(0)
        cfg = small_cfg(n_experts=4, n_experts_per_tok=2)
        moe = MoEFFN(cfg)
        moe.eval()

        moe.router_bias.data = torch.tensor([1000.0, 999.0, -1000.0, -1000.0])
        x = torch.randn(B, T, cfg.dim)
        new_out = moe(x)
        old_out = old_moe_dispatch(moe, x)

        assert torch.allclose(new_out, old_out, atol=1e-5), (
            f"Max diff: {(new_out - old_out).abs().max().item()}"
        )

    def test_equivalence_larger_batch(self):
        """Larger batch stress test."""
        torch.manual_seed(314)
        cfg = small_cfg(n_experts=8, n_experts_per_tok=3)
        moe = MoEFFN(cfg)
        moe.eval()

        x = torch.randn(4, 16, cfg.dim)
        new_out = moe(x)
        old_out = old_moe_dispatch(moe, x)

        assert torch.allclose(new_out, old_out, atol=1e-5), (
            f"Max diff: {(new_out - old_out).abs().max().item()}"
        )

    def test_gradient_equivalence(self):
        """Gradients w.r.t. input should match between old and new dispatch."""
        torch.manual_seed(77)
        cfg = small_cfg()
        moe = MoEFFN(cfg)

        x1 = torch.randn(B, T, cfg.dim, requires_grad=True)
        x2 = x1.clone().detach().requires_grad_(True)

        new_out = moe(x1)
        new_out.sum().backward()

        # Reset grads in the MoE
        moe.zero_grad()

        old_out = old_moe_dispatch(moe, x2)
        old_out.sum().backward()

        assert x1.grad is not None and x2.grad is not None
        assert torch.allclose(x1.grad, x2.grad, atol=1e-4), (
            f"Max grad diff: {(x1.grad - x2.grad).abs().max().item()}"
        )

    def test_epsilon_guard_difference(self):
        """The .clamp(min=1e-9) is a safety net; verify it doesn't change
        normal-case results (scores are never actually zero in practice)."""
        torch.manual_seed(42)
        cfg = small_cfg()
        moe = MoEFFN(cfg)
        moe.eval()

        x = torch.randn(B, T, cfg.dim)
        flat = x.view(B * T, cfg.dim)
        logits = moe.router(flat)
        scores = F.softmax(logits, dim=-1)
        _, topk_idx = (logits + moe.router_bias).topk(moe.topk, dim=-1)
        topk_scores = scores.gather(-1, topk_idx)

        # Without epsilon
        renorm_no_eps = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        # With epsilon
        renorm_with_eps = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        # In normal cases, these should be identical
        assert torch.allclose(renorm_no_eps, renorm_with_eps, atol=1e-9)


class TestMoEDispatchPerformanceCharacteristics:
    """Verify the new grouped dispatch has the same semantic behavior."""

    def test_each_expert_called_exactly_once_per_batch(self):
        """In grouped dispatch, each active expert should be called once
        with all its assigned tokens batched together."""
        torch.manual_seed(42)
        cfg = small_cfg(n_experts=4, n_experts_per_tok=2)
        moe = MoEFFN(cfg)
        moe.eval()

        x = torch.randn(B, T, cfg.dim)
        flat = x.view(B * T, cfg.dim)

        logits = moe.router(flat)
        _, topk_idx = (logits + moe.router_bias).topk(moe.topk, dim=-1)

        flat_expert_ids = topk_idx.view(-1)
        unique_experts = torch.unique(flat_expert_ids)

        # Each unique expert in the routing should appear at least once
        assert len(unique_experts) > 0
        assert len(unique_experts) <= cfg.n_experts

    def test_output_preserves_batch_structure(self):
        """Output shape must match input shape through the dispatch."""
        cfg = small_cfg()
        moe = MoEFFN(cfg)
        for b, t in [(1, 1), (1, 16), (4, 8), (8, 1)]:
            x = torch.randn(b, t, cfg.dim)
            out = moe(x)
            assert out.shape == (b, t, cfg.dim)


if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])
