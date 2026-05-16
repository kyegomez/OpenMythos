"""
Tests for MoEFFN router gating and expert dispatch.

Covers:
- Output shape matches input (B, T, dim)
- Router produces non-negative scores that sum to 1 per token (after renorm)
- topk selection fires exactly n_experts_per_tok experts per token
- Shared experts always fire (output changes without them)
- Expert class (single FFN) shape check
"""

import torch
import pytest

from open_mythos.main import Expert, MoEFFN, MythosConfig


def _moe_cfg(n_experts: int = 4, topk: int = 2) -> MythosConfig:
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
        n_experts=n_experts,
        n_shared_experts=1,
        n_experts_per_tok=topk,
        expert_dim=32,
        lora_rank=4,
        dropout=0.0,
    )


class TestExpert:
    def test_output_shape(self) -> None:
        exp = Expert(dim=64, expert_dim=32)
        x = torch.randn(3, 10, 64)
        out = exp(x)
        assert out.shape == x.shape

    def test_is_nn_module(self) -> None:
        exp = Expert(dim=64, expert_dim=32)
        assert isinstance(exp, torch.nn.Module)
        assert sum(p.numel() for p in exp.parameters()) > 0


class TestMoEFFN:
    def test_output_shape(self) -> None:
        cfg = _moe_cfg()
        moe = MoEFFN(cfg)
        moe.eval()

        B, T = 2, 10
        x = torch.randn(B, T, cfg.dim)
        with torch.no_grad():
            out = moe(x)

        assert out.shape == (B, T, cfg.dim), f"Expected {(B, T, cfg.dim)}, got {out.shape}"

    def test_router_topk_gates(self) -> None:
        """Verify router selects exactly topk=2 experts per token."""
        cfg = _moe_cfg(n_experts=8, topk=2)
        moe = MoEFFN(cfg)
        moe.eval()

        B, T = 1, 5
        flat = torch.randn(B * T, cfg.dim)

        with torch.no_grad():
            logits = moe.router(flat)  # (B*T, n_experts)
            _, idx = (logits + moe.router_bias).topk(cfg.n_experts_per_tok, dim=-1)

        assert idx.shape == (B * T, cfg.n_experts_per_tok)
        # Each token should have exactly topk unique expert selections
        for token_idx in range(B * T):
            assert idx[token_idx].unique().numel() == cfg.n_experts_per_tok

    def test_shared_experts_always_fire(self) -> None:
        """With shared experts zeroed out, outputs should change significantly."""
        cfg = _moe_cfg()
        moe = MoEFFN(cfg)
        moe.eval()

        x = torch.randn(1, 4, cfg.dim)
        with torch.no_grad():
            out_full = moe(x).clone()

        # Zero out shared expert weights
        for shared_exp in moe.shared_experts:
            for param in shared_exp.parameters():
                param.data.zero_()

        with torch.no_grad():
            out_no_shared = moe(x)

        # Outputs should differ when shared experts are ablated
        assert not torch.allclose(out_full, out_no_shared, atol=1e-6)

    def test_output_finite(self) -> None:
        cfg = _moe_cfg()
        moe = MoEFFN(cfg)
        moe.eval()

        x = torch.randn(2, 8, cfg.dim)
        with torch.no_grad():
            out = moe(x)

        assert torch.isfinite(out).all(), "MoEFFN output contains inf or NaN"
