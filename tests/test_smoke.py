"""
End-to-end smoke tests using TINY_CONFIG.

Ported from smoke_test_tiny.py — verifies that:
1. Model instantiates without error
2. Forward pass produces correct output shape (B, T, vocab_size)
3. Spectral radius of A_discrete is strictly < 1
4. Backward pass produces non-zero gradients (backprop is connected)
5. generate() returns the right extended shape
6. Both attn_type="mla" and attn_type="gqa" run without error
"""

import torch
import pytest

from open_mythos.main import OpenMythos, MythosConfig
from open_mythos.variants import TINY_CONFIG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gqa_tiny() -> MythosConfig:
    """GQA variant of TINY_CONFIG with n_kv_heads=2 (groups of 2)."""
    return MythosConfig(
        vocab_size=TINY_CONFIG.vocab_size,
        dim=TINY_CONFIG.dim,
        n_heads=TINY_CONFIG.n_heads,
        n_kv_heads=2,
        max_seq_len=TINY_CONFIG.max_seq_len,
        max_loop_iters=TINY_CONFIG.max_loop_iters,
        prelude_layers=TINY_CONFIG.prelude_layers,
        coda_layers=TINY_CONFIG.coda_layers,
        attn_type="gqa",
        n_experts=TINY_CONFIG.n_experts,
        n_shared_experts=TINY_CONFIG.n_shared_experts,
        n_experts_per_tok=TINY_CONFIG.n_experts_per_tok,
        expert_dim=TINY_CONFIG.expert_dim,
        lora_rank=TINY_CONFIG.lora_rank,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# Smoke tests (MLA variant)
# ---------------------------------------------------------------------------


class TestSmokeMLA:
    def test_instantiation(self) -> None:
        model = OpenMythos(TINY_CONFIG)
        assert isinstance(model, torch.nn.Module)

    def test_parameter_count_nonzero(self) -> None:
        model = OpenMythos(TINY_CONFIG)
        total = sum(p.numel() for p in model.parameters())
        assert total > 0, "Model has no parameters"

    def test_forward_shape(self) -> None:
        model = OpenMythos(TINY_CONFIG)
        model.eval()

        B, T = 2, 16
        ids = torch.randint(0, TINY_CONFIG.vocab_size, (B, T))

        with torch.no_grad():
            logits = model(ids, n_loops=TINY_CONFIG.max_loop_iters)

        expected = (B, T, TINY_CONFIG.vocab_size)
        assert logits.shape == expected, f"Expected {expected}, got {logits.shape}"

    def test_spectral_radius_lt_one(self) -> None:
        model = OpenMythos(TINY_CONFIG)
        A_diag = model.recurrent.injection.get_A()
        rho = A_diag.abs().max().item()
        assert rho < 1.0, f"Spectral radius {rho:.6f} >= 1 — LTI stability violated"

    def test_backward_pass(self) -> None:
        model = OpenMythos(TINY_CONFIG)
        model.train()

        ids = torch.randint(0, TINY_CONFIG.vocab_size, (1, 8))
        logits = model(ids, n_loops=TINY_CONFIG.max_loop_iters)
        loss = logits.mean()
        loss.backward()

        grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters()
            if p.grad is not None
        ) ** 0.5

        assert grad_norm > 0, "Backward pass produced zero gradients — graph is disconnected"

    def test_generate_shape(self) -> None:
        model = OpenMythos(TINY_CONFIG)
        model.eval()

        B, T_prompt = 1, 4
        max_new = 6
        ids = torch.randint(0, TINY_CONFIG.vocab_size, (B, T_prompt))

        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, n_loops=TINY_CONFIG.max_loop_iters)

        assert out.shape == (B, T_prompt + max_new), (
            f"Expected ({B}, {T_prompt + max_new}), got {out.shape}"
        )

    def test_output_finite(self) -> None:
        model = OpenMythos(TINY_CONFIG)
        model.eval()

        ids = torch.randint(0, TINY_CONFIG.vocab_size, (1, 16))
        with torch.no_grad():
            logits = model(ids, n_loops=TINY_CONFIG.max_loop_iters)

        assert torch.isfinite(logits).all(), "Forward pass output contains inf or NaN"


# ---------------------------------------------------------------------------
# GQA variant smoke test
# ---------------------------------------------------------------------------


class TestSmokeGQA:
    def test_forward_shape_gqa(self) -> None:
        cfg = _make_gqa_tiny()
        model = OpenMythos(cfg)
        model.eval()

        B, T = 1, 10
        ids = torch.randint(0, cfg.vocab_size, (B, T))

        with torch.no_grad():
            logits = model(ids, n_loops=cfg.max_loop_iters)

        assert logits.shape == (B, T, cfg.vocab_size)

    def test_spectral_radius_gqa(self) -> None:
        cfg = _make_gqa_tiny()
        model = OpenMythos(cfg)
        A_diag = model.recurrent.injection.get_A()
        rho = A_diag.abs().max().item()
        assert rho < 1.0, f"GQA model spectral radius {rho:.6f} >= 1"
