"""
Unit tests for the MiniMax-M2.7 OpenMythos configuration.

All tests run on CPU using synthetic tensors — no model weights are required.
The MiniMax-M2.7 config (minimax_m2_config) is instantiated with a scaled-down
hidden dim so the test suite runs quickly; a separate class validates the full
architecture dimensions without actually allocating the model.
"""

from __future__ import annotations

import pytest
import torch

from open_mythos.main import (
    MythosConfig,
    OpenMythos,
    precompute_rope_freqs,
)
from open_mythos.tokenizer import MINIMAX_M2_MODEL_ID
from open_mythos.variants import minimax_m2_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B, T = 2, 8  # batch size and sequence length used in forward-pass tests


def _small_m2_cfg(**overrides) -> MythosConfig:
    """Return a tiny MiniMax-M2.7-shaped config suitable for CPU tests.

    The structural ratios are preserved:
    - GQA ratio 6:1 (n_heads / n_kv_heads = 48 / 8)
    - MLA with rope+nope split totalling head_dim = 128
    - 32 routed experts + 2 shared, top-4 activated
    Only dim, expert_dim, lora_rank, and vocab_size are shrunk so the test
    suite can allocate and run the model on a laptop CPU.
    """
    base = dict(
        vocab_size=512,
        dim=192,  # divisible by n_heads=48 → head_dim=4 per head
        n_heads=48,
        n_kv_heads=8,
        max_seq_len=64,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        attn_type="mla",
        # Scale MLA dims proportionally to dim=192 while preserving the
        # rope/nope/v split ratios from the full config (64/64/128 scaled ÷32).
        kv_lora_rank=16,
        q_lora_rank=48,
        qk_rope_head_dim=2,
        qk_nope_head_dim=2,
        v_head_dim=4,
        n_experts=32,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=64,
        act_threshold=0.99,
        rope_theta=10_000_000.0,
        lora_rank=4,
        max_output_tokens=64,
    )
    base.update(overrides)
    return MythosConfig(**base)


# ---------------------------------------------------------------------------
# 1. Config correctness — full-scale MiniMax-M2.7 dimensions
# ---------------------------------------------------------------------------


class TestMinimaxM2ConfigDimensions:
    """Validate the full-scale minimax_m2_config() without allocating the model."""

    def setup_method(self):
        self.cfg = minimax_m2_config()

    def test_vocab_size(self):
        assert self.cfg.vocab_size == 200_064

    def test_hidden_dim(self):
        assert self.cfg.dim == 6144

    def test_query_heads(self):
        assert self.cfg.n_heads == 48

    def test_kv_heads(self):
        assert self.cfg.n_kv_heads == 8

    def test_gqa_ratio_6x(self):
        """MiniMax-M2.7 uses a 6:1 GQA ratio."""
        assert self.cfg.n_heads // self.cfg.n_kv_heads == 6

    def test_head_dim_128(self):
        """Q/K attention head dim = rope + nope = 128 to match MiniMax-M2.7."""
        head_dim = self.cfg.qk_rope_head_dim + self.cfg.qk_nope_head_dim
        assert head_dim == 128

    def test_value_head_dim_128(self):
        assert self.cfg.v_head_dim == 128

    def test_mla_output_matches_dim(self):
        """n_heads × v_head_dim must equal dim so the output projection is square."""
        assert self.cfg.n_heads * self.cfg.v_head_dim == self.cfg.dim

    def test_num_routed_experts(self):
        assert self.cfg.n_experts == 32

    def test_num_shared_experts(self):
        assert self.cfg.n_shared_experts == 2

    def test_experts_per_token(self):
        assert self.cfg.n_experts_per_tok == 4

    def test_context_length(self):
        """MiniMax-M2.7 supports a 40 960-token native context window."""
        assert self.cfg.max_seq_len == 40_960

    def test_rope_theta(self):
        """High RoPE base for long-context stability."""
        assert self.cfg.rope_theta == 10_000_000.0

    def test_attention_type_is_mla(self):
        assert self.cfg.attn_type == "mla"

    def test_prelude_and_coda_layers(self):
        assert self.cfg.prelude_layers == 4
        assert self.cfg.coda_layers == 4


# ---------------------------------------------------------------------------
# 2. MINIMAX_M2_MODEL_ID constant
# ---------------------------------------------------------------------------


class TestMinimaxM2ModelID:
    def test_constant_is_string(self):
        assert isinstance(MINIMAX_M2_MODEL_ID, str)

    def test_constant_points_to_minimax_org(self):
        assert MINIMAX_M2_MODEL_ID.startswith("MiniMaxAI/")

    def test_constant_contains_m2(self):
        assert "M2" in MINIMAX_M2_MODEL_ID

    def test_constant_importable_from_package(self):
        """Ensure top-level package re-exports the constant."""
        import open_mythos as om

        assert om.MINIMAX_M2_MODEL_ID == MINIMAX_M2_MODEL_ID


# ---------------------------------------------------------------------------
# 3. minimax_m2_config importable from top-level package
# ---------------------------------------------------------------------------


class TestMinimaxM2ConfigExport:
    def test_importable_from_package(self):
        import open_mythos as om

        cfg = om.minimax_m2_config()
        assert isinstance(cfg, MythosConfig)

    def test_returns_new_instance_each_call(self):
        cfg1 = minimax_m2_config()
        cfg2 = minimax_m2_config()
        assert cfg1 is not cfg2
        assert cfg1.dim == cfg2.dim


# ---------------------------------------------------------------------------
# 4. Small-scale forward-pass tests (CPU, synthetic tensors)
# ---------------------------------------------------------------------------


class TestMinimaxM2ForwardPass:
    """Instantiate a tiny MiniMax-M2.7-shaped model and verify forward passes."""

    def setup_method(self):
        self.cfg = _small_m2_cfg()
        self.model = OpenMythos(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (B, T))

    def test_forward_output_shape(self):
        logits = self.model(self.ids)
        assert logits.shape == (B, T, self.cfg.vocab_size)

    def test_forward_no_nan(self):
        logits = self.model(self.ids)
        assert not torch.isnan(logits).any()

    def test_forward_no_inf(self):
        logits = self.model(self.ids)
        assert torch.isfinite(logits).all()

    def test_generate_shape(self):
        out = self.model.generate(self.ids, max_new_tokens=4, n_loops=2)
        assert out.shape == (B, T + 4)

    def test_lti_spectral_radius_stable(self):
        A = self.model.recurrent.injection.get_A()
        assert A.max().item() < 1.0

    def test_mla_cache_is_compressed(self):
        cache = {}
        with torch.no_grad():
            self.model(self.ids, kv_cache=cache)
        mla_entries = {k: v for k, v in cache.items() if "c_kv" in v}
        assert len(mla_entries) > 0
        for entry in mla_entries.values():
            assert entry["c_kv"].shape[-1] == self.cfg.kv_lora_rank

    def test_deeper_loops_change_output(self):
        ids = self.ids
        with torch.no_grad():
            out1 = self.model(ids, n_loops=1)
            out4 = self.model(ids, n_loops=4)
        assert not torch.allclose(out1, out4)

    def test_weight_tying(self):
        """Embedding and output projection must share weights."""
        assert self.model.head.weight is self.model.embed.weight


# ---------------------------------------------------------------------------
# 5. Attention head geometry — GQA 6:1 ratio preserved at small scale
# ---------------------------------------------------------------------------


class TestMinimaxM2AttentionGeometry:
    def setup_method(self):
        self.cfg = _small_m2_cfg()

    def test_gqa_ratio_preserved(self):
        assert self.cfg.n_heads // self.cfg.n_kv_heads == 6

    def test_mla_output_dim_consistent(self):
        """n_heads × v_head_dim must equal dim even at small scale."""
        assert self.cfg.n_heads * self.cfg.v_head_dim == self.cfg.dim

    def test_rope_freqs_shape_for_qk_rope_dim(self):
        freqs = precompute_rope_freqs(
            dim=self.cfg.qk_rope_head_dim, max_len=self.cfg.max_seq_len
        )
        assert freqs.shape == (self.cfg.max_seq_len, self.cfg.qk_rope_head_dim // 2)


# ---------------------------------------------------------------------------
# 6. MoE: 32 routed + 2 shared, top-4 activation
# ---------------------------------------------------------------------------


class TestMinimaxM2MoE:
    def setup_method(self):
        self.cfg = _small_m2_cfg()
        self.model = OpenMythos(self.cfg)
        # MoE FFN lives inside the recurrent block's inner TransformerBlock
        self.moe = self.model.recurrent.block.ffn

    def test_n_routed_experts(self):
        assert self.moe.n_experts == 32

    def test_n_shared_experts(self):
        assert self.moe.n_shared == 2

    def test_topk_experts(self):
        assert self.moe.topk == 4

    def test_moe_output_shape(self):
        x = torch.randn(B, T, self.cfg.dim)
        out = self.moe(x)
        assert out.shape == (B, T, self.cfg.dim)

    def test_shared_experts_always_fire(self):
        """Zero out all routed experts; shared experts keep output nonzero."""
        for exp in self.moe.routed_experts:
            for p in exp.parameters():
                p.data.zero_()
        x = torch.randn(B, T, self.cfg.dim)
        out = self.moe(x)
        assert out.abs().sum() > 0


# ---------------------------------------------------------------------------
# 7. High rope_theta does not introduce NaN / Inf
# ---------------------------------------------------------------------------


class TestMinimaxM2RopeTheta:
    def test_high_theta_rope_freqs_finite(self):
        cfg = _small_m2_cfg()
        freqs = precompute_rope_freqs(
            dim=cfg.qk_rope_head_dim,
            max_len=cfg.max_seq_len,
            theta=cfg.rope_theta,
        )
        assert torch.isfinite(freqs.real).all()
        assert torch.isfinite(freqs.imag).all()

    def test_high_theta_model_forward_finite(self):
        cfg = _small_m2_cfg()
        model = OpenMythos(cfg)
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        with torch.no_grad():
            out = model(ids)
        assert torch.isfinite(out).all()


if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])
