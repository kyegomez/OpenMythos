"""
Comprehensive component-level tests for every module in open_mythos/main.py.

Covers: RMSNorm, precompute_rope_freqs, apply_rope, GQAttention, MLAttention,
Expert, TransformerBlock, LTIInjection, RecurrentBlock, and OpenMythos.

All tests run on CPU with small configs (dim=64, vocab_size=200, etc.).
"""

import pytest
import torch
import torch.nn as nn

from open_mythos.main import (
    ACTHalting,
    Expert,
    GQAttention,
    LoRAAdapter,
    LTIInjection,
    MLAttention,
    MoEFFN,
    MythosConfig,
    OpenMythos,
    RecurrentBlock,
    RMSNorm,
    TransformerBlock,
    apply_rope,
    loop_index_embedding,
    precompute_rope_freqs,
)

# ---------------------------------------------------------------------------
# Helpers
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


# =====================================================================
# TestRMSNorm
# =====================================================================


class TestRMSNorm:
    """Tests for the RMSNorm layer."""

    def test_output_shape(self):
        """Output matches input shape."""
        norm = RMSNorm(64)
        x = torch.randn(B, T, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization_magnitude(self):
        """Output RMS is approximately 1 (within tolerance)."""
        norm = RMSNorm(64)
        x = torch.randn(B, T, 64) * 10.0  # large scale input
        out = norm(x)
        # With weight=1, the RMS of each output vector should be ~1
        rms = out.float().pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_zero_input(self):
        """Zero input produces zero output."""
        norm = RMSNorm(64)
        x = torch.zeros(B, T, 64)
        out = norm(x)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_gradient_flows(self):
        """Gradients reach the weight parameter."""
        norm = RMSNorm(64)
        x = torch.randn(B, T, 64, requires_grad=True)
        out = norm(x)
        loss = out.sum()
        loss.backward()
        assert norm.weight.grad is not None
        assert norm.weight.grad.abs().sum() > 0
        assert x.grad is not None

    def test_learned_weight_effect(self):
        """Changing weight parameter changes output."""
        norm = RMSNorm(64)
        x = torch.randn(B, T, 64)
        out1 = norm(x).clone()
        # Scale the weight by 2
        with torch.no_grad():
            norm.weight.mul_(2.0)
        out2 = norm(x)
        assert not torch.allclose(out1, out2)
        # Outputs should be in a 2:1 ratio
        ratio = out2 / (out1 + 1e-12)
        assert torch.allclose(ratio[out1.abs() > 1e-6], torch.tensor(2.0), atol=0.01)

    def test_eps_prevents_nan(self):
        """Very small input doesn't produce NaN."""
        norm = RMSNorm(64, eps=1e-6)
        x = torch.full((B, T, 64), 1e-20)
        out = norm(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_preserves_dtype(self):
        """float16 and bfloat16 inputs return same dtype."""
        norm = RMSNorm(64)
        for dtype in [torch.float16, torch.bfloat16]:
            x = torch.randn(B, T, 64, dtype=dtype)
            out = norm(x)
            assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"


# =====================================================================
# TestRoPE
# =====================================================================


class TestRoPE:
    """Tests for precompute_rope_freqs and apply_rope."""

    def test_freqs_shape(self):
        """precompute_rope_freqs returns (max_len, dim//2) complex tensor."""
        dim, max_len = 16, 32
        freqs = precompute_rope_freqs(dim, max_len)
        assert freqs.shape == (max_len, dim // 2)
        assert freqs.is_complex()

    def test_freqs_unit_magnitude(self):
        """All phasors have magnitude 1."""
        freqs = precompute_rope_freqs(16, 32)
        magnitudes = freqs.abs()
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-6)

    def test_freqs_position_zero_identity(self):
        """freqs[0] are all 1+0j (zero rotation)."""
        freqs = precompute_rope_freqs(16, 32)
        expected = torch.ones(8, dtype=torch.complex64)
        assert torch.allclose(freqs[0], expected, atol=1e-6)

    def test_apply_rope_shape_preserved(self):
        """Output shape matches input."""
        dim = 16
        freqs = precompute_rope_freqs(dim, T)
        x = torch.randn(B, T, 4, dim)
        out = apply_rope(x, freqs)
        assert out.shape == x.shape

    def test_apply_rope_norm_preserved(self):
        """RoPE is an isometry (norm doesn't change)."""
        dim = 16
        freqs = precompute_rope_freqs(dim, T)
        x = torch.randn(B, T, 4, dim)
        out = apply_rope(x, freqs)
        norms_in = x.float().norm(dim=-1)
        norms_out = out.float().norm(dim=-1)
        assert torch.allclose(norms_in, norms_out, atol=1e-5)

    def test_apply_rope_position_zero_identity(self):
        """Position 0 doesn't change the tensor."""
        dim = 16
        freqs = precompute_rope_freqs(dim, 1)
        x = torch.randn(B, 1, 4, dim)
        out = apply_rope(x, freqs)
        assert torch.allclose(x, out, atol=1e-6)

    def test_apply_rope_dtype_preserved(self):
        """Preserves float16 and bfloat16."""
        dim = 16
        freqs = precompute_rope_freqs(dim, T)
        for dtype in [torch.float16, torch.bfloat16]:
            x = torch.randn(B, T, 4, dim, dtype=dtype)
            out = apply_rope(x, freqs)
            assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"


# =====================================================================
# TestGQAttention
# =====================================================================


class TestGQAttention:
    """Tests for Grouped Query Attention."""

    @pytest.fixture
    def gqa_setup(self):
        cfg = small_cfg(attn_type="gqa")
        attn = GQAttention(cfg)
        head_dim = cfg.dim // cfg.n_heads
        freqs = precompute_rope_freqs(head_dim, cfg.max_seq_len)
        x = torch.randn(B, T, cfg.dim)
        return attn, cfg, freqs, x

    def test_output_shape(self, gqa_setup):
        """(B, T, dim) output."""
        attn, cfg, freqs, x = gqa_setup
        out = attn(x, freqs[:T])
        assert out.shape == (B, T, cfg.dim)

    def test_forward_no_nan(self, gqa_setup):
        """Standard forward pass has no NaN."""
        attn, cfg, freqs, x = gqa_setup
        out = attn(x, freqs[:T])
        assert not torch.isnan(out).any()

    def test_kv_cache_populates(self, gqa_setup):
        """Passing kv_cache dict gets populated."""
        attn, cfg, freqs, x = gqa_setup
        cache = {}
        attn(x, freqs[:T], kv_cache=cache, cache_key="layer0")
        assert "layer0" in cache
        assert "k" in cache["layer0"]
        assert "v" in cache["layer0"]
        assert cache["layer0"]["k"].shape[1] == T
        assert cache["layer0"]["v"].shape[1] == T

    def test_kv_cache_decode_step(self, gqa_setup):
        """Decode with cache produces correct shape."""
        attn, cfg, freqs, x = gqa_setup
        cache = {}
        # Prefill
        attn(x, freqs[:T], kv_cache=cache, cache_key="layer0")
        # Decode step: single token
        x_decode = torch.randn(B, 1, cfg.dim)
        out = attn(x_decode, freqs[T : T + 1], kv_cache=cache, cache_key="layer0")
        assert out.shape == (B, 1, cfg.dim)
        # Cache should now have T+1 entries
        assert cache["layer0"]["k"].shape[1] == T + 1

    def test_causal_mask_effect(self, gqa_setup):
        """With mask, future tokens don't leak."""
        attn, cfg, freqs, x = gqa_setup
        mask = OpenMythos._causal_mask(T, x.device, x.dtype)
        out_masked = attn(x, freqs[:T], mask=mask)
        out_unmasked = attn(x, freqs[:T], mask=None)
        # Outputs should differ because the mask blocks future tokens
        assert not torch.allclose(out_masked, out_unmasked, atol=1e-5)

    def test_gradient_flows(self, gqa_setup):
        """Gradients reach wq, wk, wv, wo."""
        attn, cfg, freqs, x = gqa_setup
        x = x.requires_grad_(True)
        out = attn(x, freqs[:T])
        loss = out.sum()
        loss.backward()
        for name in ["wq", "wk", "wv", "wo"]:
            param = getattr(attn, name).weight
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_different_n_kv_heads(self):
        """GQA grouping works with different ratios."""
        for n_kv_heads in [1, 2, 4]:
            cfg = small_cfg(attn_type="gqa", n_heads=4, n_kv_heads=n_kv_heads)
            attn = GQAttention(cfg)
            head_dim = cfg.dim // cfg.n_heads
            freqs = precompute_rope_freqs(head_dim, cfg.max_seq_len)
            x = torch.randn(B, T, cfg.dim)
            out = attn(x, freqs[:T])
            assert out.shape == (B, T, cfg.dim)
            assert not torch.isnan(out).any()


# =====================================================================
# TestMLAttention
# =====================================================================


class TestMLAttention:
    """Tests for Multi-Latent Attention."""

    @pytest.fixture
    def mla_setup(self):
        cfg = small_cfg(attn_type="mla")
        attn = MLAttention(cfg)
        freqs = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len)
        x = torch.randn(B, T, cfg.dim)
        return attn, cfg, freqs, x

    def test_output_shape(self, mla_setup):
        """(B, T, dim) output."""
        attn, cfg, freqs, x = mla_setup
        out = attn(x, freqs[:T])
        assert out.shape == (B, T, cfg.dim)

    def test_forward_no_nan(self, mla_setup):
        """Standard forward pass has no NaN."""
        attn, cfg, freqs, x = mla_setup
        out = attn(x, freqs[:T])
        assert not torch.isnan(out).any()

    def test_kv_cache_populates(self, mla_setup):
        """Cache stores c_kv and k_rope (not full K/V)."""
        attn, cfg, freqs, x = mla_setup
        cache = {}
        attn(x, freqs[:T], kv_cache=cache, cache_key="mla0")
        assert "mla0" in cache
        assert "c_kv" in cache["mla0"]
        assert "k_rope" in cache["mla0"]
        # c_kv should have shape (B, T, kv_lora_rank)
        assert cache["mla0"]["c_kv"].shape == (B, T, cfg.kv_lora_rank)

    def test_kv_cache_decode_step(self, mla_setup):
        """Decode step with cache."""
        attn, cfg, freqs, x = mla_setup
        cache = {}
        # Prefill
        attn(x, freqs[:T], kv_cache=cache, cache_key="mla0")
        # Decode
        x_decode = torch.randn(B, 1, cfg.dim)
        out = attn(x_decode, freqs[T : T + 1], kv_cache=cache, cache_key="mla0")
        assert out.shape == (B, 1, cfg.dim)
        assert cache["mla0"]["c_kv"].shape[1] == T + 1

    def test_cache_size_smaller_than_gqa(self):
        """Verify MLA cache is smaller than equivalent GQA cache."""
        cfg = small_cfg(attn_type="mla")
        mla = MLAttention(cfg)
        freqs_mla = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len)
        x = torch.randn(B, T, cfg.dim)

        mla_cache = {}
        mla(x, freqs_mla[:T], kv_cache=mla_cache, cache_key="mla")

        # MLA stores c_kv (B, T, kv_lora_rank) + k_rope (B, T, n_heads, qk_rope_head_dim)
        mla_size = (
            mla_cache["mla"]["c_kv"].numel() + mla_cache["mla"]["k_rope"].numel()
        )

        # Equivalent GQA stores k (B, T, n_kv_heads, head_dim) + v (same)
        cfg_gqa = small_cfg(attn_type="gqa")
        gqa = GQAttention(cfg_gqa)
        head_dim = cfg_gqa.dim // cfg_gqa.n_heads
        freqs_gqa = precompute_rope_freqs(head_dim, cfg_gqa.max_seq_len)

        gqa_cache = {}
        gqa(x, freqs_gqa[:T], kv_cache=gqa_cache, cache_key="gqa")

        gqa_size = gqa_cache["gqa"]["k"].numel() + gqa_cache["gqa"]["v"].numel()

        assert mla_size < gqa_size, (
            f"MLA cache ({mla_size}) should be smaller than GQA cache ({gqa_size})"
        )

    def test_gradient_flows(self, mla_setup):
        """Gradients reach key projections."""
        attn, cfg, freqs, x = mla_setup
        x = x.requires_grad_(True)
        out = attn(x, freqs[:T])
        loss = out.sum()
        loss.backward()
        for name in ["q_down", "q_up_nope", "q_up_rope", "kv_down", "kv_up", "wo"]:
            param = getattr(attn, name).weight
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


# =====================================================================
# TestExpert
# =====================================================================


class TestExpert:
    """Tests for the SwiGLU Expert FFN."""

    def test_output_shape(self):
        """(B, T, dim) output."""
        expert = Expert(64, 16)
        x = torch.randn(B, T, 64)
        out = expert(x)
        assert out.shape == (B, T, 64)

    def test_swiglu_forward(self):
        """Basic forward pass works and is not trivially zero."""
        expert = Expert(64, 16)
        x = torch.randn(B, T, 64)
        out = expert(x)
        assert not torch.isnan(out).any()
        assert out.abs().sum() > 0

    def test_gradient_flows(self):
        """All three weight matrices get gradients."""
        expert = Expert(64, 16)
        x = torch.randn(B, T, 64, requires_grad=True)
        out = expert(x)
        loss = out.sum()
        loss.backward()
        for name in ["gate", "up", "down"]:
            param = getattr(expert, name).weight
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_dtype_alignment(self):
        """float16 input works with float32 params (the FSDP dtype cast)."""
        expert = Expert(64, 16)  # float32 params
        x = torch.randn(B, T, 64, dtype=torch.float16)
        out = expert(x)
        # The expert casts x to param dtype internally, so output is float32
        assert not torch.isnan(out).any()
        assert out.shape == (B, T, 64)


# =====================================================================
# TestTransformerBlock
# =====================================================================


class TestTransformerBlock:
    """Tests for the pre-norm TransformerBlock."""

    def test_output_shape_dense_ffn(self):
        """With use_moe=False."""
        cfg = small_cfg(attn_type="gqa")
        block = TransformerBlock(cfg, use_moe=False)
        head_dim = cfg.dim // cfg.n_heads
        freqs = precompute_rope_freqs(head_dim, cfg.max_seq_len)
        x = torch.randn(B, T, cfg.dim)
        out = block(x, freqs[:T])
        assert out.shape == (B, T, cfg.dim)

    def test_output_shape_moe_ffn(self):
        """With use_moe=True."""
        cfg = small_cfg(attn_type="gqa")
        block = TransformerBlock(cfg, use_moe=True)
        head_dim = cfg.dim // cfg.n_heads
        freqs = precompute_rope_freqs(head_dim, cfg.max_seq_len)
        x = torch.randn(B, T, cfg.dim)
        out = block(x, freqs[:T])
        assert out.shape == (B, T, cfg.dim)

    def test_residual_connection(self):
        """Output is not identical to just FFN(Attn(x)) -- residual adds input."""
        cfg = small_cfg(attn_type="gqa")
        block = TransformerBlock(cfg, use_moe=False)
        head_dim = cfg.dim // cfg.n_heads
        freqs = precompute_rope_freqs(head_dim, cfg.max_seq_len)
        x = torch.randn(B, T, cfg.dim)
        out = block(x, freqs[:T])
        # If there were no residual, the output would be independent of x's exact
        # values in a very different way. Check that out != 0 (non-trivial) and
        # that it is close to x + something (the residual sum pattern).
        diff = out - x
        # The residual connection ensures out != x (attention+FFN output is non-zero)
        assert diff.abs().sum() > 0, "Block should modify input via attention+FFN"
        # But also out should be correlated with x (residual keeps the signal)
        cosine_sim = torch.nn.functional.cosine_similarity(
            out.flatten(), x.flatten(), dim=0
        )
        assert cosine_sim > 0.5, "Residual connection should preserve input signal"

    def test_forward_no_nan(self):
        """Both GQA and MLA modes."""
        for attn_type in ["gqa", "mla"]:
            cfg = small_cfg(attn_type=attn_type)
            block = TransformerBlock(cfg, use_moe=False)
            if attn_type == "mla":
                freqs = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len)
            else:
                head_dim = cfg.dim // cfg.n_heads
                freqs = precompute_rope_freqs(head_dim, cfg.max_seq_len)
            x = torch.randn(B, T, cfg.dim)
            out = block(x, freqs[:T])
            assert not torch.isnan(out).any(), f"NaN in {attn_type} mode"

    def test_gradient_flows(self):
        """Gradients propagate through the block."""
        cfg = small_cfg(attn_type="gqa")
        block = TransformerBlock(cfg, use_moe=False)
        head_dim = cfg.dim // cfg.n_heads
        freqs = precompute_rope_freqs(head_dim, cfg.max_seq_len)
        x = torch.randn(B, T, cfg.dim, requires_grad=True)
        out = block(x, freqs[:T])
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        # Check that attention weights got gradients
        assert block.attn.wq.weight.grad is not None


# =====================================================================
# TestLTIInjection
# =====================================================================


class TestLTIInjection:
    """Tests for the LTI-stable injection module."""

    def test_spectral_radius_below_one(self):
        """get_A() values are all in (0, 1) -- THE key invariant."""
        lti = LTIInjection(64)
        A = lti.get_A()
        assert (A > 0).all(), "A values must be strictly positive"
        assert (A < 1).all(), "A values must be strictly less than 1"

    def test_spectral_radius_extreme_params(self):
        """Even with extreme log_A and log_dt, A stays in [0, 1] and is finite.

        Mathematically A is in the open interval (0, 1), but float32 can round
        to 0.0 when exp(-very_large) underflows, or to 1.0 when exp(-very_small)
        rounds up.  The important guarantee is: A never exceeds 1 and never goes
        negative, so the system is non-explosive (spectral radius <= 1).
        """
        lti = LTIInjection(64)

        # Large positive params -> exp(log_dt + log_A) is huge -> A ~ 0
        with torch.no_grad():
            lti.log_A.fill_(10.0)
            lti.log_dt.fill_(10.0)
        A = lti.get_A()
        assert (A >= 0).all() and (A <= 1).all(), "A out of [0,1] with large params"
        assert torch.isfinite(A).all(), "A not finite with large params"

        # Large negative params -> exp(log_dt + log_A) is tiny -> A ~ 1
        with torch.no_grad():
            lti.log_A.fill_(-10.0)
            lti.log_dt.fill_(-10.0)
        A = lti.get_A()
        assert (A >= 0).all() and (A <= 1).all(), "A out of [0,1] with negative params"
        assert torch.isfinite(A).all(), "A not finite with negative params"

        # Mixed extremes
        with torch.no_grad():
            lti.log_A.fill_(15.0)
            lti.log_dt.fill_(-15.0)
        A = lti.get_A()
        assert (A >= 0).all() and (A <= 1).all(), "A out of [0,1] with mixed params"
        assert torch.isfinite(A).all(), "A not finite with mixed params"

        # Moderate values -> A strictly in (0, 1)
        with torch.no_grad():
            lti.log_A.fill_(0.0)
            lti.log_dt.fill_(0.0)
        A = lti.get_A()
        assert (A > 0).all() and (A < 1).all(), "A out of (0,1) with moderate params"

    def test_forward_shape(self):
        """Output matches input shape."""
        lti = LTIInjection(64)
        h = torch.randn(B, T, 64)
        e = torch.randn(B, T, 64)
        trans_out = torch.randn(B, T, 64)
        out = lti(h, e, trans_out)
        assert out.shape == (B, T, 64)

    def test_stability_many_iterations(self):
        """Iterated application doesn't explode."""
        lti = LTIInjection(64)
        h = torch.randn(B, T, 64)
        e = torch.randn(B, T, 64) * 0.1
        for _ in range(100):
            trans_out = torch.zeros(B, T, 64)
            h = lti(h, e, trans_out)
        assert not torch.isnan(h).any(), "NaN after 100 iterations"
        assert not torch.isinf(h).any(), "Inf after 100 iterations"
        # The state should converge toward a fixed point since A < 1
        h_norm = h.norm()
        assert h_norm < 1e6, f"State norm {h_norm} is too large after 100 steps"

    def test_gradient_flows(self):
        """Gradients reach log_A, log_dt, B."""
        lti = LTIInjection(64)
        h = torch.randn(B, T, 64, requires_grad=True)
        e = torch.randn(B, T, 64)
        trans_out = torch.randn(B, T, 64)
        out = lti(h, e, trans_out)
        loss = out.sum()
        loss.backward()
        assert lti.log_A.grad is not None, "No gradient for log_A"
        assert lti.log_dt.grad is not None, "No gradient for log_dt"
        assert lti.B.grad is not None, "No gradient for B"
        assert lti.log_A.grad.abs().sum() > 0


# =====================================================================
# TestRecurrentBlock
# =====================================================================


class TestRecurrentBlock:
    """Tests for the RecurrentBlock with ACT, LoRA, and LTI."""

    @pytest.fixture
    def recurrent_setup(self):
        cfg = small_cfg(attn_type="gqa")
        block = RecurrentBlock(cfg)
        head_dim = cfg.dim // cfg.n_heads
        freqs = precompute_rope_freqs(head_dim, cfg.max_seq_len)
        h = torch.randn(B, T, cfg.dim)
        e = torch.randn(B, T, cfg.dim)
        mask = OpenMythos._causal_mask(T, h.device, h.dtype)
        return block, cfg, freqs, h, e, mask

    def test_output_shape_gqa(self, recurrent_setup):
        """(B, T, dim) with GQA attention."""
        block, cfg, freqs, h, e, mask = recurrent_setup
        out = block(h, e, freqs[:T], mask)
        assert out.shape == (B, T, cfg.dim)

    def test_output_shape_mla(self):
        """(B, T, dim) with MLA attention."""
        cfg = small_cfg(attn_type="mla")
        block = RecurrentBlock(cfg)
        freqs = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len)
        h = torch.randn(B, T, cfg.dim)
        e = torch.randn(B, T, cfg.dim)
        mask = OpenMythos._causal_mask(T, h.device, h.dtype)
        out = block(h, e, freqs[:T], mask)
        assert out.shape == (B, T, cfg.dim)

    def test_loops_override(self, recurrent_setup):
        """n_loops parameter changes behavior."""
        block, cfg, freqs, h, e, mask = recurrent_setup
        torch.manual_seed(42)
        out_2 = block(h, e, freqs[:T], mask, n_loops=2)
        torch.manual_seed(42)
        out_3 = block(h, e, freqs[:T], mask, n_loops=3)
        # Different number of loops should yield different results
        assert not torch.allclose(out_2, out_3, atol=1e-5)

    def test_act_early_exit_without_cache(self):
        """When halted.all() is true and no cache, the loop breaks early."""
        # Use an extremely low threshold so halting triggers early
        cfg = small_cfg(attn_type="gqa", act_threshold=0.01, max_loop_iters=10)
        block = RecurrentBlock(cfg)
        h = torch.randn(B, T, cfg.dim)
        e = torch.randn(B, T, cfg.dim)
        head_dim = cfg.dim // cfg.n_heads
        freqs = precompute_rope_freqs(head_dim, cfg.max_seq_len)
        mask = OpenMythos._causal_mask(T, h.device, h.dtype)

        # Bias the halting head strongly so sigmoid outputs near 1.0
        with torch.no_grad():
            block.act.halt.weight.fill_(0.0)
            block.act.halt.bias.fill_(10.0)  # sigmoid(10) ~ 1.0

        out = block(h, e, freqs[:T], mask, n_loops=10)
        assert out.shape == (B, T, cfg.dim)
        assert not torch.isnan(out).any()

    def test_kv_cache_no_early_exit(self, recurrent_setup):
        """With cache, loop always runs all iterations (no early exit)."""
        block, cfg, freqs, h, e, mask = recurrent_setup
        cache = {}

        # Bias halting high so it would normally exit early
        with torch.no_grad():
            block.act.halt.weight.fill_(0.0)
            block.act.halt.bias.fill_(10.0)

        out = block(h, e, freqs[:T], mask, n_loops=3, kv_cache=cache)
        assert out.shape == (B, T, cfg.dim)
        # All 3 loop iterations should have created cache entries
        for t in range(3):
            assert f"recurrent_loop_{t}" in cache, (
                f"Cache key for loop {t} missing -- early exit happened with cache"
            )

    def test_gradient_flows(self, recurrent_setup):
        """End-to-end gradient through the recurrent block."""
        block, cfg, freqs, h, e, mask = recurrent_setup
        h = h.requires_grad_(True)
        out = block(h, e, freqs[:T], mask, n_loops=2)
        loss = out.sum()
        loss.backward()
        assert h.grad is not None
        assert h.grad.abs().sum() > 0
        # Check LTI gets gradients
        assert block.injection.log_A.grad is not None
        # Check LoRA gets gradients
        assert block.lora.down.weight.grad is not None

    def test_depth_extrapolation(self, recurrent_setup):
        """n_loops > max_loop_iters works (LoRA clamping)."""
        block, cfg, freqs, h, e, mask = recurrent_setup
        # cfg.max_loop_iters=3, so n_loops=5 exceeds it
        out = block(h, e, freqs[:T], mask, n_loops=5)
        assert out.shape == (B, T, cfg.dim)
        assert not torch.isnan(out).any()


# =====================================================================
# TestOpenMythosModel
# =====================================================================


class TestOpenMythosModel:
    """Tests for the full OpenMythos model."""

    @pytest.fixture
    def gqa_model(self):
        cfg = small_cfg(attn_type="gqa")
        model = OpenMythos(cfg)
        model.eval()
        return model, cfg

    @pytest.fixture
    def mla_model(self):
        cfg = small_cfg(attn_type="mla")
        model = OpenMythos(cfg)
        model.eval()
        return model, cfg

    def test_weight_tying(self, gqa_model):
        """head.weight is embed.weight."""
        model, cfg = gqa_model
        assert model.head.weight is model.embed.weight

    def test_causal_mask_shape(self):
        """_causal_mask returns correct shape."""
        mask = OpenMythos._causal_mask(T, torch.device("cpu"), torch.float32)
        assert mask.shape == (1, 1, T, T)

    def test_causal_mask_values(self):
        """Upper triangle is -inf, lower triangle and diagonal are 0."""
        mask = OpenMythos._causal_mask(T, torch.device("cpu"), torch.float32)
        mask_2d = mask.squeeze(0).squeeze(0)
        # Diagonal and below should be 0
        lower = torch.tril(torch.ones(T, T, dtype=torch.bool))
        assert (mask_2d[lower] == 0.0).all(), "Lower triangle should be 0"
        # Above diagonal should be -inf
        upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        assert (mask_2d[upper] == float("-inf")).all(), "Upper triangle should be -inf"

    def test_attn_type_gqa(self, gqa_model):
        """Model works with attn_type='gqa'."""
        model, cfg = gqa_model
        input_ids = torch.randint(0, cfg.vocab_size, (B, T))
        logits = model(input_ids)
        assert logits.shape == (B, T, cfg.vocab_size)
        assert not torch.isnan(logits).any()

    def test_attn_type_mla(self, mla_model):
        """Model works with attn_type='mla'."""
        model, cfg = mla_model
        input_ids = torch.randint(0, cfg.vocab_size, (B, T))
        logits = model(input_ids)
        assert logits.shape == (B, T, cfg.vocab_size)
        assert not torch.isnan(logits).any()

    def test_generate_basic(self, gqa_model):
        """Generate produces correct shape."""
        model, cfg = gqa_model
        input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
        max_new = 3
        output = model.generate(input_ids, max_new_tokens=max_new, n_loops=2)
        assert output.shape == (1, 4 + max_new)

    def test_generate_temperature(self, gqa_model):
        """Temperature=0.01 is near-greedy (repeated runs are nearly identical)."""
        model, cfg = gqa_model
        input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
        results = []
        for _ in range(3):
            torch.manual_seed(0)
            out = model.generate(
                input_ids.clone(), max_new_tokens=3, n_loops=2, temperature=0.01
            )
            results.append(out)
        # With very low temperature and same seed, all should be identical
        assert torch.equal(results[0], results[1])
        assert torch.equal(results[1], results[2])

    def test_generate_top_k(self, gqa_model):
        """Top_k=1 is deterministic."""
        model, cfg = gqa_model
        input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
        out1 = model.generate(
            input_ids.clone(), max_new_tokens=3, n_loops=2, top_k=1
        )
        out2 = model.generate(
            input_ids.clone(), max_new_tokens=3, n_loops=2, top_k=1
        )
        # top_k=1 forces the argmax token each step, so results must match
        assert torch.equal(out1, out2)

    def test_forward_with_kv_cache(self, gqa_model):
        """Cache-based forward works."""
        model, cfg = gqa_model
        input_ids = torch.randint(0, cfg.vocab_size, (B, T))
        cache = {}
        logits = model(input_ids, kv_cache=cache, start_pos=0)
        assert logits.shape == (B, T, cfg.vocab_size)
        # Cache should be populated
        assert len(cache) > 0

        # Decode step
        next_ids = torch.randint(0, cfg.vocab_size, (B, 1))
        logits_decode = model(next_ids, kv_cache=cache, start_pos=T)
        assert logits_decode.shape == (B, 1, cfg.vocab_size)

    def test_start_pos_affects_rope(self, gqa_model):
        """Different start_pos gives different results when used with KV cache.

        RoPE encodes *relative* positions: without a cache, shifting all Q and K
        by the same offset cancels out in the dot product.  The effect of
        start_pos becomes visible during decode, where cached keys were encoded
        at earlier positions and a new query is encoded at a different offset.
        """
        model, cfg = gqa_model

        prompt = torch.randint(0, cfg.vocab_size, (1, 4))
        next_tok = torch.randint(0, cfg.vocab_size, (1, 1))

        # Path A: prefill at pos 0, decode at pos 4
        cache_a = {}
        model(prompt, kv_cache=cache_a, start_pos=0)
        logits_a = model(next_tok, kv_cache=cache_a, start_pos=4)

        # Path B: prefill at pos 0, decode at pos 10 (wrong position)
        cache_b = {}
        model(prompt, kv_cache=cache_b, start_pos=0)
        logits_b = model(next_tok, kv_cache=cache_b, start_pos=10)

        # The cached keys were encoded at positions 0..3 in both cases, but the
        # query token is encoded at position 4 vs 10, changing the relative
        # distances and therefore the attention weights via RoPE.
        assert not torch.allclose(logits_a, logits_b, atol=1e-4), (
            "Different start_pos during decode should change logits via RoPE "
            "relative position encoding"
        )
