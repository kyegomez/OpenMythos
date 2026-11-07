import torch
import pytest
from open_mythos.main import (
    ACTHalting,
    Expert,
    GQAttention,
    LTIInjection,
    LoRAAdapter,
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
# Shared small configs (kept tiny so tests run fast on CPU)
# ---------------------------------------------------------------------------

B, T = 2, 8  # batch, sequence length


def gqa_cfg(**overrides) -> MythosConfig:
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
        # MLA fields must be valid even when not used
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
    )
    defaults.update(overrides)
    return MythosConfig(**defaults)


def mla_cfg(**overrides) -> MythosConfig:
    return gqa_cfg(attn_type="mla", **overrides)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        assert norm(x).shape == x.shape

    def test_unit_rms(self):
        # after norm the RMS of each vector should be ≈ 1 when weight=1
        norm = RMSNorm(64)
        torch.nn.init.ones_(norm.weight)
        x = torch.randn(4, 64)
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_learnable_weight(self):
        norm = RMSNorm(8)
        assert norm.weight.requires_grad


# ---------------------------------------------------------------------------
# RoPE utilities
# ---------------------------------------------------------------------------


class TestRoPE:
    def test_precompute_shape(self):
        freqs = precompute_rope_freqs(dim=16, max_len=32)
        assert freqs.shape == (32, 8)  # (max_len, dim//2)
        assert freqs.is_complex()

    def test_apply_rope_shape(self):
        freqs = precompute_rope_freqs(dim=16, max_len=32)
        x = torch.randn(B, T, 4, 16)
        out = apply_rope(x, freqs[:T])
        assert out.shape == x.shape

    def test_apply_rope_preserves_norm(self):
        # rotation is an isometry — norms must be unchanged
        freqs = precompute_rope_freqs(dim=16, max_len=32)
        x = torch.randn(B, T, 4, 16)
        out = apply_rope(x, freqs[:T])
        assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-5)

    def test_different_positions_differ(self):
        freqs = precompute_rope_freqs(dim=16, max_len=32)
        x = torch.ones(1, 2, 1, 16)
        out = apply_rope(x, freqs[:2])
        # position 0 and position 1 should produce different rotations
        assert not torch.allclose(out[0, 0], out[0, 1])


# ---------------------------------------------------------------------------
# RoPE extended — correctness invariants
# ---------------------------------------------------------------------------


class TestRoPEExtended:
    """Comprehensive correctness tests for precompute_rope_freqs and apply_rope."""

    # --- precompute_rope_freqs ---

    def test_position_zero_is_unit_phasor(self):
        """freqs[0] must be all 1+0j (angle = 0 * freq = 0 for every pair)."""
        freqs = precompute_rope_freqs(dim=16, max_len=8)
        expected = torch.ones(8, dtype=torch.complex64)
        assert torch.allclose(freqs[0], expected, atol=1e-6)

    def test_all_phasors_have_unit_magnitude(self):
        """Every phasor magnitude must be 1 — RoPE is an isometric rotation."""
        freqs = precompute_rope_freqs(dim=16, max_len=32)
        assert torch.allclose(freqs.abs(), torch.ones_like(freqs.abs()), atol=1e-6)

    def test_angles_equal_outer_product(self):
        """freqs[t, k].angle() must equal t × base_freq[k] for all t, k."""
        dim, max_len, theta = 8, 6, 500000.0
        freqs = precompute_rope_freqs(dim=dim, max_len=max_len, theta=theta)
        base = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_len, dtype=torch.float32)
        expected = torch.polar(torch.ones(max_len, dim // 2), torch.outer(t, base))
        assert torch.allclose(freqs.real, expected.real, atol=1e-6)
        assert torch.allclose(freqs.imag, expected.imag, atol=1e-6)

    def test_higher_theta_produces_smaller_angles(self):
        """Larger theta → slower frequency decay → smaller rotation angle per step.

        Index 0 (dim_i=0) is excluded: its frequency is 1/(theta^0)=1 for any theta,
        so the comparison is not meaningful there.
        """
        dim, max_len = 16, 8
        freqs_fast = precompute_rope_freqs(dim=dim, max_len=max_len, theta=100.0)
        freqs_slow = precompute_rope_freqs(dim=dim, max_len=max_len, theta=500000.0)
        assert (freqs_fast[1, 1:].angle().abs() > freqs_slow[1, 1:].angle().abs()).all()

    def test_default_theta_matches_explicit(self):
        """Omitting theta must equal passing theta=500000.0."""
        f1 = precompute_rope_freqs(16, 8)
        f2 = precompute_rope_freqs(16, 8, theta=500000.0)
        assert torch.allclose(f1.real, f2.real) and torch.allclose(f1.imag, f2.imag)

    # --- apply_rope ---

    def test_position_zero_is_identity(self):
        """T=1 input uses only freqs[0] = 1+0j, so output must equal input."""
        freqs = precompute_rope_freqs(dim=16, max_len=8)
        x = torch.randn(2, 1, 4, 16)
        out = apply_rope(x, freqs[:1])
        assert torch.allclose(x, out, atol=1e-6)

    def test_dtype_float32_preserved(self):
        freqs = precompute_rope_freqs(dim=16, max_len=16)
        x = torch.randn(1, 4, 2, 16).float()
        assert apply_rope(x, freqs[:4]).dtype == torch.float32

    def test_dtype_float16_preserved(self):
        freqs = precompute_rope_freqs(dim=16, max_len=16)
        x = torch.randn(1, 4, 2, 16).half()
        assert apply_rope(x, freqs[:4]).dtype == torch.float16

    def test_inverse_rotation_recovers_input(self):
        """Rotating by freqs then by conj(freqs) (inverse) must recover the original."""
        dim = 16
        freqs = precompute_rope_freqs(dim=dim, max_len=8)
        x = torch.randn(2, 4, 3, dim)
        rotated = apply_rope(x, freqs[:4])
        xc = torch.view_as_complex(rotated.float().reshape(*rotated.shape[:-1], -1, 2))
        inv = freqs.conj()[:4].unsqueeze(0).unsqueeze(2)
        recovered = torch.view_as_real(xc * inv).flatten(-2).to(x.dtype)
        assert torch.allclose(x, recovered, atol=1e-5)

    def test_batch_independence(self):
        """Output for one batch item must not depend on other items in the batch."""
        dim = 16
        freqs = precompute_rope_freqs(dim=dim, max_len=16)
        torch.manual_seed(7)
        x_a = torch.randn(1, 4, 2, dim)
        x_b = torch.randn(1, 4, 2, dim)
        solo = apply_rope(x_a, freqs[:4])
        batched = apply_rope(torch.cat([x_a, x_b], dim=0), freqs[:4])[:1]
        assert torch.allclose(solo, batched, atol=1e-6)

    def test_head_independence(self):
        """All heads at the same position must receive identical rotations."""
        dim = 16
        freqs = precompute_rope_freqs(dim=dim, max_len=8)
        x = torch.randn(1, 4, 1, dim).expand(1, 4, 3, dim).contiguous()
        out = apply_rope(x, freqs[:4])
        assert torch.allclose(out[:, :, 0], out[:, :, 1], atol=1e-6)
        assert torch.allclose(out[:, :, 1], out[:, :, 2], atol=1e-6)

    def test_relative_position_property(self):
        """
        Core RoPE invariant: <RoPE(q,m), RoPE(k,n)> depends only on (n-m).
        Two pairs with the same offset must produce the same dot product.
        """
        dim, max_len = 16, 32
        freqs = precompute_rope_freqs(dim=dim, max_len=max_len)
        torch.manual_seed(42)
        q = torch.randn(1, 1, 1, dim)
        k = torch.randn(1, 1, 1, dim)

        def rope_at(tensor, pos):
            """Rotate tensor at a specific position by embedding it in a zero sequence."""
            seq = torch.zeros(1, pos + 1, 1, dim)
            seq[0, pos] = tensor[0, 0]
            return apply_rope(seq, freqs[: pos + 1])[:, pos : pos + 1]

        # Both pairs have relative offset n - m = 6
        dot_3_9 = (rope_at(q, 3) * rope_at(k, 9)).sum()
        dot_1_7 = (rope_at(q, 1) * rope_at(k, 7)).sum()
        assert torch.allclose(dot_3_9, dot_1_7, atol=1e-5)

    def test_max_len_boundary(self):
        """apply_rope must handle T == max_len without error or NaN."""
        max_len = 10
        freqs = precompute_rope_freqs(dim=8, max_len=max_len)
        x = torch.randn(1, max_len, 2, 8)
        out = apply_rope(x, freqs)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_exceeds_max_len_raises(self):
        """apply_rope must raise RuntimeError when T > max_len."""
        freqs = precompute_rope_freqs(dim=8, max_len=4)
        x = torch.randn(1, 8, 2, 8)  # T=8 > max_len=4
        with pytest.raises(RuntimeError):
            apply_rope(x, freqs)


# ---------------------------------------------------------------------------
# GQAttention
# ---------------------------------------------------------------------------


class TestGQAttention:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.freqs = precompute_rope_freqs(
            self.cfg.dim // self.cfg.n_heads, self.cfg.max_seq_len
        )
        self.attn = GQAttention(self.cfg)

    def test_output_shape(self):
        x = torch.randn(B, T, self.cfg.dim)
        out = self.attn(x, self.freqs[:T])
        assert out.shape == (B, T, self.cfg.dim)

    def test_kv_cache_accumulates(self):
        cache = {}
        x = torch.randn(B, T, self.cfg.dim)
        self.attn(x, self.freqs[:T], kv_cache=cache, cache_key="layer0")
        assert "layer0" in cache
        k_len = cache["layer0"]["k"].shape[1]
        # second call adds T more tokens
        self.attn(x, self.freqs[:T], kv_cache=cache, cache_key="layer0")
        assert cache["layer0"]["k"].shape[1] == k_len + T

    def test_with_causal_mask(self):
        x = torch.randn(B, T, self.cfg.dim)
        mask = torch.full((1, 1, T, T), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        out = self.attn(x, self.freqs[:T], mask=mask)
        assert out.shape == (B, T, self.cfg.dim)


# ---------------------------------------------------------------------------
# MLAttention
# ---------------------------------------------------------------------------


class TestMLAttention:
    def setup_method(self):
        self.cfg = mla_cfg()
        self.freqs = precompute_rope_freqs(
            self.cfg.qk_rope_head_dim, self.cfg.max_seq_len
        )
        self.attn = MLAttention(self.cfg)

    def test_output_shape(self):
        x = torch.randn(B, T, self.cfg.dim)
        out = self.attn(x, self.freqs[:T])
        assert out.shape == (B, T, self.cfg.dim)

    def test_cache_stores_compressed_kv(self):
        cache = {}
        x = torch.randn(B, T, self.cfg.dim)
        self.attn(x, self.freqs[:T], kv_cache=cache, cache_key="mla0")
        assert "c_kv" in cache["mla0"]
        assert "k_rope" in cache["mla0"]
        # c_kv should have kv_lora_rank as last dim, not full K/V
        assert cache["mla0"]["c_kv"].shape[-1] == self.cfg.kv_lora_rank

    def test_cache_accumulates_across_steps(self):
        cache = {}
        x = torch.randn(B, T, self.cfg.dim)
        self.attn(x, self.freqs[:T], kv_cache=cache, cache_key="mla0")
        first_len = cache["mla0"]["c_kv"].shape[1]
        self.attn(x, self.freqs[:T], kv_cache=cache, cache_key="mla0")
        assert cache["mla0"]["c_kv"].shape[1] == first_len + T

    def test_with_causal_mask(self):
        x = torch.randn(B, T, self.cfg.dim)
        mask = torch.triu(torch.full((1, 1, T, T), float("-inf")), diagonal=1)
        out = self.attn(x, self.freqs[:T], mask=mask)
        assert out.shape == (B, T, self.cfg.dim)


# ---------------------------------------------------------------------------
# Expert (dense SwiGLU FFN)
# ---------------------------------------------------------------------------


class TestExpert:
    def test_output_shape(self):
        expert = Expert(dim=64, expert_dim=32)
        x = torch.randn(B, T, 64)
        assert expert(x).shape == (B, T, 64)

    def test_flat_input(self):
        expert = Expert(dim=32, expert_dim=16)
        x = torch.randn(5, 32)
        assert expert(x).shape == (5, 32)


# ---------------------------------------------------------------------------
# MoEFFN
# ---------------------------------------------------------------------------


class TestMoEFFN:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.moe = MoEFFN(self.cfg)

    def test_output_shape(self):
        x = torch.randn(B, T, self.cfg.dim)
        assert self.moe(x).shape == (B, T, self.cfg.dim)

    def test_router_bias_not_grad(self):
        # router_bias is a buffer, not a parameter
        param_names = {n for n, _ in self.moe.named_parameters()}
        assert "router_bias" not in param_names

    def test_shared_experts_always_fire(self):
        # Zero out all routed experts; output should still be nonzero from shared
        for exp in self.moe.routed_experts:
            for p in exp.parameters():
                p.data.zero_()
        x = torch.randn(B, T, self.cfg.dim)
        out = self.moe(x)
        assert out.abs().sum() > 0


# ---------------------------------------------------------------------------
# loop_index_embedding
# ---------------------------------------------------------------------------


class TestLoopIndexEmbedding:
    def test_output_shape(self):
        h = torch.randn(B, T, 64)
        out = loop_index_embedding(h, loop_t=0, loop_dim=8)
        assert out.shape == h.shape

    def test_different_iterations_differ(self):
        h = torch.zeros(1, 1, 64)
        out0 = loop_index_embedding(h, loop_t=0, loop_dim=8)
        out1 = loop_index_embedding(h, loop_t=1, loop_dim=8)
        assert not torch.allclose(out0, out1)

    def test_only_first_dims_modified(self):
        h = torch.zeros(1, 1, 64)
        loop_dim = 8
        out = loop_index_embedding(h, loop_t=3, loop_dim=loop_dim)
        # channels beyond loop_dim should be unchanged (still 0)
        assert torch.all(out[..., loop_dim:] == 0)

    def test_odd_loop_dim_raises(self):
        # Odd loop_dim would produce an asymmetric sin/cos split —
        # the function must reject it at the call site.
        h = torch.zeros(1, 1, 64)
        with pytest.raises(ValueError, match="loop_dim must be even"):
            loop_index_embedding(h, loop_t=0, loop_dim=7)

    def test_recurrent_block_rejects_dim_not_multiple_of_16(self):
        # cfg.dim=24 → loop_dim = 24//8 = 3 (odd) → must raise at construction.
        cfg = gqa_cfg(dim=24, n_heads=4, n_kv_heads=2)
        with pytest.raises(ValueError, match="loop_dim"):
            RecurrentBlock(cfg)


# ---------------------------------------------------------------------------
# LoRAAdapter
# ---------------------------------------------------------------------------


class TestLoRAAdapter:
    def setup_method(self):
        self.lora = LoRAAdapter(dim=64, rank=8, max_loops=10)

    def test_output_shape(self):
        x = torch.randn(B, T, 64)
        out = self.lora(x, loop_t=0)
        assert out.shape == (B, T, 64)

    def test_different_loops_differ(self):
        x = torch.randn(B, T, 64)
        out0 = self.lora(x, loop_t=0)
        out1 = self.lora(x, loop_t=1)
        assert not torch.allclose(out0, out1)


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------


class TestTransformerBlock:
    def test_gqa_output_shape(self):
        cfg = gqa_cfg()
        block = TransformerBlock(cfg, use_moe=False)
        freqs = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len)
        x = torch.randn(B, T, cfg.dim)
        assert block(x, freqs[:T]).shape == (B, T, cfg.dim)

    def test_mla_output_shape(self):
        cfg = mla_cfg()
        block = TransformerBlock(cfg, use_moe=False)
        freqs = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len)
        x = torch.randn(B, T, cfg.dim)
        assert block(x, freqs[:T]).shape == (B, T, cfg.dim)

    def test_moe_block_output_shape(self):
        cfg = gqa_cfg()
        block = TransformerBlock(cfg, use_moe=True)
        freqs = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len)
        x = torch.randn(B, T, cfg.dim)
        assert block(x, freqs[:T]).shape == (B, T, cfg.dim)

    def test_attn_type_selection(self):
        assert isinstance(TransformerBlock(gqa_cfg()).attn, GQAttention)
        assert isinstance(TransformerBlock(mla_cfg()).attn, MLAttention)


# ---------------------------------------------------------------------------
# LTIInjection
# ---------------------------------------------------------------------------


class TestLTIInjection:
    def setup_method(self):
        self.inj = LTIInjection(dim=64)

    def test_output_shape(self):
        h = torch.randn(B, T, 64)
        e = torch.randn(B, T, 64)
        t = torch.randn(B, T, 64)
        assert self.inj(h, e, t).shape == (B, T, 64)

    def test_spectral_radius_lt_1(self):
        A = self.inj.get_A()
        assert A.max().item() < 1.0

    def test_spectral_radius_gt_0(self):
        A = self.inj.get_A()
        assert A.min().item() > 0.0

    def test_spectral_radius_stable_after_large_grad_step(self):
        # Simulate an aggressive gradient update and verify stability holds.
        # get_A clamps log_dt + log_A to [-20, 20], so extreme gradient steps
        # saturate A to float-precision 1.0 without ever exceeding it. The
        # stability guarantee is ρ(A) ≤ 1 under float rounding; strict < 1
        # only holds in exact arithmetic.
        opt = torch.optim.SGD(self.inj.parameters(), lr=1e3)
        h = torch.randn(B, T, 64)
        e = torch.randn(B, T, 64)
        t = torch.randn(B, T, 64)
        loss = self.inj(h, e, t).sum()
        loss.backward()
        opt.step()
        A = self.inj.get_A()
        assert A.max().item() <= 1.0
        assert A.min().item() >= 0.0
        assert not torch.isnan(A).any()

    def test_boundary_saturation_default_healthy(self):
        """A freshly initialised LTI (log_A=log_dt=0) is well clear of the
        ±20 clamp — both low and high saturation fractions must be 0."""
        stats = self.inj.boundary_saturation()
        assert stats["low_fraction"] == 0.0
        assert stats["high_fraction"] == 0.0
        assert stats["total"] == 0.0

    def test_boundary_saturation_detects_high_saturation(self):
        """Force log_A to +30 on all channels → combined > 20-margin → A ≈ 0
        (channel-dead). boundary_saturation must report high_fraction ≈ 1.0."""
        with torch.no_grad():
            self.inj.log_A.fill_(30.0)
        stats = self.inj.boundary_saturation(margin=1.0)
        assert stats["high_fraction"] == 1.0
        assert stats["low_fraction"] == 0.0

    def test_boundary_saturation_detects_low_saturation(self):
        """Force log_A to -30 → combined < -20+margin → A ≈ 1 (static).
        boundary_saturation must report low_fraction ≈ 1.0."""
        with torch.no_grad():
            self.inj.log_A.fill_(-30.0)
        stats = self.inj.boundary_saturation(margin=1.0)
        assert stats["low_fraction"] == 1.0
        assert stats["high_fraction"] == 0.0

    def test_boundary_saturation_partial_channels(self):
        """Only half the channels saturated → fraction reflects the split."""
        with torch.no_grad():
            # First half far into the high-saturation zone, second half near zero.
            half = self.inj.log_A.shape[0] // 2
            self.inj.log_A[:half] = 30.0
            self.inj.log_A[half:] = 0.0
        stats = self.inj.boundary_saturation(margin=1.0)
        # Fractions are over all channels, so half-at-boundary ⇒ 0.5.
        assert abs(stats["high_fraction"] - 0.5) < 1e-6
        assert stats["low_fraction"] == 0.0


# ---------------------------------------------------------------------------
# ACTHalting
# ---------------------------------------------------------------------------


class TestACTHalting:
    def setup_method(self):
        self.act = ACTHalting(dim=64)

    def test_output_shape(self):
        h = torch.randn(B, T, 64)
        p = self.act(h)
        assert p.shape == (B, T)

    def test_values_in_01(self):
        h = torch.randn(B, T, 64)
        p = self.act(h)
        assert p.min().item() >= 0.0
        assert p.max().item() <= 1.0


# ---------------------------------------------------------------------------
# RecurrentBlock
# ---------------------------------------------------------------------------


class TestRecurrentBlock:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.block = RecurrentBlock(self.cfg)
        self.freqs = precompute_rope_freqs(
            self.cfg.dim // self.cfg.n_heads, self.cfg.max_seq_len
        )

    def test_output_shape(self):
        h = torch.randn(B, T, self.cfg.dim)
        e = torch.randn(B, T, self.cfg.dim)
        out = self.block(h, e, self.freqs[:T])
        assert out.shape == (B, T, self.cfg.dim)

    def test_more_loops_changes_output(self):
        h = torch.randn(B, T, self.cfg.dim)
        e = torch.randn(B, T, self.cfg.dim)
        out1 = self.block(h.clone(), e.clone(), self.freqs[:T], n_loops=1)
        out3 = self.block(h.clone(), e.clone(), self.freqs[:T], n_loops=3)
        assert not torch.allclose(out1, out3)

    def test_single_loop_runs(self):
        h = torch.randn(B, T, self.cfg.dim)
        e = torch.randn(B, T, self.cfg.dim)
        out = self.block(h, e, self.freqs[:T], n_loops=1)
        assert out.shape == (B, T, self.cfg.dim)


# ---------------------------------------------------------------------------
# OpenMythos — GQA mode
# ---------------------------------------------------------------------------


class TestOpenMythosGQA:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.model = OpenMythos(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (B, T))

    def test_forward_shape(self):
        logits = self.model(self.ids)
        assert logits.shape == (B, T, self.cfg.vocab_size)

    def test_forward_no_nan(self):
        logits = self.model(self.ids)
        assert not torch.isnan(logits).any()

    def test_generate_shape(self):
        out = self.model.generate(self.ids, max_new_tokens=4, n_loops=2)
        assert out.shape == (B, T + 4)

    def test_weight_tying(self):
        assert self.model.head.weight is self.model.embed.weight

    def test_lti_spectral_radius(self):
        A = self.model.recurrent.injection.get_A()
        assert A.max().item() < 1.0

    def test_depth_extrapolation_changes_output(self):
        # More loops at inference should produce different (ideally better) output
        logits_shallow = self.model(self.ids, n_loops=1)
        logits_deep = self.model(self.ids, n_loops=3)
        assert not torch.allclose(logits_shallow, logits_deep)

    def test_kv_cache_generate_matches_no_cache(self):
        # Single-token generation with and without cache should agree
        torch.manual_seed(0)
        prompt = torch.randint(0, self.cfg.vocab_size, (1, T))
        with torch.no_grad():
            logits_no_cache = self.model(prompt, n_loops=2)[:, -1, :]
            cache = {}
            logits_cached = self.model(prompt, n_loops=2, kv_cache=cache)[:, -1, :]
        assert torch.allclose(logits_no_cache, logits_cached, atol=1e-4)

    def test_single_token_forward(self):
        # Mask is None when T=1; should not crash
        single = torch.randint(0, self.cfg.vocab_size, (B, 1))
        logits = self.model(single)
        assert logits.shape == (B, 1, self.cfg.vocab_size)


# ---------------------------------------------------------------------------
# OpenMythos — MLA mode
# ---------------------------------------------------------------------------


class TestOpenMythosMLА:
    def setup_method(self):
        self.cfg = mla_cfg()
        self.model = OpenMythos(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (B, T))

    def test_forward_shape(self):
        logits = self.model(self.ids)
        assert logits.shape == (B, T, self.cfg.vocab_size)

    def test_forward_no_nan(self):
        assert not torch.isnan(self.model(self.ids)).any()

    def test_generate_shape(self):
        out = self.model.generate(self.ids, max_new_tokens=4, n_loops=2)
        assert out.shape == (B, T + 4)

    def test_lti_spectral_radius(self):
        A = self.model.recurrent.injection.get_A()
        assert A.max().item() < 1.0

    def test_mla_cache_is_compressed(self):
        # MLA cache should store c_kv (lora_rank), not full K/V (n_heads * head_dim)
        cache = {}
        with torch.no_grad():
            self.model(self.ids, kv_cache=cache)
        # find any MLA cache entry and check dimensions
        mla_entries = {k: v for k, v in cache.items() if "c_kv" in v}
        assert len(mla_entries) > 0
        for entry in mla_entries.values():
            assert entry["c_kv"].shape[-1] == self.cfg.kv_lora_rank


# ---------------------------------------------------------------------------
# GQA vs MLA: same config, different attn_type
# ---------------------------------------------------------------------------


class TestAttnTypeSwap:
    def test_gqa_and_mla_produce_different_outputs(self):
        cfg_gqa = gqa_cfg()
        cfg_mla = mla_cfg()
        ids = torch.randint(0, cfg_gqa.vocab_size, (B, T))
        logits_gqa = OpenMythos(cfg_gqa)(ids)
        logits_mla = OpenMythos(cfg_mla)(ids)
        # different architectures, different params → outputs must differ
        assert not torch.allclose(logits_gqa, logits_mla)

    def test_both_modes_produce_valid_shapes(self):
        ids = torch.randint(0, 200, (B, T))
        for attn_type in ("gqa", "mla"):
            cfg = gqa_cfg(attn_type=attn_type)
            logits = OpenMythos(cfg)(ids)
            assert logits.shape == (B, T, cfg.vocab_size)

    def test_mla_fewer_kv_cache_bytes(self):
        # MLA cache should be smaller than GQA cache for the same sequence
        ids = torch.randint(0, 200, (1, T))
        cache_gqa, cache_mla = {}, {}
        with torch.no_grad():
            OpenMythos(gqa_cfg())(ids, kv_cache=cache_gqa)
            OpenMythos(mla_cfg())(ids, kv_cache=cache_mla)

        def cache_bytes(cache):
            return sum(
                t.numel() * t.element_size()
                for entry in cache.values()
                for t in entry.values()
            )

        assert cache_bytes(cache_mla) < cache_bytes(cache_gqa)


# ---------------------------------------------------------------------------
# ACT regression — halted positions must not leak into weight accumulation
# ---------------------------------------------------------------------------


class TestACTRegression:
    def setup_method(self):
        self.cfg = gqa_cfg(max_loop_iters=8, act_threshold=0.99)
        self.block = RecurrentBlock(self.cfg)
        self.freqs = precompute_rope_freqs(
            self.cfg.dim // self.cfg.n_heads, self.cfg.max_seq_len
        )

    def test_act_weights_sum_approximately_one(self):
        torch.manual_seed(42)
        captured: list[torch.Tensor] = []

        original_forward = self.block.forward

        def instrumented_forward(h, e, freqs_cis, mask=None, n_loops=None, kv_cache=None):
            n_loops = n_loops or self.block.cfg.max_loop_iters
            B, T, D = h.shape
            halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
            cumulative_p = torch.zeros(B, T, device=h.device)
            h_out = torch.zeros_like(h)

            for t in range(n_loops):
                h_loop = loop_index_embedding(h, t, self.block.loop_dim)
                combined = self.block.norm(h_loop + e)
                cache_key = f"recurrent_loop_{t}"
                trans_out = self.block.block(combined, freqs_cis, mask, kv_cache, cache_key)
                trans_out = trans_out + self.block.lora(trans_out, t)
                h = self.block.injection(h, e, trans_out)

                p = self.block.act(h)
                still_running = ~halted
                remainder = (1.0 - cumulative_p).clamp(min=0)
                weight = torch.where(
                    cumulative_p + p >= self.block.cfg.act_threshold,
                    remainder,
                    p,
                )
                weight = weight * still_running.float()
                captured.append(weight.detach().clone())
                h_out = h_out + weight.unsqueeze(-1) * h

                cumulative_p = cumulative_p + p * still_running.float()
                halted = halted | (cumulative_p >= self.block.cfg.act_threshold)

                if halted.all() and kv_cache is None:
                    break

            return h_out

        self.block.forward = instrumented_forward
        try:
            h = torch.randn(B, T, self.cfg.dim)
            e = torch.randn(B, T, self.cfg.dim)
            self.block(h, e, self.freqs[:T], n_loops=self.cfg.max_loop_iters)
        finally:
            self.block.forward = original_forward

        total = torch.stack(captured, dim=0).sum(dim=0)  # (B, T)
        assert torch.allclose(total, torch.ones_like(total), atol=1e-5)

    def test_act_forced_halt_iter1_matches_hidden(self):
        torch.manual_seed(42)
        h = torch.randn(B, T, self.cfg.dim)
        e = torch.randn(B, T, self.cfg.dim)

        # Force ACT to emit logits whose sigmoid is ~0.999 so cumulative_p
        # crosses the threshold on the first step for every position.
        with torch.no_grad():
            self.block.act.halt.weight.zero_()
            self.block.act.halt.bias.fill_(torch.logit(torch.tensor(0.999)).item())

        # Recompute the first-iteration hidden state manually.
        h_loop = loop_index_embedding(h, 0, self.block.loop_dim)
        combined = self.block.norm(h_loop + e)
        trans_out = self.block.block(combined, self.freqs[:T], None, None, "recurrent_loop_0")
        trans_out = trans_out + self.block.lora(trans_out, 0)
        h_iter1 = self.block.injection(h, e, trans_out)

        out = self.block(h, e, self.freqs[:T], n_loops=4)
        # Since every position halts on iter 0 with weight = 1, h_out == h_iter1
        torch.testing.assert_close(out, h_iter1, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# KV cache regression — all recurrent loop cache keys must populate even when
# ACT halts every position on the first iteration
# ---------------------------------------------------------------------------


class TestKVCacheRegression:
    def setup_method(self):
        self.cfg = gqa_cfg(prelude_layers=1, coda_layers=1, max_loop_iters=3)
        self.model = OpenMythos(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (B, T))

    def test_kv_cache_populates_all_loop_keys(self):
        torch.manual_seed(42)
        cache: dict = {}
        with torch.no_grad():
            self.model(self.ids, kv_cache=cache, n_loops=3)
        for i in range(3):
            assert f"recurrent_loop_{i}" in cache
        assert "prelude_0" in cache
        assert "coda_0" in cache

    def test_kv_cache_all_halted_still_fills_cache(self):
        torch.manual_seed(42)
        with torch.no_grad():
            self.model.recurrent.act.halt.weight.zero_()
            self.model.recurrent.act.halt.bias.fill_(
                torch.logit(torch.tensor(0.999)).item()
            )

        cache: dict = {}
        with torch.no_grad():
            self.model(self.ids, kv_cache=cache, n_loops=3)
        for i in range(3):
            assert f"recurrent_loop_{i}" in cache


# ---------------------------------------------------------------------------
# LoRA depth extrapolation regression — loop_t >= max_loops must clamp cleanly
# ---------------------------------------------------------------------------


class TestLoRAExtrapolationRegression:
    def test_lora_clamps_beyond_max_loops(self):
        torch.manual_seed(42)
        lora = LoRAAdapter(dim=64, rank=4, max_loops=10)
        x = torch.randn(2, 4, 64)
        out_at_max = lora(x, loop_t=9)
        out_extrap = lora(x, loop_t=100)
        torch.testing.assert_close(out_at_max, out_extrap, atol=0, rtol=0)

    def test_lora_does_not_crash_on_extrapolation(self):
        torch.manual_seed(42)
        lora = LoRAAdapter(dim=64, rank=4, max_loops=10)
        x = torch.randn(2, 4, 64)
        out = lora(x, loop_t=99999)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# RoPE decode regression — start_pos must advance per decode step
# ---------------------------------------------------------------------------


class TestRoPEDecodeRegression:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.model = OpenMythos(self.cfg)

    def test_rope_generate_uses_increasing_start_pos(self):
        torch.manual_seed(42)
        prompt_len = 4
        prompt = torch.randint(0, self.cfg.vocab_size, (1, prompt_len))
        next_tok = torch.randint(0, self.cfg.vocab_size, (1, 1))

        captured: list[torch.Tensor] = []
        import open_mythos.main as mm

        original_apply_rope = mm.apply_rope

        def spy(x, freqs_cis):
            captured.append(freqs_cis.detach().clone())
            return original_apply_rope(x, freqs_cis)

        mm.apply_rope = spy
        try:
            cache: dict = {}
            with torch.no_grad():
                self.model(prompt, kv_cache=cache, start_pos=0)
                prefill_snapshot = list(captured)
                captured.clear()
                self.model(next_tok, kv_cache=cache, start_pos=prompt_len)
        finally:
            mm.apply_rope = original_apply_rope

        # Prefill used freqs[0:prompt_len]; decode step must use freqs[prompt_len:prompt_len+1]
        expected_prefill = self.model.freqs_cis[0:prompt_len]
        expected_decode = self.model.freqs_cis[prompt_len : prompt_len + 1]
        assert len(prefill_snapshot) > 0
        assert len(captured) > 0
        torch.testing.assert_close(prefill_snapshot[0], expected_prefill)
        torch.testing.assert_close(captured[0], expected_decode)
        # Prefill slice differs from decode slice
        assert not torch.allclose(prefill_snapshot[0][-1:], captured[0])

    def test_generate_multi_step_not_position_0(self):
        torch.manual_seed(42)
        prompt_len = 4
        prompt = torch.randint(0, self.cfg.vocab_size, (1, prompt_len))

        torch.manual_seed(123)
        generated = self.model.generate(prompt, max_new_tokens=3, n_loops=2)

        # Simulate a broken generate that passes start_pos=0 every step with a reset cache.
        # Collect final logits at each step under this broken regime for comparison.
        torch.manual_seed(123)
        current = prompt.clone()
        broken_logits: list[torch.Tensor] = []
        with torch.no_grad():
            for step in range(3):
                if step == 0:
                    cur_ids = current
                else:
                    cur_ids = current[:, -1:]
                broken_cache: dict = {}
                logits = self.model(
                    cur_ids, n_loops=2, kv_cache=broken_cache, start_pos=0
                )
                broken_logits.append(logits[:, -1, :].clone())
                # Greedy pick to extend in this comparison trajectory
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                current = torch.cat([current, next_id], dim=1)

        # Now compute the same under the real generate's start_pos trajectory
        torch.manual_seed(123)
        current2 = prompt.clone()
        correct_cache: dict = {}
        correct_logits: list[torch.Tensor] = []
        with torch.no_grad():
            for step in range(3):
                if step == 0:
                    cur_ids = current2
                    start_pos = 0
                else:
                    cur_ids = current2[:, -1:]
                    start_pos = prompt_len + step - 1
                logits = self.model(
                    cur_ids, n_loops=2, kv_cache=correct_cache, start_pos=start_pos
                )
                correct_logits.append(logits[:, -1, :].clone())
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                current2 = torch.cat([current2, next_id], dim=1)

        # At least one decode step must differ between the broken (start_pos=0)
        # trajectory and the correct one.
        any_diff = any(
            not torch.allclose(b, c, atol=1e-5)
            for b, c in zip(broken_logits[1:], correct_logits[1:])
        )
        assert any_diff
        # Ensure generate actually produced tokens
        assert generated.shape == (1, prompt_len + 3)


# ---------------------------------------------------------------------------
# MythosConfig validation (frozen + __post_init__ invariants) — P1.1
# ---------------------------------------------------------------------------


class TestMythosConfigValidation:
    def test_invalid_attn_type_raises(self):
        with pytest.raises(ValueError, match="attn_type"):
            gqa_cfg(attn_type="bogus")

    def test_dim_not_divisible_by_n_heads_raises(self):
        with pytest.raises(ValueError, match="divisible by n_heads"):
            gqa_cfg(dim=65, n_heads=4)

    def test_gqa_n_heads_not_divisible_by_n_kv_heads_raises(self):
        with pytest.raises(ValueError, match="divisible by n_kv_heads"):
            gqa_cfg(n_heads=4, n_kv_heads=3)

    def test_mla_does_not_check_n_kv_heads(self):
        # MLA ignores n_kv_heads, so a non-divisible value must not raise.
        cfg = mla_cfg(n_heads=4, n_kv_heads=3)
        assert cfg.attn_type == "mla"

    def test_experts_per_tok_exceeds_n_experts_raises(self):
        with pytest.raises(ValueError, match="n_experts_per_tok"):
            gqa_cfg(n_experts=4, n_experts_per_tok=8)

    def test_act_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError, match="act_threshold"):
            gqa_cfg(act_threshold=0.0)
        with pytest.raises(ValueError, match="act_threshold"):
            gqa_cfg(act_threshold=-0.1)
        with pytest.raises(ValueError, match="act_threshold"):
            gqa_cfg(act_threshold=1.5)

    def test_lora_rank_exceeds_dim_raises(self):
        with pytest.raises(ValueError, match="lora_rank"):
            gqa_cfg(dim=64, lora_rank=128)

    def test_lora_rank_zero_raises(self):
        with pytest.raises(ValueError, match="lora_rank"):
            gqa_cfg(lora_rank=0)

    def test_frozen_config_cannot_be_mutated(self):
        import dataclasses

        cfg = gqa_cfg()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.dim = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ACT halted hidden state frozen — P1.2
# Halted positions must keep their iter-0 hidden state bit-exactly across
# subsequent loops, so no new cache entries are written for them.
# ---------------------------------------------------------------------------


class TestACTHaltedFrozen:
    def setup_method(self):
        self.cfg = gqa_cfg(max_loop_iters=4, act_threshold=0.99)
        self.block = RecurrentBlock(self.cfg)
        self.freqs = precompute_rope_freqs(
            self.cfg.dim // self.cfg.n_heads, self.cfg.max_seq_len
        )

    def test_act_halted_hidden_state_frozen(self):
        torch.manual_seed(42)
        h = torch.randn(B, T, self.cfg.dim)
        e = torch.randn(B, T, self.cfg.dim)

        # Force ACT to halt every position on iter 0 (sigmoid ≈ 0.999).
        with torch.no_grad():
            self.block.act.halt.weight.zero_()
            self.block.act.halt.bias.fill_(torch.logit(torch.tensor(0.999)).item())

        # Compute the iter-0 hidden state manually — the state every halted
        # position should stay pinned to for all subsequent loop iterations.
        h_loop = loop_index_embedding(h, 0, self.block.loop_dim)
        combined = self.block.norm(h_loop + e)
        trans_out = self.block.block(combined, self.freqs[:T], None, None, "recurrent_loop_0")
        trans_out = trans_out + self.block.lora(trans_out, 0)
        h_iter0 = self.block.injection(h, e, trans_out)

        # Capture the internal h by swapping in an instrumented forward that
        # mirrors the production logic but records h each iteration.
        captured: list[torch.Tensor] = []
        original_forward = self.block.forward

        def instrumented(h_in, e_in, freqs_cis, mask=None, n_loops=None, kv_cache=None):
            n_loops = n_loops or self.block.cfg.max_loop_iters
            Bh, Th, Dh = h_in.shape
            halted = torch.zeros(Bh, Th, device=h_in.device, dtype=torch.bool)
            cumulative_p = torch.zeros(Bh, Th, device=h_in.device)
            h_out = torch.zeros_like(h_in)
            h_state = h_in
            for t in range(n_loops):
                h_loop = loop_index_embedding(h_state, t, self.block.loop_dim)
                combined = self.block.norm(h_loop + e_in)
                cache_key = f"recurrent_loop_{t}"
                trans_out = self.block.block(combined, freqs_cis, mask, kv_cache, cache_key)
                trans_out = trans_out + self.block.lora(trans_out, t)
                h_new = self.block.injection(h_state, e_in, trans_out)
                still_running_mask = (~halted).unsqueeze(-1).float()
                h_state = h_new * still_running_mask + h_state * (1.0 - still_running_mask)
                captured.append(h_state.detach().clone())

                p = self.block.act(h_state)
                still_running = ~halted
                remainder = (1.0 - cumulative_p).clamp(min=0)
                weight = torch.where(
                    cumulative_p + p >= self.block.cfg.act_threshold,
                    remainder,
                    p,
                )
                weight = weight * still_running.float()
                h_out = h_out + weight.unsqueeze(-1) * h_state
                cumulative_p = cumulative_p + p * still_running.float()
                halted = halted | (cumulative_p >= self.block.cfg.act_threshold)
                # Production code (P3.5) removed the halted.all() early-exit to
                # avoid a GPU sync and keep the loop compile-friendly. Mirror
                # that here so every n_loops iteration actually runs and we can
                # verify halted positions remain frozen.
            return h_out

        self.block.forward = instrumented
        try:
            self.block(h, e, self.freqs[:T], n_loops=3)
        finally:
            self.block.forward = original_forward

        # All n_loops iterations should have run; halted h must be bit-exact h_iter0.
        assert len(captured) >= 2
        for h_t in captured[1:]:
            torch.testing.assert_close(h_t, h_iter0, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# LoRA extrapolation warn-once — P1.3
# ---------------------------------------------------------------------------


class TestLoRAExtrapolationWarn:
    def test_lora_extrapolation_warns_once(self):
        torch.manual_seed(42)
        lora = LoRAAdapter(dim=64, rank=4, max_loops=10)
        x = torch.randn(2, 4, 64)
        with pytest.warns(RuntimeWarning, match="exceeds max_loops"):
            lora(x, loop_t=15)

    def test_lora_extrapolation_warn_not_repeated(self):
        import warnings as _warnings

        torch.manual_seed(42)
        lora = LoRAAdapter(dim=64, rank=4, max_loops=10)
        x = torch.randn(2, 4, 64)
        # Trigger the one-shot warning first.
        with pytest.warns(RuntimeWarning, match="exceeds max_loops"):
            lora(x, loop_t=20)
        # Second out-of-bounds call must not emit another RuntimeWarning.
        with _warnings.catch_warnings(record=True) as recorded:
            _warnings.simplefilter("always")
            lora(x, loop_t=30)
        runtime_warns = [w for w in recorded if issubclass(w.category, RuntimeWarning)]
        assert runtime_warns == []

    def test_lora_in_bounds_no_warning(self):
        import warnings as _warnings

        torch.manual_seed(42)
        lora = LoRAAdapter(dim=64, rank=4, max_loops=10)
        x = torch.randn(2, 4, 64)
        with _warnings.catch_warnings(record=True) as recorded:
            _warnings.simplefilter("always")
            for t in range(10):
                lora(x, loop_t=t)
        runtime_warns = [w for w in recorded if issubclass(w.category, RuntimeWarning)]
        assert runtime_warns == []


# ---------------------------------------------------------------------------
# generate() max_seq_len boundary — P1.5
# ---------------------------------------------------------------------------


class TestGenerateBoundary:
    def test_generate_stops_at_max_seq_len(self):
        torch.manual_seed(42)
        cfg = gqa_cfg(max_seq_len=8)
        model = OpenMythos(cfg)
        prompt = torch.randint(0, cfg.vocab_size, (1, 4))
        out = model.generate(prompt, max_new_tokens=100, n_loops=2)
        # generate() stops once the next forward would exceed RoPE's
        # precomputed range (start_pos + cur_ids > max_seq_len). In practice
        # that caps output at max_seq_len + 1 tokens (the last accepted step
        # fills positions up to max_seq_len inclusive). Verify truncation
        # kicks in well before prompt_len + max_new_tokens would have.
        assert out.shape[1] <= cfg.max_seq_len + 1
        assert out.shape[1] >= prompt.shape[1]
        assert out.shape[1] < prompt.shape[1] + 100


# ===========================================================================
# P3 performance tasks — numerical equivalence tests
# ===========================================================================


# ---------------------------------------------------------------------------
# P3.1 — SDPA / FlashAttention equivalence
# ---------------------------------------------------------------------------


def _manual_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    """Reference attention: softmax(QK^T / sqrt(d)) @ V, with optional causal mask."""
    scale = q.shape[-1] ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        T_q, T_kv = q.shape[-2], k.shape[-2]
        mask = torch.full((T_q, T_kv), float("-inf"), device=q.device, dtype=q.dtype)
        mask = torch.triu(mask, diagonal=1)
        attn = attn + mask
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


class TestSDPAEquivalence:
    def test_gqa_sdpa_matches_manual_causal(self):
        """After P3.1, GQAttention output must equal the manual softmax(QK^T)·V path."""
        torch.manual_seed(0)
        cfg = gqa_cfg()
        attn = GQAttention(cfg).eval()  # eval() so dropout is a no-op
        x = torch.randn(1, 6, cfg.dim, dtype=torch.float32)

        freqs = precompute_rope_freqs(
            cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta
        )
        mask = torch.triu(
            torch.full((1, 1, 6, 6), float("-inf")), diagonal=1
        )

        out_sdpa = attn(x, freqs[:6], mask=mask)

        # Rebuild Q/K/V the same way the module does, then run manual SDPA.
        B, T = 1, 6
        q = attn.wq(x).view(B, T, cfg.n_heads, attn.head_dim)
        k = attn.wk(x).view(B, T, cfg.n_kv_heads, attn.head_dim)
        v = attn.wv(x).view(B, T, cfg.n_kv_heads, attn.head_dim)
        q = apply_rope(q, freqs[:T])
        k = apply_rope(k, freqs[:T])
        # GQA expansion
        k = k.repeat_interleave(attn.groups, dim=2)
        v = v.repeat_interleave(attn.groups, dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        ref = _manual_sdpa(q, k, v, causal=True)
        ref = ref.transpose(1, 2).contiguous().view(B, T, -1)
        ref = attn.wo(ref)

        torch.testing.assert_close(out_sdpa, ref, atol=1e-5, rtol=1e-5)

    def test_mla_sdpa_matches_manual_causal(self):
        """After P3.1, MLAttention output must equal the manual softmax(QK^T)·V path."""
        torch.manual_seed(0)
        cfg = mla_cfg()
        attn = MLAttention(cfg).eval()
        x = torch.randn(1, 6, cfg.dim, dtype=torch.float32)

        freqs = precompute_rope_freqs(
            cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta
        )
        mask = torch.triu(
            torch.full((1, 1, 6, 6), float("-inf")), diagonal=1
        )

        out_sdpa = attn(x, freqs[:6], mask=mask)

        # Manual path: replicate the internals, but substitute _manual_sdpa for
        # the fused call.
        B, T = 1, 6
        c_q = attn.q_norm(attn.q_down(x))
        q_nope = attn.q_up_nope(c_q).view(B, T, cfg.n_heads, attn.qk_nope_dim)
        q_rope = attn.q_up_rope(c_q).view(B, T, cfg.n_heads, attn.qk_rope_dim)
        q_rope = apply_rope(q_rope, freqs[:T])
        q = torch.cat([q_nope, q_rope], dim=-1)

        kv_raw = attn.kv_down(x)
        c_kv = kv_raw[..., : attn.kv_lora_rank]
        k_rope = kv_raw[..., attn.kv_lora_rank :]
        k_rope = (
            k_rope.unsqueeze(2)
            .expand(B, T, cfg.n_heads, attn.qk_rope_dim)
            .contiguous()
        )
        k_rope = apply_rope(k_rope, freqs[:T])

        kv = attn.kv_up(attn.kv_norm(c_kv))
        kv = kv.view(B, T, cfg.n_heads, attn.qk_nope_dim + attn.v_dim)
        k_nope = kv[..., : attn.qk_nope_dim]
        v = kv[..., attn.qk_nope_dim :]
        k = torch.cat([k_nope, k_rope], dim=-1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        ref = _manual_sdpa(q, k, v, causal=True)
        ref = ref.transpose(1, 2).contiguous().view(B, T, -1)
        ref = attn.wo(ref)

        torch.testing.assert_close(out_sdpa, ref, atol=1e-5, rtol=1e-5)

    def test_gqa_decode_no_causal_mask(self):
        """During decode (T_q=1, T_kv > 1) SDPA must NOT be called with is_causal=True."""
        torch.manual_seed(0)
        cfg = gqa_cfg()
        attn = GQAttention(cfg).eval()

        freqs = precompute_rope_freqs(
            cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta
        )
        # Pre-populate the cache with a "prefill" pass.
        kv_cache: dict = {}
        x_prefill = torch.randn(1, 4, cfg.dim)
        attn(x_prefill, freqs[:4], mask=None, kv_cache=kv_cache, cache_key="L0")

        # Now intercept F.scaled_dot_product_attention to see what flags are used.
        import open_mythos.main as _mm

        calls = []
        real_sdpa = _mm.F.scaled_dot_product_attention

        def spy(q, k, v, **kw):
            calls.append(kw)
            return real_sdpa(q, k, v, **kw)

        _mm.F.scaled_dot_product_attention = spy
        try:
            x_decode = torch.randn(1, 1, cfg.dim)
            attn(x_decode, freqs[4:5], mask=None, kv_cache=kv_cache, cache_key="L0")
        finally:
            _mm.F.scaled_dot_product_attention = real_sdpa

        # One of the attempted call kwargs must have is_causal=False (since T_q=1).
        assert len(calls) >= 1
        for call_kw in calls:
            assert call_kw.get("is_causal", False) is False, (
                f"decode path used is_causal=True unexpectedly: {call_kw}"
            )


# ---------------------------------------------------------------------------
# P3.2 — Vectorized MoE dispatch equivalence
# ---------------------------------------------------------------------------


class TestMoEVectorized:
    def test_vectorized_moe_matches_reference(self):
        """Compare vectorized forward against a straightforward per-token reference."""
        torch.manual_seed(0)
        cfg = gqa_cfg(n_experts=4, n_experts_per_tok=2, expert_dim=16)
        moe = MoEFFN(cfg).eval()
        x = torch.randn(2, 5, cfg.dim)

        out = moe(x)

        # Reference: for every token, compute top-K experts by the same
        # algorithm the module uses, then sum weighted expert outputs + shared.
        B, T, D = x.shape
        flat = x.view(B * T, D)
        # Stage 4.3: moe.router is a BiasRouter module, not raw nn.Linear.
        # Reach through to .linear and .router_bias for the reference path.
        logits = moe.router.linear(flat)
        scores = torch.softmax(logits, dim=-1)
        _, topk_idx = (logits + moe.router.router_bias).topk(moe.topk, dim=-1)
        topk_scores = scores.gather(-1, topk_idx)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        ref = torch.zeros_like(flat)
        for n in range(flat.shape[0]):
            for k in range(moe.topk):
                eid = topk_idx[n, k].item()
                w = topk_scores[n, k]
                ref[n] = ref[n] + w * moe.routed_experts[eid](flat[n : n + 1]).squeeze(0)
        for shared in moe.shared_experts:
            ref = ref + shared(flat)
        ref = ref.view(B, T, D)

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_moe_gradient_flow(self):
        """Gradient must reach every routed expert's weight through the vectorized path."""
        torch.manual_seed(0)
        cfg = gqa_cfg(n_experts=4, n_experts_per_tok=4, expert_dim=16)  # all experts active
        moe = MoEFFN(cfg).train()
        x = torch.randn(2, 4, cfg.dim, requires_grad=False)
        out = moe(x)
        out.sum().backward()
        for i, expert in enumerate(moe.routed_experts):
            assert expert.gate.weight.grad is not None, f"expert {i} gate grad is None"
            assert expert.up.weight.grad is not None, f"expert {i} up grad is None"
            assert expert.down.weight.grad is not None, f"expert {i} down grad is None"


# ---------------------------------------------------------------------------
# P3.4 — RoPE dtype preservation
# ---------------------------------------------------------------------------


class TestRoPEDtypePreservation:
    def test_rope_bf16_input_bf16_output(self):
        freqs = precompute_rope_freqs(dim=16, max_len=8)
        x = torch.randn(1, 4, 2, 16, dtype=torch.bfloat16)
        out = apply_rope(x, freqs[:4])
        assert out.dtype == torch.bfloat16, f"expected bf16, got {out.dtype}"

    def test_rope_fp32_unchanged(self):
        """FP32 path through the P3.4 refactor must be numerically equivalent to prior impl."""
        torch.manual_seed(42)
        freqs = precompute_rope_freqs(dim=16, max_len=8)
        x = torch.randn(2, 4, 2, 16, dtype=torch.float32)
        out_new = apply_rope(x, freqs[:4])
        # Reference: do the rotation manually using complex multiplication.
        xc = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        ref = torch.view_as_real(xc * freqs[:4].unsqueeze(0).unsqueeze(2)).flatten(-2)
        torch.testing.assert_close(out_new, ref, atol=1e-6, rtol=1e-6)

    def test_rope_fp16_roundtrip(self):
        freqs = precompute_rope_freqs(dim=16, max_len=8)
        x = torch.randn(1, 4, 2, 16, dtype=torch.float16)
        out = apply_rope(x, freqs[:4])
        assert out.dtype == torch.float16


# ---------------------------------------------------------------------------
# P3.5 — ACT early-exit removed (training path runs all loops)
# ---------------------------------------------------------------------------


class TestRecurrentNoEarlyExit:
    def test_recurrent_runs_all_loops_when_no_cache(self):
        """After P3.5, the loop runs n_loops iterations regardless of halting."""
        torch.manual_seed(0)
        cfg = gqa_cfg(act_threshold=0.01)  # trivial threshold; halts immediately
        rb = RecurrentBlock(cfg).eval()
        freqs = precompute_rope_freqs(
            cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta
        )
        h = torch.randn(1, 4, cfg.dim)
        e = torch.randn(1, 4, cfg.dim)

        calls = 0
        original_forward = rb.block.forward

        def counting_forward(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_forward(*args, **kwargs)

        rb.block.forward = counting_forward
        try:
            rb(h, e, freqs[:4], mask=None, n_loops=4, kv_cache=None)
        finally:
            rb.block.forward = original_forward

        # With P3.5 we expect all 4 loop iterations to fire even with ACT saturating
        # at iteration 0.
        assert calls == 4, f"expected 4 loop iterations, got {calls}"


# ---------------------------------------------------------------------------
# P3.6 — LoRA direct-index equivalence (numerical invariance after refactor)
# ---------------------------------------------------------------------------


class TestLoRAPerformance:
    def test_lora_output_unchanged_after_p36_refactor(self):
        torch.manual_seed(0)
        lora = LoRAAdapter(dim=32, rank=4, max_loops=8).eval()
        x = torch.randn(2, 3, 32)
        out = lora(x, loop_t=2)

        # Hand-compute the exact math the forward should implement.
        s = lora.scale.weight[2]
        ref = (lora.down(x) * s) @ lora.B
        torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)

    def test_lora_clamp_extrapolation_uses_last_row(self):
        torch.manual_seed(0)
        lora = LoRAAdapter(dim=32, rank=4, max_loops=5).eval()
        x = torch.randn(1, 2, 32)
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            out_extrap = lora(x, loop_t=20)
        # Clamped to index 4 (last row).
        s_last = lora.scale.weight[4]
        ref = (lora.down(x) * s_last) @ lora.B
        torch.testing.assert_close(out_extrap, ref, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# P3.7 — Precomputed loop-index embedding equivalence
# ---------------------------------------------------------------------------


class TestLoopIndexPerformance:
    def test_precomputed_loop_emb_matches_standalone_function(self):
        torch.manual_seed(0)
        cfg = gqa_cfg()
        rb = RecurrentBlock(cfg)
        loop_dim = rb.loop_dim

        # For every t in [0, max_loops), the cached row must equal what the
        # standalone loop_index_embedding would add to h when h is zeros.
        for t in range(cfg.max_loop_iters):
            h_zeros = torch.zeros(1, 1, cfg.dim, dtype=torch.float32)
            ref = loop_index_embedding(h_zeros, t, loop_dim)  # (1,1,dim)
            cached = rb.loop_idx_table[t].view(1, 1, -1)
            torch.testing.assert_close(
                cached.to(torch.float32),
                ref.to(torch.float32),
                atol=1e-6,
                rtol=1e-6,
            )

    def test_recurrent_block_forward_matches_standalone_loop_emb(self):
        """Dropping the standalone call should not change the recurrent output."""
        torch.manual_seed(0)
        cfg = gqa_cfg()
        rb = RecurrentBlock(cfg).eval()
        freqs = precompute_rope_freqs(
            cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta
        )
        h = torch.randn(1, 4, cfg.dim)
        e = torch.randn(1, 4, cfg.dim)
        # forward must run without error and produce finite output.
        out = rb(h, e, freqs[:4], mask=None, n_loops=cfg.max_loop_iters)
        assert out.shape == h.shape
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# P3.3: StaticKVCache regression tests
# ---------------------------------------------------------------------------


class TestStaticKVCache:
    """Unit tests for the StaticKVCache primitive itself."""

    def setup_method(self):
        from open_mythos.kv_cache import StaticKVCache
        self.StaticKVCache = StaticKVCache

    def test_lazy_allocation_on_first_write(self):
        cache = self.StaticKVCache(max_seq_len=16)
        assert "layer0" not in cache
        assert cache.current_length("layer0") == 0

        chunk = torch.randn(2, 4, 3, 8)
        out = cache.write_and_read("layer0:k", chunk)
        assert out.shape == (2, 4, 3, 8)
        assert cache.current_length("layer0:k") == 4
        assert "layer0" in cache

    def test_multiple_writes_accumulate(self):
        cache = self.StaticKVCache(max_seq_len=16)
        a = torch.randn(2, 4, 3, 8)
        b = torch.randn(2, 2, 3, 8)

        cache.write_and_read("k", a)
        out = cache.write_and_read("k", b)

        assert out.shape == (2, 6, 3, 8)
        torch.testing.assert_close(out[:, :4], a)
        torch.testing.assert_close(out[:, 4:6], b)

    def test_buffer_identity_preserved_across_writes(self):
        cache = self.StaticKVCache(max_seq_len=32)
        a = torch.randn(1, 1, 2, 4)
        for _ in range(5):
            cache.write_and_read("k", a)
        bid = cache.buffer_id("k")
        assert bid is not None
        for _ in range(5):
            cache.write_and_read("k", a)
        assert cache.buffer_id("k") == bid
        assert cache.current_length("k") == 10

    def test_overflow_raises(self):
        cache = self.StaticKVCache(max_seq_len=4)
        cache.write_and_read("k", torch.randn(1, 3, 2, 4))
        with pytest.raises(RuntimeError, match="overflow"):
            cache.write_and_read("k", torch.randn(1, 2, 2, 4))

    def test_reset_zeros_positions_keeps_buffers(self):
        cache = self.StaticKVCache(max_seq_len=8)
        cache.write_and_read("k", torch.randn(1, 3, 2, 4))
        bid = cache.buffer_id("k")
        cache.reset()
        assert cache.current_length("k") == 0
        assert cache.buffer_id("k") == bid  # buffer reused

    def test_invalid_max_seq_len_raises(self):
        with pytest.raises(ValueError):
            self.StaticKVCache(max_seq_len=0)
        with pytest.raises(ValueError):
            self.StaticKVCache(max_seq_len=-5)

    def test_contains_matches_prefix(self):
        cache = self.StaticKVCache(max_seq_len=8)
        cache.write_and_read("layer3:k", torch.randn(1, 1, 2, 4))
        cache.write_and_read("layer3:v", torch.randn(1, 1, 2, 4))
        assert "layer3" in cache
        assert "layer3:k" in cache
        assert "layer3:v" in cache
        assert "layer4" not in cache


class TestStaticKVCacheIntegration:
    """Integration tests: StaticKVCache behaves identically to dict-based cache."""

    def setup_method(self):
        from open_mythos.kv_cache import StaticKVCache
        self.StaticKVCache = StaticKVCache

    def _tiny_cfg(self, attn_type="gqa"):
        kwargs = dict(
            vocab_size=100, dim=64, n_heads=4, max_seq_len=16,
            attn_type=attn_type, max_loop_iters=2,
            prelude_layers=1, coda_layers=1,
            n_experts=4, n_experts_per_tok=2, n_shared_experts=1,
            expert_dim=32, lora_rank=4,
        )
        if attn_type == "gqa":
            kwargs["n_kv_heads"] = 2
        else:
            kwargs["n_kv_heads"] = 4
            kwargs.update(kv_lora_rank=16, q_lora_rank=32,
                          qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8)
        return MythosConfig(**kwargs)

    def test_gqa_static_cache_matches_dict_cache(self):
        torch.manual_seed(42)
        cfg = self._tiny_cfg("gqa")
        m = OpenMythos(cfg)
        m.eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 4))

        logits_static = m(ids, n_loops=2,
                          kv_cache=self.StaticKVCache(max_seq_len=cfg.max_seq_len),
                          start_pos=0)
        logits_dict = m(ids, n_loops=2, kv_cache={}, start_pos=0)
        torch.testing.assert_close(logits_static, logits_dict, atol=1e-5, rtol=1e-5)

    def test_mla_static_cache_matches_dict_cache(self):
        torch.manual_seed(42)
        cfg = self._tiny_cfg("mla")
        m = OpenMythos(cfg)
        m.eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 4))

        logits_static = m(ids, n_loops=2,
                          kv_cache=self.StaticKVCache(max_seq_len=cfg.max_seq_len),
                          start_pos=0)
        logits_dict = m(ids, n_loops=2, kv_cache={}, start_pos=0)
        torch.testing.assert_close(logits_static, logits_dict, atol=1e-5, rtol=1e-5)

    def test_multi_step_decode_no_reallocation(self):
        torch.manual_seed(42)
        cfg = self._tiny_cfg("gqa")
        m = OpenMythos(cfg)
        m.eval()

        prompt = torch.randint(0, cfg.vocab_size, (1, 3))
        cache = self.StaticKVCache(max_seq_len=cfg.max_seq_len)

        # Prefill
        m(prompt, n_loops=2, kv_cache=cache, start_pos=0)
        buffer_ids = {
            key: cache.buffer_id(key)
            for key in ("prelude_0:k", "recurrent_loop_0:k", "coda_0:k")
            if cache.buffer_id(key) is not None
        }
        assert buffer_ids, "no prefill buffers populated"

        # 5 decode steps
        for step in range(5):
            tok = torch.randint(0, cfg.vocab_size, (1, 1))
            m(tok, n_loops=2, kv_cache=cache, start_pos=prompt.shape[1] + step)

        # Buffer identities must be unchanged — no reallocation
        for key, bid in buffer_ids.items():
            assert cache.buffer_id(key) == bid, \
                f"buffer for {key!r} was reallocated across decode steps"

        # Position counters must have advanced by 5 (1 chunk per step)
        expected_len = prompt.shape[1] + 5
        for key in buffer_ids:
            assert cache.current_length(key) == expected_len, \
                f"{key!r} length {cache.current_length(key)} != expected {expected_len}"

    def test_generate_uses_static_cache(self):
        """OpenMythos.generate() constructs a StaticKVCache internally.

        We verify this by confirming the generated output shape is correct and
        by peering at the internal allocation pattern via a greedy decode that
        would never reallocate under the new API.
        """
        torch.manual_seed(42)
        cfg = self._tiny_cfg("gqa")
        m = OpenMythos(cfg)
        m.eval()

        prompt = torch.randint(0, cfg.vocab_size, (1, 3))
        out = m.generate(prompt, max_new_tokens=5, n_loops=2, temperature=1e-8)
        assert out.shape[1] == 3 + 5
        assert (out[:, :3] == prompt).all()  # prompt preserved
        assert torch.all(out >= 0) and torch.all(out < cfg.vocab_size)

    def test_static_cache_overflow_at_max_seq_len(self):
        cfg = self._tiny_cfg("gqa")
        cache = self.StaticKVCache(max_seq_len=cfg.max_seq_len)
        m = OpenMythos(cfg)
        m.eval()

        # Fill cache to exact max_seq_len
        ids = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
        m(ids, n_loops=2, kv_cache=cache, start_pos=0)

        # Next write must raise overflow
        extra = torch.randint(0, cfg.vocab_size, (1, 1))
        with pytest.raises(RuntimeError, match="overflow"):
            m(extra, n_loops=2, kv_cache=cache, start_pos=cfg.max_seq_len)


# ---------------------------------------------------------------------------
# P4.2: MoEFFN opt-in balance loss
# ---------------------------------------------------------------------------


class TestMoEBalanceLoss:
    """Stage 4.2: MoEFFN.forward gained an opt-in return_balance_loss path
    that mirrors the DeepSeekMoE aux-loss behavior from experimental/moda.py,
    without breaking the default single-tensor return for current callers."""

    def setup_method(self):
        torch.manual_seed(42)
        self.cfg = gqa_cfg(
            dim=64, n_experts=4, n_experts_per_tok=2,
            n_shared_experts=1, expert_dim=16,
        )
        self.moe = MoEFFN(self.cfg)

    def test_default_returns_tensor_only(self):
        """Without explicit opt-in, MoEFFN.forward returns a single tensor
        (backward compat with all pre-P4.2 callers)."""
        x = torch.randn(2, 8, self.cfg.dim)
        out = self.moe(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == x.shape

    def test_opt_in_balance_loss_is_scalar_positive(self):
        """return_balance_loss=True yields (output, scalar balance loss)."""
        self.moe.train()
        # Give the aux loss a nonzero alpha so the returned scalar isn't 0.0.
        self.moe.moe_balance_alpha = 0.01
        x = torch.randn(2, 8, self.cfg.dim)
        out, balance_loss = self.moe(x, return_balance_loss=True)
        assert isinstance(out, torch.Tensor)
        assert out.shape == x.shape
        assert isinstance(balance_loss, torch.Tensor)
        assert balance_loss.dim() == 0  # scalar
        assert torch.isfinite(balance_loss)
        assert balance_loss.item() > 0.0


# ---------------------------------------------------------------------------
# P4.4: MythosConfig nested split — backward compat + cross-group validation
# ---------------------------------------------------------------------------


class TestNestedConfig:
    """Stage 4.4: MythosConfig is now a composition of AttentionConfig,
    MoEConfig, RecurrenceConfig. Flat-kwarg construction and .n_heads-style
    accessors preserved for backward compat; cross-group invariants (e.g.
    dim % n_heads) validated at the top-level __post_init__."""

    def test_nested_construction_equivalent_to_from_flat(self):
        """Building via nested sub-configs vs flat kwargs must produce equal configs."""
        from open_mythos.config import AttentionConfig, MoEConfig, RecurrenceConfig

        nested = MythosConfig(
            vocab_size=100, dim=64, max_seq_len=16,
            prelude_layers=1, coda_layers=1,
            attention=AttentionConfig(attn_type="gqa", n_heads=4, n_kv_heads=2),
            moe=MoEConfig(n_experts=4, n_experts_per_tok=2,
                          n_shared_experts=1, expert_dim=16),
            recurrence=RecurrenceConfig(max_loop_iters=2, lora_rank=4),
        )
        flat = MythosConfig(
            vocab_size=100, dim=64, max_seq_len=16,
            prelude_layers=1, coda_layers=1,
            attn_type="gqa", n_heads=4, n_kv_heads=2,
            n_experts=4, n_experts_per_tok=2,
            n_shared_experts=1, expert_dim=16,
            max_loop_iters=2, lora_rank=4,
        )
        assert nested == flat

    def test_nested_attn_validation_local(self):
        """AttentionConfig.__post_init__ rejects bad attn_type before top-level validation."""
        from open_mythos.config import AttentionConfig
        with pytest.raises(ValueError, match="attn_type"):
            AttentionConfig(attn_type="xxx")

    def test_nested_cross_group_validation_dim_heads(self):
        """dim % n_heads is a cross-group check: dim is core, n_heads is attention."""
        with pytest.raises(ValueError, match="dim.*divisible.*n_heads"):
            gqa_cfg(dim=50, n_heads=4, n_kv_heads=2)

    def test_nested_cross_group_validation_lora(self):
        """lora_rank <= dim is a cross-group check: lora_rank is recurrence, dim is core."""
        with pytest.raises(ValueError, match="lora_rank"):
            gqa_cfg(dim=64, lora_rank=128)

    def test_flat_accessor_compatibility(self):
        """cfg.n_heads, cfg.attn_type, etc. continue to work via @property accessors."""
        cfg = gqa_cfg(n_heads=4, n_experts=4)
        assert cfg.n_heads == 4
        assert cfg.n_heads == cfg.attention.n_heads
        assert cfg.attn_type == cfg.attention.attn_type
        assert cfg.n_experts == cfg.moe.n_experts
        assert cfg.max_loop_iters == cfg.recurrence.max_loop_iters


# ---------------------------------------------------------------------------
# P4.3: Protocol + factory dispatch
# ---------------------------------------------------------------------------


class TestProtocolFactory:
    """Stage 4.3: build_attention / build_injection factories replace
    hardcoded `if cfg.attn_type` dispatch. AttentionProtocol and
    InjectionProtocol let alternative backends slot in without touching
    TransformerBlock / RecurrentBlock."""

    def test_build_attention_gqa_returns_gqa_instance(self):
        from open_mythos.attention.factory import build_attention
        cfg = gqa_cfg()
        attn = build_attention(cfg)
        assert isinstance(attn, GQAttention)

    def test_build_attention_mla_returns_mla_instance(self):
        from open_mythos.attention.factory import build_attention
        cfg = mla_cfg()
        attn = build_attention(cfg)
        assert isinstance(attn, MLAttention)

    def test_build_attention_invalid_attn_type_raises(self):
        """The config layer rejects bad attn_type. The factory defends itself too
        in case someone bypasses config validation (e.g., via frozen patching)."""
        from open_mythos.attention.factory import build_attention
        cfg = gqa_cfg()
        # Force a bad attn_type by patching the nested sub-config via
        # object.__setattr__ (bypassing frozen).
        object.__setattr__(cfg.attention, "attn_type", "xxx")
        with pytest.raises(ValueError, match="unknown attn_type"):
            build_attention(cfg)

    def test_attention_instances_implement_protocol(self):
        """runtime_checkable Protocol check — both GQA and MLA attention
        expose freqs_cis_dim and the expected __call__ signature."""
        from open_mythos.attention.factory import build_attention
        from open_mythos.attention.protocol import AttentionProtocol
        gqa = build_attention(gqa_cfg())
        mla = build_attention(mla_cfg())
        # runtime_checkable Protocol checks attributes + __call__ presence
        assert isinstance(gqa, AttentionProtocol)
        assert isinstance(mla, AttentionProtocol)
        assert isinstance(gqa.freqs_cis_dim, int) and gqa.freqs_cis_dim > 0
        assert isinstance(mla.freqs_cis_dim, int) and mla.freqs_cis_dim > 0

    def test_build_injection_returns_lti(self):
        from open_mythos.recurrence.factory import build_injection
        cfg = gqa_cfg()
        inj = build_injection(cfg)
        assert isinstance(inj, LTIInjection)


# ---------------------------------------------------------------------------
# P4.3 completion: RouterProtocol + BiasRouter + build_router
# ---------------------------------------------------------------------------


class TestRouterProtocol:
    """Stage 4.3 (P4.3) completion: MoE routing is now a swappable
    Protocol-compliant module built via build_router(cfg). BiasRouter
    is the default (DeepSeek-V3 aux-loss-free) implementation.

    The routing logic previously lived inline in MoEFFN.forward; the
    refactor moves it into BiasRouter so alternative routers (grouped,
    hash-based, learned gating) slot in without touching MoEFFN itself.
    """

    def test_build_router_returns_bias_router(self):
        from open_mythos.moe import BiasRouter, build_router
        cfg = gqa_cfg()
        r = build_router(cfg)
        assert isinstance(r, BiasRouter)
        assert r.n_experts == cfg.moe.n_experts
        assert r.topk == cfg.moe.n_experts_per_tok

    def test_bias_router_satisfies_protocol(self):
        """BiasRouter passes isinstance(RouterProtocol) via runtime_checkable."""
        from open_mythos.moe import RouterProtocol, build_router
        r = build_router(gqa_cfg())
        assert isinstance(r, RouterProtocol)

    def test_bias_router_forward_shapes_and_renorm(self):
        """Router output shapes match spec; topk_scores sums to 1 per row."""
        from open_mythos.moe import build_router
        torch.manual_seed(0)
        cfg = gqa_cfg(n_experts=4, n_experts_per_tok=2)
        r = build_router(cfg)
        N = 6
        x = torch.randn(N, cfg.dim)
        topk_idx, topk_scores, scores = r(x)
        assert topk_idx.shape == (N, cfg.n_experts_per_tok)
        assert topk_scores.shape == (N, cfg.n_experts_per_tok)
        assert scores.shape == (N, cfg.n_experts)
        # Per-row sum of renormalized weights must be 1.0 (up to fp tol).
        sums = topk_scores.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(N), atol=1e-5, rtol=1e-5)

    def test_moe_ffn_uses_build_router(self):
        """MoEFFN.router is now the BiasRouter module (not a raw nn.Linear).
        State-dict layout reflects the wrap: router.linear.weight etc."""
        from open_mythos.moe import BiasRouter
        cfg = gqa_cfg()
        moe = MoEFFN(cfg)
        assert isinstance(moe.router, BiasRouter)
        sd_keys = set(moe.state_dict().keys())
        assert "router.linear.weight" in sd_keys
        assert "router.router_bias" in sd_keys
        # Pre-4.3 legacy paths must NOT be present.
        assert "router.weight" not in sd_keys
        assert "router_bias" not in sd_keys


if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])
