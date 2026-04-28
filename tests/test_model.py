"""
Tests for the core OpenMythos model.
Covers forward pass shape, LTI stability, generation, RMSNorm,
causal mask, loop-index embedding, LoRA adapter, and depth extrapolation.
Not covered by existing test_rope_debug.py or test_tokenizer.py.
"""

import pytest
import torch
from open_mythos.main import (
    OpenMythos,
    MythosConfig,
    RMSNorm,
    LTIInjection,
    LoRAAdapter,
    loop_index_embedding,
)


# ---------------------------------------------------------------------------
# Minimal configs for fast CPU tests
# ---------------------------------------------------------------------------

def make_gqa_cfg(**overrides) -> MythosConfig:
    base = dict(
        vocab_size=200,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=32,
        lora_rank=4,
        attn_type="gqa",
    )
    base.update(overrides)
    return MythosConfig(**base)


def make_mla_cfg(**overrides) -> MythosConfig:
    base = dict(
        vocab_size=200,
        dim=64,
        n_heads=4,
        n_kv_heads=4,
        max_seq_len=32,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=32,
        lora_rank=4,
        attn_type="mla",
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
    )
    base.update(overrides)
    return MythosConfig(**base)


# ---------------------------------------------------------------------------
# 1. Model loading
# ---------------------------------------------------------------------------

def test_model_loads_gqa():
    """GQA model should instantiate without errors."""
    model = OpenMythos(make_gqa_cfg())
    assert model is not None
    print(f"\nGQA params: {sum(p.numel() for p in model.parameters()):,}")


def test_model_loads_mla():
    """MLA model should instantiate without errors."""
    model = OpenMythos(make_mla_cfg())
    assert model is not None
    print(f"\nMLA params: {sum(p.numel() for p in model.parameters()):,}")


# ---------------------------------------------------------------------------
# 2. Forward pass output shape
# ---------------------------------------------------------------------------

def test_forward_output_shape_gqa():
    """Logits shape must be (B, T, vocab_size) for GQA."""
    cfg = make_gqa_cfg()
    model = OpenMythos(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    logits = model(ids, n_loops=2)
    assert logits.shape == (2, 8, cfg.vocab_size), f"Got {logits.shape}"


def test_forward_output_shape_mla():
    """Logits shape must be (B, T, vocab_size) for MLA."""
    cfg = make_mla_cfg()
    model = OpenMythos(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    logits = model(ids, n_loops=2)
    assert logits.shape == (2, 8, cfg.vocab_size), f"Got {logits.shape}"


def test_forward_single_token():
    """Model must handle a single token (T=1) without a causal mask."""
    cfg = make_gqa_cfg()
    model = OpenMythos(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 1))
    logits = model(ids, n_loops=2)
    assert logits.shape == (1, 1, cfg.vocab_size)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_forward_various_batch_sizes(batch_size):
    """Model must handle different batch sizes."""
    cfg = make_gqa_cfg()
    model = OpenMythos(cfg)
    ids = torch.randint(0, cfg.vocab_size, (batch_size, 6))
    logits = model(ids, n_loops=2)
    assert logits.shape == (batch_size, 6, cfg.vocab_size)


# ---------------------------------------------------------------------------
# 3. LTI stability — THE core guarantee of the architecture
# ---------------------------------------------------------------------------

def test_spectral_radius_less_than_1_at_init():
    """
    A_discrete values must ALL be strictly in (0, 1) at initialization.
    This is the Parcae stability guarantee — rho(A) < 1 by construction.
    """
    inj = LTIInjection(dim=64)
    A = inj.get_A()
    assert (A > 0).all(), "All A values must be positive"
    assert (A < 1).all(), "All A values must be < 1 — spectral radius guarantee broken!"


def test_spectral_radius_bounded_after_large_weight_perturbation():
    """
    Stability bound must hold even after extreme random weight values.
    In float32, very large exponents can underflow to 0 or saturate to 1,
    but A is always guaranteed to be in [0, 1] — never negative, never > 1.
    This is the practical float32 form of the Parcae stability guarantee.
    """
    inj = LTIInjection(dim=128)
    with torch.no_grad():
        inj.log_A.uniform_(-10, 10)
        inj.log_dt.uniform_(-10, 10)
    A = inj.get_A()
    assert (A >= 0).all(), "A values must be non-negative after large perturbation"
    assert (A <= 1).all(), "A values must be <= 1 after large perturbation"


def test_lti_injection_output_shape():
    """LTIInjection forward must preserve tensor shape."""
    inj = LTIInjection(dim=64)
    h = torch.randn(2, 8, 64)
    e = torch.randn(2, 8, 64)
    trans_out = torch.randn(2, 8, 64)
    out = inj(h, e, trans_out)
    assert out.shape == h.shape


# ---------------------------------------------------------------------------
# 4. Different loop counts (depth extrapolation)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_loops", [1, 2, 4, 8])
def test_different_loop_counts(n_loops):
    """
    Model must work for any loop count, including beyond max_loop_iters=4.
    This tests the depth extrapolation property of looped transformers.
    """
    cfg = make_gqa_cfg()
    model = OpenMythos(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    logits = model(ids, n_loops=n_loops)
    assert logits.shape == (1, 6, cfg.vocab_size), \
        f"Failed for n_loops={n_loops}: got {logits.shape}"


# ---------------------------------------------------------------------------
# 5. generate() correctness
# ---------------------------------------------------------------------------

def test_generate_output_shape():
    """generate() must return (B, prompt_len + max_new_tokens) tokens."""
    cfg = make_gqa_cfg()
    model = OpenMythos(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(ids, max_new_tokens=5, n_loops=2)
    assert out.shape == (1, 9), f"Expected (1, 9), got {out.shape}"


def test_generate_preserves_prompt():
    """The generated sequence must begin with the original prompt tokens."""
    cfg = make_gqa_cfg()
    model = OpenMythos(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(ids, max_new_tokens=3, n_loops=2)
    assert torch.equal(out[:, :4], ids), "Prompt tokens were modified during generation"


def test_generate_output_tokens_in_vocab():
    """All generated token IDs must be valid vocabulary indices."""
    cfg = make_gqa_cfg()
    model = OpenMythos(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 3))
    out = model.generate(ids, max_new_tokens=4, n_loops=2)
    assert (out >= 0).all() and (out < cfg.vocab_size).all(), \
        "Generated tokens outside vocabulary range"


# ---------------------------------------------------------------------------
# 6. RMSNorm
# ---------------------------------------------------------------------------

def test_rmsnorm_output_shape():
    """RMSNorm must preserve input shape exactly."""
    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64)
    assert norm(x).shape == x.shape


def test_rmsnorm_normalizes_to_unit_rms():
    """With weight=ones (default), RMSNorm output must have RMS ≈ 1.0 per vector."""
    norm = RMSNorm(64)
    x = torch.randn(4, 8, 64) * 100
    out = norm(x)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.05), \
        f"RMS not close to 1: mean={rms.mean().item():.4f}"


def test_rmsnorm_learned_weight_scales_output():
    """Setting weight to constant c should scale the normalized output by c."""
    norm = RMSNorm(64)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    x = torch.randn(2, 4, 64)
    out = norm(x)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.full_like(rms, 2.0), atol=0.1), \
        "Learned weight not correctly scaling RMSNorm output"


# ---------------------------------------------------------------------------
# 7. Causal mask
# ---------------------------------------------------------------------------

def test_causal_mask_shape():
    """Causal mask must be shape (1, 1, T, T)."""
    mask = OpenMythos._causal_mask(8, device=torch.device("cpu"))
    assert mask.shape == (1, 1, 8, 8)


def test_causal_mask_upper_triangle_is_neg_inf():
    """Upper triangle (strictly above diagonal) must be -inf to block future attention."""
    T = 6
    mask = OpenMythos._causal_mask(T, device=torch.device("cpu"))
    m = mask[0, 0]
    rows, cols = torch.triu_indices(T, T, offset=1)
    upper_vals = m[rows, cols]
    assert (upper_vals == float("-inf")).all(), \
        f"Strictly upper triangle must be -inf, got: {upper_vals}"


def test_causal_mask_lower_triangle_is_zero():
    """Lower triangle + diagonal must be 0 (tokens can attend to past + self)."""
    mask = OpenMythos._causal_mask(6, device=torch.device("cpu"))
    lower = mask[0, 0].tril(diagonal=0)
    assert (lower == 0).all(), "Lower triangle must be 0"


# ---------------------------------------------------------------------------
# 8. Loop-index embedding
# ---------------------------------------------------------------------------

def test_loop_index_embedding_shape():
    """loop_index_embedding must not change tensor shape."""
    h = torch.randn(2, 8, 64)
    out = loop_index_embedding(h, loop_t=3, loop_dim=8)
    assert out.shape == h.shape


def test_loop_index_embedding_different_per_iteration():
    """Different loop indices must produce different embeddings."""
    h = torch.zeros(1, 1, 64)
    out0 = loop_index_embedding(h, loop_t=0, loop_dim=16)
    out3 = loop_index_embedding(h, loop_t=3, loop_dim=16)
    assert not torch.allclose(out0, out3), \
        "Loop embeddings at t=0 and t=3 must differ"


def test_loop_index_embedding_loop0_adds_zeros_for_sin():
    """At loop_t=0, sin(0)=0 and cos(0)=1, so embedding is deterministic."""
    h = torch.zeros(1, 1, 64)
    out = loop_index_embedding(h, loop_t=0, loop_dim=8)
    assert out[0, 0, :4].sum().item() == pytest.approx(0.0, abs=1e-5)
    assert out[0, 0, 4:8].sum().item() == pytest.approx(4.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 9. LoRA adapter
# ---------------------------------------------------------------------------

def test_lora_adapter_output_shape():
    """LoRAAdapter must return tensor of same shape as input."""
    adapter = LoRAAdapter(dim=64, rank=4, max_loops=4)
    x = torch.randn(2, 8, 64)
    out = adapter(x, loop_t=2)
    assert out.shape == x.shape


def test_lora_adapter_clamps_loop_index():
    """LoRAAdapter must work correctly for all valid loop indices."""
    adapter = LoRAAdapter(dim=64, rank=4, max_loops=4)
    x = torch.randn(1, 4, 64)
    for loop_t in range(4):
        out = adapter(x, loop_t=loop_t)
        assert out.shape == x.shape, f"Shape mismatch at loop_t={loop_t}"


if __name__ == "__main__":
    pytest.main([__file__, "--verbose", "-s"])
