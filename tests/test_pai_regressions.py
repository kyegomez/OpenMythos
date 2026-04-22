"""Regression tests for fixes landed in the PAI review 2026-04 branch.

Each test covers one previously-unguarded failure mode documented in
PAI_REVIEW_2026-04.md.
"""

from __future__ import annotations

import pytest
import torch

from open_mythos.main import (
    LoRAAdapter,
    LTIInjection,
    MythosConfig,
    OpenMythos,
    RecurrentBlock,
    loop_index_embedding,
)


# ---------------------------------------------------------------------------
# MythosConfig validation
# ---------------------------------------------------------------------------


def test_config_rejects_unknown_attn_type():
    with pytest.raises(ValueError, match="attn_type"):
        MythosConfig(attn_type="bogus")  # type: ignore[arg-type]


def test_config_rejects_indivisible_heads():
    with pytest.raises(ValueError, match="n_kv_heads"):
        MythosConfig(n_heads=16, n_kv_heads=5)


def test_config_rejects_topk_gt_experts():
    with pytest.raises(ValueError, match="n_experts_per_tok"):
        MythosConfig(n_experts=4, n_experts_per_tok=8)


def test_config_rejects_odd_head_dim():
    with pytest.raises(ValueError, match="head_dim"):
        MythosConfig(dim=30, n_heads=3, n_kv_heads=3)


def test_config_rejects_zero_loops():
    with pytest.raises(ValueError, match="max_loop_iters"):
        MythosConfig(max_loop_iters=0)


# ---------------------------------------------------------------------------
# LTI stability under bf16
# ---------------------------------------------------------------------------


def test_lti_get_a_strictly_in_open_unit_interval_bf16():
    """Even when log_dt + log_A produces an input that bf16 would underflow,
    the fp32 compute inside get_A keeps A strictly in (0, 1)."""
    lti = LTIInjection(dim=8).to(torch.bfloat16)
    # Push the param toward the clamp boundary.
    with torch.no_grad():
        lti.log_dt.fill_(-9.0)
        lti.log_A.fill_(9.0)
    a = lti.get_A()
    assert a.min().item() > 0.0
    assert a.max().item() < 1.0


# ---------------------------------------------------------------------------
# loop_index_embedding determinism across k
# ---------------------------------------------------------------------------


def test_loop_index_embedding_distinct_freqs_in_bf16():
    """Under bf16 inputs, frequencies must still be computed in fp32 so that
    adjacent channel pairs carry distinct sin/cos values."""
    h = torch.zeros(1, 1, 64, dtype=torch.bfloat16)
    out = loop_index_embedding(h, loop_t=1, loop_dim=32)
    # Slice the embedding portion; distinct pairs must not collapse.
    emb = out[0, 0, :32]
    assert emb.unique().numel() > 8  # at least half the pairs should differ


# ---------------------------------------------------------------------------
# Weight tying init order
# ---------------------------------------------------------------------------


def test_head_and_embed_share_storage():
    cfg = MythosConfig(
        vocab_size=256, dim=64, n_heads=4, n_kv_heads=2,
        max_seq_len=32, max_loop_iters=2,
        prelude_layers=1, coda_layers=1,
        n_experts=4, n_experts_per_tok=2, expert_dim=32,
        attn_type="gqa",
    )
    model = OpenMythos(cfg)
    assert model.head.weight.data_ptr() == model.embed.weight.data_ptr()


# ---------------------------------------------------------------------------
# generate() bounds + EOS + eval mode
# ---------------------------------------------------------------------------


def _tiny_model() -> OpenMythos:
    cfg = MythosConfig(
        vocab_size=32, dim=32, n_heads=4, n_kv_heads=2,
        max_seq_len=16, max_loop_iters=2,
        prelude_layers=1, coda_layers=1,
        n_experts=4, n_experts_per_tok=2, expert_dim=16,
        attn_type="gqa", dropout=0.5,
    )
    return OpenMythos(cfg)


def test_forward_rejects_empty_input():
    model = _tiny_model()
    with pytest.raises(ValueError, match="non-empty"):
        model(torch.zeros(1, 0, dtype=torch.long))


def test_forward_rejects_over_max_seq_len():
    model = _tiny_model()
    ids = torch.zeros(1, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="max_seq_len"):
        model(ids, start_pos=model.cfg.max_seq_len)


def test_generate_clamps_to_max_seq_len():
    model = _tiny_model()
    prompt = torch.zeros(1, 10, dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=100, n_loops=1)
    # budget = 16 - 10 = 6 new tokens max
    assert out.shape[1] <= model.cfg.max_seq_len


def test_generate_returns_training_mode():
    model = _tiny_model()
    model.train()
    assert model.training is True
    prompt = torch.zeros(1, 2, dtype=torch.long)
    model.generate(prompt, max_new_tokens=2, n_loops=1)
    assert model.training is True  # restored


def test_generate_eos_stops_early():
    model = _tiny_model()
    model.eval()
    # Force eos by making it the argmax: bias the head to token 0.
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.weight[0].fill_(10.0)
    prompt = torch.zeros(1, 2, dtype=torch.long)
    out = model.generate(
        prompt, max_new_tokens=10, n_loops=1,
        temperature=1.0, top_k=1, eos_token_id=0,
    )
    # first generated token is 0, so finished.all() → break after step 0
    assert out.shape[1] == 3


# ---------------------------------------------------------------------------
# LoRA clamp for depth extrapolation
# ---------------------------------------------------------------------------


def test_lora_clamps_beyond_max_loops():
    adapter = LoRAAdapter(dim=16, rank=4, max_loops=3)
    x = torch.randn(1, 2, 16)
    a = adapter(x, loop_t=2)  # last trained index
    b = adapter(x, loop_t=10)  # out of range → clamp
    assert torch.allclose(a, b)
