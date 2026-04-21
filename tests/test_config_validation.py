"""
Validation of `MythosConfig.__post_init__`.

The `__post_init__` guards exist so a bad hyperparameter combination fails
at *config construction* time instead of halfway through a multi-day
pretraining run. Every test here pokes one axis at a time so regressions
in the validator pinpoint exactly which rule broke.
"""

import pytest

from open_mythos.main import MythosConfig


def _base(**overrides) -> dict:
    """
    Minimal kwargs that produce a valid config. Overrides mutate one axis;
    other tests rely on this being a known-good baseline.
    """
    cfg = dict(
        vocab_size=128,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=32,
    )
    cfg.update(overrides)
    return cfg


def test_baseline_is_valid() -> None:
    MythosConfig(**_base())


def test_attn_type_invalid_raises() -> None:
    with pytest.raises(ValueError, match="attn_type"):
        MythosConfig(**_base(attn_type="linear"))


def test_dim_not_divisible_by_n_heads_raises() -> None:
    with pytest.raises(ValueError, match="dim"):
        MythosConfig(**_base(dim=65, n_heads=4))


def test_gqa_divisibility_raises() -> None:
    # 5 Q heads / 2 KV groups → 5 not divisible by 2
    with pytest.raises(ValueError, match="n_heads"):
        MythosConfig(**_base(n_heads=5, n_kv_heads=2, dim=80))


def test_mla_odd_rope_head_dim_raises() -> None:
    with pytest.raises(ValueError, match=r"qk_rope_head_dim"):
        MythosConfig(
            **_base(
                attn_type="mla",
                kv_lora_rank=16,
                q_lora_rank=16,
                qk_rope_head_dim=15,  # odd
                qk_nope_head_dim=16,
                v_head_dim=16,
            )
        )


def test_moe_experts_per_tok_exceeds_n_experts_raises() -> None:
    with pytest.raises(ValueError, match="experts_per_tok"):
        MythosConfig(**_base(n_experts=4, n_experts_per_tok=5))


def test_dropout_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="dropout"):
        MythosConfig(**_base(dropout=1.5))


def test_negative_bias_update_speed_raises() -> None:
    with pytest.raises(ValueError, match="bias_update_speed"):
        MythosConfig(**_base(bias_update_speed=-1e-3))


def test_non_positive_init_std_raises() -> None:
    with pytest.raises(ValueError, match="init_std"):
        MythosConfig(**_base(init_std=0.0))


def test_non_positive_vocab_raises() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        MythosConfig(**_base(vocab_size=0))


def test_non_positive_max_seq_len_raises() -> None:
    with pytest.raises(ValueError, match="max_seq_len"):
        MythosConfig(**_base(max_seq_len=0))
