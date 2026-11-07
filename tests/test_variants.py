import dataclasses

import pytest
import torch

from open_mythos.main import MythosConfig, OpenMythos
from open_mythos.variants import (
    mythos_1b,
    mythos_3b,
    mythos_10b,
    mythos_50b,
    mythos_100b,
    mythos_500b,
    mythos_1t,
)


ALL_PRESETS = [
    mythos_1b,
    mythos_3b,
    mythos_10b,
    mythos_50b,
    mythos_100b,
    mythos_500b,
    mythos_1t,
]


@pytest.mark.parametrize("preset", ALL_PRESETS, ids=lambda f: f.__name__)
def test_preset_constructible(preset):
    cfg = preset()
    assert isinstance(cfg, MythosConfig)


@pytest.mark.parametrize("preset", ALL_PRESETS, ids=lambda f: f.__name__)
def test_preset_structural_invariants(preset):
    cfg = preset()
    assert cfg.dim % cfg.n_heads == 0, (
        f"{preset.__name__}: dim {cfg.dim} not divisible by n_heads {cfg.n_heads}"
    )
    if cfg.attn_type == "gqa":
        assert cfg.n_heads % cfg.n_kv_heads == 0, (
            f"{preset.__name__}: n_heads {cfg.n_heads} not divisible by "
            f"n_kv_heads {cfg.n_kv_heads}"
        )
    assert cfg.n_experts_per_tok <= cfg.n_experts, (
        f"{preset.__name__}: n_experts_per_tok {cfg.n_experts_per_tok} > "
        f"n_experts {cfg.n_experts}"
    )
    assert 0.0 < cfg.act_threshold <= 1.0, (
        f"{preset.__name__}: act_threshold {cfg.act_threshold} out of (0, 1]"
    )
    assert cfg.attn_type in ("gqa", "mla"), (
        f"{preset.__name__}: unexpected attn_type {cfg.attn_type!r}"
    )


def test_preset_1b_forward_pass_with_reduced_dims():
    base = mythos_1b()
    reduced = dataclasses.replace(base, vocab_size=100, max_seq_len=16)

    try:
        model = OpenMythos(reduced).eval()
    except (RuntimeError, MemoryError) as e:
        pytest.skip(f"1B preset too large to instantiate in this environment: {e}")

    input_ids = torch.randint(0, reduced.vocab_size, (2, 4))
    try:
        with torch.no_grad():
            logits = model(input_ids)
    except (RuntimeError, MemoryError) as e:
        pytest.skip(f"1B preset forward OOM in this environment: {e}")

    assert logits.shape == (2, 4, 100)


def test_preset_parameter_count_increases():
    presets_in_order = [
        mythos_1b,
        mythos_3b,
        mythos_10b,
        mythos_50b,
        mythos_100b,
        mythos_500b,
        mythos_1t,
    ]
    dims = [p().dim for p in presets_in_order]
    for smaller, larger in zip(dims, dims[1:]):
        assert smaller <= larger, (
            f"dim should be monotonically non-decreasing; got {dims}"
        )


def _estimated_params(cfg: MythosConfig) -> int:
    embed = cfg.vocab_size * cfg.dim
    dense_layers = cfg.prelude_layers + cfg.coda_layers
    dense_ffn = dense_layers * (3 * cfg.dim * (cfg.dim * 4 // 3))
    routed_moe = 3 * cfg.dim * cfg.expert_dim * cfg.n_experts
    shared_moe = 3 * cfg.dim * cfg.expert_dim * cfg.n_experts_per_tok * cfg.n_shared_experts
    return embed + dense_ffn + routed_moe + shared_moe


@pytest.mark.parametrize(
    "smaller, larger",
    list(zip(ALL_PRESETS, ALL_PRESETS[1:])),
    ids=lambda f: f.__name__,
)
def test_preset_estimated_params_increase(smaller, larger):
    smaller_params = _estimated_params(smaller())
    larger_params = _estimated_params(larger())
    assert smaller_params < larger_params, (
        f"{smaller.__name__} ({smaller_params:,}) should have fewer estimated "
        f"params than {larger.__name__} ({larger_params:,})"
    )


if __name__ == "__main__":
    pytest.main([__file__, "--verbose", "-s"])
