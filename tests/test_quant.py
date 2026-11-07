import pytest
import torch

from open_mythos.config import MythosConfig, QuantConfig
from open_mythos.primitives.quant import (
    BitLinear,
    absmax_act_quant,
    absmean_quantize,
)


def test_absmean_quantize_maps_to_ternary_domain():
    torch.manual_seed(0)
    weight = torch.randn(64, 32)

    ternary, gamma = absmean_quantize(weight)

    assert set(ternary.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert torch.isclose(gamma, weight.abs().mean())


def test_absmean_quantize_clamps_scale_on_constant_zero_weight():
    ternary, gamma = absmean_quantize(torch.zeros(4, 4))

    assert torch.all(ternary == 0)
    assert torch.isfinite(gamma) and gamma > 0


def test_absmean_quantize_sparsity_is_realistic_for_gaussian_weights():
    # absmean (b1.58) yields roughly a third zeros on Gaussian weights.
    # The ~70% figures seen elsewhere come from synthetic hard-coded
    # sparsity, not from this scheme.
    torch.manual_seed(4)
    ternary, _ = absmean_quantize(torch.randn(128, 128))

    zeros = (ternary == 0).float().mean().item()
    assert 0.25 < zeros < 0.40


def test_bitlinear_disabled_matches_linear_exactly():
    torch.manual_seed(1)
    layer = BitLinear(8, 4, enabled=False)
    reference = torch.nn.Linear(8, 4, bias=False)
    reference.weight.data.copy_(layer.weight.data)
    x = torch.randn(2, 8)

    assert torch.equal(layer(x), reference(x))


def test_bitlinear_forward_uses_dequantized_ternary_weights():
    torch.manual_seed(2)
    layer = BitLinear(8, 4)
    ternary, gamma = layer.quantized_weight()
    x = torch.randn(3, 8)

    expected = x @ (ternary * gamma).T

    assert set(ternary.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert torch.allclose(layer(x), expected)


def test_bitlinear_ste_gradient_matches_linear_surrogate():
    torch.manual_seed(3)
    quant = BitLinear(16, 8)
    linear = torch.nn.Linear(16, 8, bias=False)
    linear.weight.data.copy_(quant.weight.data)
    x = torch.randn(4, 16)
    upstream = torch.randn(4, 8)

    quant(x).backward(upstream)
    linear(x).backward(upstream)

    assert quant.weight.grad is not None
    assert linear.weight.grad is not None
    assert torch.allclose(
        quant.weight.grad, linear.weight.grad, rtol=1e-5, atol=1e-7
    )


def test_quantize_dequantize_roundtrip_stabilizes_ternary_values():
    torch.manual_seed(5)
    weight = torch.randn(32, 16)

    ternary, gamma = absmean_quantize(weight)
    ternary_again, _ = absmean_quantize(ternary * gamma)

    assert torch.equal(ternary, ternary_again)
    assert torch.isfinite(gamma)


def test_absmax_act_quant_whole_row_matches_int8_domain():
    torch.manual_seed(6)
    x = torch.randn(4, 64)

    q, scale = absmax_act_quant(x, bits=8)

    assert scale.shape == (4, 1)
    assert q.abs().max() <= 127.0
    reconstructed = (q.reshape(4, 1, 64) * scale.reshape(4, 1, 1)).reshape(4, 64)
    assert (reconstructed - x).abs().max() <= 0.51 * scale.max()


def test_absmax_act_quant_grouped_partition():
    torch.manual_seed(7)
    x = torch.randn(2, 32)

    q, scale = absmax_act_quant(x, bits=8, group_size=8)

    assert scale.shape == (2, 4)
    grouped = q.reshape(2, 4, 8)
    reconstructed = (grouped * scale.unsqueeze(-1)).reshape(2, 32)
    assert (reconstructed - x).abs().max() <= 0.51 * scale.max()


def test_absmax_act_quant_rejects_indivisible_group_size():
    with pytest.raises(ValueError, match="divisible"):
        absmax_act_quant(torch.randn(2, 10), bits=8, group_size=4)


def test_absmax_act_quant_rejects_too_few_bits():
    with pytest.raises(ValueError, match="bits"):
        absmax_act_quant(torch.randn(2, 8), bits=1)


def test_bitlinear_mlp_trains_loss_decreases():
    torch.manual_seed(8)
    model = torch.nn.Sequential(
        BitLinear(16, 32),
        torch.nn.ReLU(),
        BitLinear(32, 8),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    torch.manual_seed(9)
    x = torch.randn(64, 16)
    target = torch.randn(64, 8)

    first_loss = last_loss = None
    for step in range(30):
        optimizer.zero_grad()
        loss = ((model(x) - target) ** 2).mean()
        if step == 0:
            first_loss = loss.item()
        last_loss = loss.item()
        loss.backward()
        optimizer.step()

    assert last_loss is not None and first_loss is not None
    assert last_loss < first_loss * 0.95


def test_config_defaults_to_no_quantization():
    cfg = MythosConfig()

    assert cfg.quant.mode == "none"
    assert cfg.quant_mode == "none"
    assert cfg.quant_group_size == 128


def test_config_accepts_flat_quant_kwargs():
    cfg = MythosConfig(quant_mode="w158", quant_group_size=64)

    assert cfg.quant.mode == "w158"
    assert cfg.quant.group_size == 64


def test_config_accepts_nested_quant_config():
    cfg = MythosConfig(quant=QuantConfig(mode="w158", group_size=32))

    assert cfg.quant_mode == "w158"
    assert cfg.quant_group_size == 32


def test_config_rejects_unknown_quant_mode():
    with pytest.raises(ValueError, match="quant mode"):
        MythosConfig(quant_mode="int4")


def test_config_rejects_w158_with_nonpositive_group_size():
    with pytest.raises(ValueError, match="group_size"):
        MythosConfig(quant_mode="w158", quant_group_size=0)


def test_config_rejects_mixed_flat_and_nested_quant_kwargs():
    with pytest.raises(TypeError, match="flat and nested"):
        MythosConfig(quant=QuantConfig(), quant_mode="w158")


def _tiny_cfg(quant_mode: str):
    return MythosConfig(
        vocab_size=64,
        dim=32,
        max_seq_len=16,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_heads=4,
        n_kv_heads=2,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=16,
        max_loop_iters=2,
        lora_rank=4,
        quant_mode=quant_mode,
    )


def test_build_linear_selects_bitlinear_under_w158():
    from open_mythos.primitives.quant import build_linear

    assert isinstance(build_linear(_tiny_cfg("w158"), 8, 4), BitLinear)
    quant_free = build_linear(_tiny_cfg("none"), 8, 4)
    assert type(quant_free) is torch.nn.Linear


def test_w158_model_places_bitlinear_only_in_quantized_sites():
    import torch.nn as nn

    from open_mythos import OpenMythos

    model = OpenMythos(_tiny_cfg("w158"))
    quantized = {
        name
        for name, module in model.named_modules()
        if isinstance(module, BitLinear)
    }

    assert quantized, "expected BitLinear modules under quant_mode=w158"
    assert all(
        name.startswith(("prelude", "coda", "recurrent.block"))
        for name in quantized
    )
    # Head, router, and LoRA stay full precision by design.
    assert "head" not in quantized
    assert not any(".router." in name for name in quantized)
    assert not any(".lora." in name for name in quantized)
    assert type(model.head) is nn.Linear


def test_none_model_contains_no_bitlinear():
    from open_mythos import OpenMythos

    model = OpenMythos(_tiny_cfg("none"))

    assert not any(
        isinstance(module, BitLinear) for module in model.modules()
    )


def test_w158_forward_is_finite_and_differs_from_fp_twin():
    from open_mythos import OpenMythos

    torch.manual_seed(0)
    fp_model = OpenMythos(_tiny_cfg("none"))
    torch.manual_seed(0)
    quant_model = OpenMythos(_tiny_cfg("w158"))
    x = torch.randint(0, 64, (2, 8))

    with torch.no_grad():
        fp_logits = fp_model(x)
        quant_logits = quant_model(x)

    assert quant_logits.shape == fp_logits.shape
    assert torch.isfinite(quant_logits).all()
    assert not torch.allclose(fp_logits, quant_logits)


def test_fp_checkpoint_state_dict_loads_into_w158_model():
    from open_mythos import OpenMythos

    fp_model = OpenMythos(_tiny_cfg("none"))
    quant_model = OpenMythos(_tiny_cfg("w158"))

    quant_model.load_state_dict(fp_model.state_dict())
