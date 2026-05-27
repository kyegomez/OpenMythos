"""Tests for stochastic-depth (Option B) training path: bypass_act flag."""

import torch

from open_mythos.main import MythosConfig, OpenMythos


def _small_cfg() -> MythosConfig:
    """Small CPU config used by the existing test suite."""
    return MythosConfig(
        vocab_size=128,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=2,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=64,
        act_threshold=0.99,
        lora_rank=4,
    )


def _build_block_inputs(cfg: MythosConfig, B: int = 2, T: int = 8):
    """Build the (h, e, freqs_cis) inputs needed by RecurrentBlock.forward."""
    torch.manual_seed(0)
    model = OpenMythos(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    x = model.embed(input_ids)
    freqs_cis = model.freqs_cis[:T]
    mask = model._causal_mask(T, x.device, x.dtype)
    for i, layer in enumerate(model.prelude):
        x = layer(x, freqs_cis, mask, None, cache_key=f"prelude_{i}")
    return model.recurrent, x.clone(), x.clone(), freqs_cis, mask


def test_recurrent_block_bypass_act_differs_from_act():
    """bypass_act=True should produce a different output than bypass_act=False."""
    cfg = _small_cfg()
    block, h, e, freqs_cis, mask = _build_block_inputs(cfg)
    torch.manual_seed(1)
    out_act = block(h.clone(), e, freqs_cis, mask, n_loops=4, bypass_act=False)
    torch.manual_seed(1)
    out_bypass = block(h.clone(), e, freqs_cis, mask, n_loops=4, bypass_act=True)
    assert out_act.shape == out_bypass.shape
    assert not torch.allclose(
        out_act, out_bypass, atol=1e-6
    ), "bypass_act=True should not equal ACT-weighted output"


def test_recurrent_block_bypass_act_runs_full_n_loops():
    """With bypass_act=True there should be no early exit; all n_loops iterations run."""
    cfg = _small_cfg()
    block, h, e, freqs_cis, mask = _build_block_inputs(cfg)
    call_count = {"n": 0}
    original_block = block.block.forward

    def counting_forward(*args, **kwargs):
        call_count["n"] += 1
        return original_block(*args, **kwargs)

    block.block.forward = counting_forward
    try:
        _ = block(h, e, freqs_cis, mask, n_loops=3, bypass_act=True)
    finally:
        block.block.forward = original_block
    assert call_count["n"] == 3, f"expected 3 block calls, got {call_count['n']}"


def test_recurrent_block_bypass_act_returns_final_h():
    """bypass_act=True output should match a manual iteration returning the final h."""
    from open_mythos.main import loop_index_embedding

    cfg = _small_cfg()
    block, h, e, freqs_cis, mask = _build_block_inputs(cfg)
    n_loops = 3

    torch.manual_seed(1)
    h_manual = h.clone()
    for t in range(n_loops):
        h_loop = loop_index_embedding(h_manual, t, block.loop_dim)
        combined = block.norm(h_loop + e)
        trans_out = block.block(combined, freqs_cis, mask, None, f"recurrent_loop_{t}")
        trans_out = trans_out + block.lora(trans_out, t)
        h_manual = block.injection(h_manual, e, trans_out)

    torch.manual_seed(1)
    out_bypass = block(h.clone(), e, freqs_cis, mask, n_loops=n_loops, bypass_act=True)

    assert torch.allclose(
        out_bypass, h_manual, atol=1e-5
    ), "bypass_act=True should return the final hidden state after n_loops iterations"


def test_openmythos_forward_bypass_act_propagates():
    """OpenMythos.forward(bypass_act=True) should route through RecurrentBlock with bypass_act=True."""
    cfg = _small_cfg()
    torch.manual_seed(0)
    model = OpenMythos(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))

    torch.manual_seed(1)
    logits_act = model(input_ids, n_loops=3, bypass_act=False)
    torch.manual_seed(1)
    logits_bypass = model(input_ids, n_loops=3, bypass_act=True)

    assert logits_act.shape == logits_bypass.shape
    assert not torch.allclose(
        logits_act, logits_bypass, atol=1e-6
    ), "bypass_act should change model output"


def test_state_dict_compatible_across_modes(tmp_path):
    """state_dict round-trips cleanly and the loaded model works in both ACT and bypass modes."""
    cfg = _small_cfg()
    torch.manual_seed(0)
    model_a = OpenMythos(cfg)
    ckpt_path = tmp_path / "model.pt"
    torch.save(model_a.state_dict(), ckpt_path)

    torch.manual_seed(1)
    model_b = OpenMythos(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    # strict=True raises if any keys are missing or unexpected, which is the
    # actual compatibility check.
    model_b.load_state_dict(state, strict=True)

    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    torch.manual_seed(2)
    logits_act = model_b(input_ids, n_loops=3, bypass_act=False)
    torch.manual_seed(2)
    logits_bypass = model_b(input_ids, n_loops=3, bypass_act=True)
    assert logits_act.shape == logits_bypass.shape
    assert torch.isfinite(logits_act).all(), "ACT logits must be finite"
    assert torch.isfinite(logits_bypass).all(), "bypass logits must be finite"


def test_training_step_runs_in_each_mode():
    """One forward+backward+optimizer step works in both modes without error."""
    cfg = _small_cfg()
    torch.manual_seed(0)
    model = OpenMythos(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))

    # ACT mode
    optimizer.zero_grad()
    logits = model(input_ids, n_loops=None, bypass_act=False)
    loss_act = torch.nn.functional.cross_entropy(
        logits.view(-1, cfg.vocab_size), targets.view(-1)
    )
    loss_act.backward()
    optimizer.step()
    assert torch.isfinite(loss_act), "ACT-mode loss must be finite"

    # Stochastic-depth mode
    optimizer.zero_grad()
    logits = model(input_ids, n_loops=3, bypass_act=True)
    loss_sd = torch.nn.functional.cross_entropy(
        logits.view(-1, cfg.vocab_size), targets.view(-1)
    )
    loss_sd.backward()
    optimizer.step()
    assert torch.isfinite(loss_sd), "stochastic-depth-mode loss must be finite"
