"""
Tests for the MoE router: aux-loss-free bias update + vectorized dispatch.

Two invariants we guard here:

1. **Router bias actually moves.** ``MoEFFN.router_bias`` is a non-persistent
   buffer updated by ``update_bias`` after each optimizer step. The
   upstream version never wired this up — router_bias stayed at zeros
   forever and balancing was silently inert. These tests would catch
   that regression.

2. **Vectorized dispatch matches the reference loop.** The per-expert
   argsort + index_add_ path replaces an O(topk * n_experts) Python
   loop. The two paths must produce byte-identical (modulo tiny fp
   error) outputs for the same inputs, otherwise the optimization is a
   silent semantics change.
"""

import torch

from open_mythos.main import MoEFFN, MythosConfig, OpenMythos


def _moe_cfg(**overrides) -> MythosConfig:
    cfg = dict(
        vocab_size=128,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        max_loop_iters=1,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=8,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=32,
        bias_update_speed=1e-2,  # larger than default so 1 step is visible
    )
    cfg.update(overrides)
    return MythosConfig(**cfg)


# ---------------------------------------------------------------------------
# Router-bias update
# ---------------------------------------------------------------------------


def test_router_bias_is_a_buffer() -> None:
    """Must be a buffer (non-persistent) — not a trainable Parameter."""
    cfg = _moe_cfg()
    moe = MoEFFN(cfg)
    assert "router_bias" in dict(moe.named_buffers())
    # expert_load is also a non-persistent buffer, not a param.
    assert "expert_load" in dict(moe.named_buffers())
    # And neither should be in the Parameter list (they must not be trained).
    params = {name for name, _ in moe.named_parameters()}
    assert "router_bias" not in params
    assert "expert_load" not in params


def test_expert_load_accumulates_on_forward() -> None:
    cfg = _moe_cfg()
    moe = MoEFFN(cfg).train()
    x = torch.randn(2, 4, cfg.dim)

    assert float(moe.expert_load.sum()) == 0.0
    _ = moe(x)
    # N*K tokens were routed (batch*seq*topk).
    expected_total = 2 * 4 * cfg.n_experts_per_tok
    assert int(moe.expert_load.sum().item()) == expected_total


def test_expert_load_not_accumulated_in_eval() -> None:
    cfg = _moe_cfg()
    moe = MoEFFN(cfg).eval()
    x = torch.randn(2, 4, cfg.dim)
    _ = moe(x)
    assert float(moe.expert_load.sum()) == 0.0


def test_update_bias_shifts_toward_underused_experts() -> None:
    """
    Construct a synthetic load vector where one expert is underused and
    one is overused, then verify update_bias pushes the bias for the
    underused expert up and the overused one down.
    """
    cfg = _moe_cfg(n_experts=4)
    moe = MoEFFN(cfg)

    # Load: expert 0 underused (0 tokens), expert 3 overused (100 tokens),
    # rest neutral.
    moe.expert_load.copy_(torch.tensor([0.0, 20.0, 20.0, 100.0]))
    speed = 1e-2
    before = moe.router_bias.clone()
    moe.update_bias(speed)

    # Mean = (0+20+20+100)/4 = 35.  Direction = sign(mean - load).
    # expert 0: sign(35 - 0) = +1   → bias[0] increases by speed
    # expert 3: sign(35 - 100) = -1 → bias[3] decreases by speed
    assert (moe.router_bias[0] - before[0]).item() > 0
    assert (moe.router_bias[3] - before[3]).item() < 0


def test_update_bias_zeroes_load_after_step() -> None:
    cfg = _moe_cfg()
    moe = MoEFFN(cfg)
    moe.expert_load.fill_(5.0)
    moe.update_bias(1e-2)
    assert float(moe.expert_load.sum()) == 0.0


def test_update_bias_noop_when_speed_zero() -> None:
    cfg = _moe_cfg()
    moe = MoEFFN(cfg)
    moe.expert_load.fill_(5.0)
    before = moe.router_bias.clone()
    moe.update_bias(0.0)
    # Bias unchanged ...
    assert torch.equal(moe.router_bias, before)
    # ... but load still flushed so memory doesn't grow unbounded.
    assert float(moe.expert_load.sum()) == 0.0


def test_update_bias_magnitude_bounded_by_speed() -> None:
    """
    Sign-based update means per-step delta is exactly `speed` per expert
    regardless of how large the imbalance is — the whole point of the
    design, as opposed to a direct delta that explodes on spiky loads.
    """
    cfg = _moe_cfg(n_experts=4)
    moe = MoEFFN(cfg)

    moe.expert_load.copy_(torch.tensor([0.0, 0.0, 0.0, 1_000_000.0]))
    speed = 1e-2
    before = moe.router_bias.clone()
    moe.update_bias(speed)

    diff = (moe.router_bias - before).abs().max().item()
    assert abs(diff - speed) < 1e-6


# ---------------------------------------------------------------------------
# End-to-end: OpenMythos.update_router_biases drives every MoE layer
# ---------------------------------------------------------------------------


def test_openmythos_update_router_biases_walks_all_moe_layers() -> None:
    cfg = _moe_cfg()
    model = OpenMythos(cfg)

    # Fabricate imbalance into every MoE layer.
    moes = [m for m in model.modules() if isinstance(m, MoEFFN)]
    assert len(moes) > 0

    # Make expert 0 look underused across the whole model.
    for m in moes:
        m.expert_load.copy_(torch.tensor([0.0] + [10.0] * (cfg.n_experts - 1)))

    before = [m.router_bias.clone() for m in moes]
    model.update_router_biases(ddp=False)

    for b_before, m in zip(before, moes):
        # Expert 0's bias must have moved up.
        assert (m.router_bias[0] - b_before[0]).item() > 0
        # Load flushed.
        assert float(m.expert_load.sum()) == 0.0


# ---------------------------------------------------------------------------
# Vectorized dispatch semantic equivalence to the naive loop.
# ---------------------------------------------------------------------------


def _naive_moe_forward(moe: MoEFFN, x: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation of the routed-expert dispatch using the
    obvious (and slow) double loop: for each (token, k) pair, find the
    selected expert and add its weighted contribution to the output.

    Deliberately verbose so it's easy to read as a spec. Only used here
    to cross-check the fast implementation; never called by the model.
    """
    import torch.nn.functional as F

    B, T, D = x.shape
    flat = x.view(B * T, D)

    logits = moe.router(flat)
    scores = F.softmax(logits, dim=-1, dtype=torch.float32).to(flat.dtype)
    _, topk_idx = (logits.float() + moe.router_bias).topk(moe.topk, dim=-1)
    topk_scores = scores.gather(-1, topk_idx)
    denom = topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    topk_scores = topk_scores / denom

    out = torch.zeros_like(flat)
    N = flat.shape[0]
    for n in range(N):
        for k in range(moe.topk):
            eid = int(topk_idx[n, k])
            w = topk_scores[n, k]
            out[n] = out[n] + w * moe.routed_experts[eid](flat[n : n + 1]).squeeze(0)

    for shared in moe.shared_experts:
        out = out + shared(flat)

    return out.view(B, T, D)


def test_vectorized_matches_naive_dispatch() -> None:
    """
    The fast path uses argsort + index_add_; the reference path iterates
    per token per k. Same inputs, same (eval-mode) state → same outputs
    within fp tolerance.
    """
    cfg = _moe_cfg(n_experts=6, n_experts_per_tok=3, dim=32, expert_dim=16)
    moe = MoEFFN(cfg).eval()  # eval so expert_load isn't mutated

    torch.manual_seed(0)
    x = torch.randn(2, 5, cfg.dim)

    fast = moe(x)
    slow = _naive_moe_forward(moe, x)

    torch.testing.assert_close(fast, slow, rtol=1e-5, atol=1e-5)
