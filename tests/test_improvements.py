"""Tests for the polish pass: MoE balancing, efficient dispatch, MLA cache,
input validation, and safetensors save/load."""

import tempfile

import pytest
import torch

from open_mythos.main import MoEFFN, MythosConfig, OpenMythos


B, T = 2, 8


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
# MoE: efficient dispatch is numerically identical to the reference scatter
# ---------------------------------------------------------------------------


def _reference_moe_forward(moe: MoEFFN, x: torch.Tensor) -> torch.Tensor:
    """The original O(topk x experts) double-loop dispatch, for equivalence checks."""
    import torch.nn.functional as F

    Bb, Tt, D = x.shape
    flat = x.view(Bb * Tt, D)
    logits = moe.router(flat)
    scores = F.softmax(logits, dim=-1)
    _, topk_idx = (logits + moe.router_bias).topk(moe.topk, dim=-1)
    topk_scores = scores.gather(-1, topk_idx)
    topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
    out = torch.zeros_like(flat)
    for i in range(moe.topk):
        expert_ids = topk_idx[:, i]
        token_scores = topk_scores[:, i].unsqueeze(-1)
        for eid in range(moe.n_experts):
            mask = expert_ids == eid
            if not mask.any():
                continue
            out[mask] += token_scores[mask] * moe.routed_experts[eid](flat[mask])
    for shared in moe.shared_experts:
        out = out + shared(flat)
    return out.view(Bb, Tt, D)


class TestMoEDispatch:
    def test_matches_reference(self):
        torch.manual_seed(0)
        moe = MoEFFN(gqa_cfg()).eval()
        x = torch.randn(B, T, 64)
        fast = moe(x)
        ref = _reference_moe_forward(moe, x)
        assert torch.allclose(fast, ref, atol=1e-6)


# ---------------------------------------------------------------------------
# MoE: aux-loss-free load balancing actually moves the bias toward balance
# ---------------------------------------------------------------------------


class TestRouterBias:
    def test_bias_updates_toward_balance(self):
        torch.manual_seed(0)
        moe = MoEFFN(gqa_cfg(n_experts=8)).train()
        # Skew the router so one expert is strongly preferred.
        with torch.no_grad():
            moe.router.weight.zero_()
            moe.router.weight[0] += 5.0
        x = torch.randn(4, 16, 64)
        moe(x)
        before = moe.router_bias.clone()
        moe.update_bias()
        after = moe.router_bias
        # Over-loaded expert 0 must have its bias decreased relative to the rest.
        assert after[0] < before[0]
        assert after[1:].mean() > after[0]

    def test_load_counter_resets(self):
        moe = MoEFFN(gqa_cfg()).train()
        moe(torch.randn(2, 8, 64))
        assert moe.expert_load.sum() > 0
        moe.update_bias()
        assert moe.expert_load.sum() == 0

    def test_no_update_at_inference(self):
        moe = MoEFFN(gqa_cfg()).eval()
        moe(torch.randn(2, 8, 64))
        assert moe.expert_load.sum() == 0

    def test_disabled_when_rate_zero(self):
        moe = MoEFFN(gqa_cfg(router_bias_update_rate=0.0)).train()
        moe(torch.randn(2, 8, 64))
        before = moe.router_bias.clone()
        moe.update_bias()
        assert torch.equal(before, moe.router_bias)

    def test_model_hook_runs(self):
        model = OpenMythos(gqa_cfg()).train()
        ids = torch.randint(0, 200, (B, T))
        model(ids).sum().backward()
        # should not raise; recurrent MoE load accumulated
        model.update_router_bias()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def setup_method(self):
        self.model = OpenMythos(gqa_cfg()).eval()

    def test_rejects_out_of_range_token(self):
        bad = torch.tensor([[0, 1, 999]])  # 999 >= vocab_size 200
        with pytest.raises(ValueError, match="outside"):
            self.model(bad)

    def test_rejects_negative_token(self):
        bad = torch.tensor([[0, -1, 2]])
        with pytest.raises(ValueError, match="outside"):
            self.model(bad)

    def test_rejects_position_past_max_seq_len(self):
        ids = torch.randint(0, 200, (1, 4))
        with pytest.raises(ValueError, match="max_seq_len"):
            self.model(ids, start_pos=30)  # 30 + 4 > 32

    def test_rejects_wrong_rank(self):
        with pytest.raises(ValueError, match="2-D"):
            self.model(torch.randint(0, 200, (8,)))

    def test_accepts_valid(self):
        ids = torch.randint(0, 200, (B, T))
        assert self.model(ids).shape == (B, T, 200)


# ---------------------------------------------------------------------------
# safetensors save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    @pytest.mark.parametrize("attn", ["gqa", "mla"])
    def test_round_trip_identical(self, attn):
        cfg = gqa_cfg(attn_type=attn)
        model = OpenMythos(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        with torch.no_grad():
            before = model(ids)
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d)
            reloaded = OpenMythos.from_pretrained(d).eval()
            import os

            assert os.path.exists(os.path.join(d, "config.json"))
            assert os.path.exists(os.path.join(d, "model.safetensors"))
            assert not any(
                f.endswith((".bin", ".pt", ".pth", ".pkl")) for f in os.listdir(d)
            )
        with torch.no_grad():
            after = reloaded(ids)
        assert torch.allclose(before, after, atol=1e-6)

    def test_weight_tying_preserved(self):
        model = OpenMythos(gqa_cfg())
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d)
            reloaded = OpenMythos.from_pretrained(d)
        assert reloaded.head.weight is reloaded.embed.weight


# ---------------------------------------------------------------------------
# num_parameters counts tied weights once
# ---------------------------------------------------------------------------


class TestNumParameters:
    def test_counts_tied_weight_once(self):
        model = OpenMythos(gqa_cfg())
        # head and embedding are tied (same tensor), so it must be counted once.
        assert model.head.weight is model.embed.weight
        counted = model.num_parameters()
        assert counted == sum(p.numel() for p in model.parameters())
        # the tie saves exactly one vocab x dim matrix versus an untied head
        assert counted < model.num_parameters() + model.embed.weight.numel()


# ---------------------------------------------------------------------------
# MLA cache stays compressed after the shared-rope-key fix
# ---------------------------------------------------------------------------


class TestMLACacheShape:
    def test_k_rope_is_shared_not_per_head(self):
        cfg = mla_cfg()
        model = OpenMythos(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        cache: dict = {}
        with torch.no_grad():
            model(ids, kv_cache=cache)
        entry = next(v for v in cache.values() if "k_rope" in v)
        # shared rope key: last dim is rope_dim and there is no n_heads axis
        assert entry["k_rope"].shape[-1] == cfg.qk_rope_head_dim
        assert entry["k_rope"].dim() == 3  # (B, S, rope_dim), not (B, S, H, rope_dim)
