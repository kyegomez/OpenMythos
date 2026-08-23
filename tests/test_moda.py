import torch
import pytest
from open_mythos.moda import (
    DeepSeekExpert,
    DeepSeekGate,
    DeepSeekMoE,
    MoDAAttention,
    MoDABlock,
    MoDAConfig,
    MoDAModel,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_emb,
)

# ---------------------------------------------------------------------------
# Shared small configs (kept tiny so tests run fast on CPU)
# ---------------------------------------------------------------------------

B, T = 2, 8  # batch, sequence length


def moda_cfg(**overrides) -> MoDAConfig:
    defaults = dict(
        vocab_size=200,
        d_model=32,
        n_layers=2,
        n_heads_q=4,
        n_heads_kv=2,
        head_dim=8,
        max_seq_len=32,
        rope_base=10000.0,
        attn_dropout=0.0,
        norm_eps=1e-6,
        n_shared_experts=1,
        n_routed_experts=4,
        n_activated_experts=2,
        expert_hidden_dim=16,
        moe_balance_alpha=0.01,
        moe_score_func="softmax",
        moe_n_groups=1,
        moe_topk_groups=1,
        moe_route_scale=1.0,
    )
    defaults.update(overrides)
    return MoDAConfig(**defaults)


# ---------------------------------------------------------------------------
# MoDAConfig
# ---------------------------------------------------------------------------


class TestMoDAConfig:
    def test_defaults_construct(self):
        cfg = MoDAConfig()
        assert cfg.d_model == cfg.n_heads_q * cfg.head_dim

    def test_overrides_applied(self):
        cfg = moda_cfg(n_layers=5)
        assert cfg.n_layers == 5


# ---------------------------------------------------------------------------
# RMSNorm (moda's own copy)
# ---------------------------------------------------------------------------


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(dim=32)
        x = torch.randn(B, T, 32)
        assert norm(x).shape == (B, T, 32)

    def test_unit_rms(self):
        norm = RMSNorm(dim=32)
        x = torch.randn(B, T, 32) * 5.0
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


# ---------------------------------------------------------------------------
# RotaryEmbedding / apply_rotary_emb
# ---------------------------------------------------------------------------


class TestRotaryEmbedding:
    def test_forward_shape(self):
        rope = RotaryEmbedding(dim=8, max_seq_len=16)
        cos, sin = rope(6)
        assert cos.shape == (1, 1, 6, 8)
        assert sin.shape == (1, 1, 6, 8)

    def test_cache_grows_when_exceeded(self):
        rope = RotaryEmbedding(dim=8, max_seq_len=4)
        assert rope._cos.shape[2] == 4
        cos, sin = rope(10)
        assert rope._cos.shape[2] >= 10
        assert cos.shape == (1, 1, 10, 8)

    def test_cache_reused_when_sufficient(self):
        rope = RotaryEmbedding(dim=8, max_seq_len=16)
        cache_before = rope._cos
        rope(6)
        assert rope._cos is cache_before


class TestApplyRotaryEmb:
    def test_output_shape(self):
        rope = RotaryEmbedding(dim=8, max_seq_len=16)
        cos, sin = rope(T)
        x = torch.randn(1, 2, T, 8)
        assert apply_rotary_emb(x, cos, sin).shape == x.shape

    def test_preserves_norm(self):
        """RoPE is a rotation — per-position L2 norm must be unchanged."""
        rope = RotaryEmbedding(dim=8, max_seq_len=16)
        cos, sin = rope(T)
        x = torch.randn(1, 2, T, 8)
        out = apply_rotary_emb(x, cos, sin)
        assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-4)

    def test_position_zero_is_identity(self):
        rope = RotaryEmbedding(dim=8, max_seq_len=16)
        cos, sin = rope(1)
        x = torch.randn(1, 2, 1, 8)
        assert torch.allclose(x, apply_rotary_emb(x, cos, sin), atol=1e-5)


# ---------------------------------------------------------------------------
# DeepSeekExpert
# ---------------------------------------------------------------------------


class TestDeepSeekExpert:
    def test_output_shape(self):
        expert = DeepSeekExpert(d_model=32, hidden_dim=16)
        x = torch.randn(5, 32)
        assert expert(x).shape == (5, 32)


# ---------------------------------------------------------------------------
# DeepSeekGate
# ---------------------------------------------------------------------------


class TestDeepSeekGate:
    def test_output_shapes(self):
        gate = DeepSeekGate(d_model=32, n_routed_experts=4, n_activated=2)
        x = torch.randn(5, 32)
        weights, indices, scores = gate(x)
        assert weights.shape == (5, 2)
        assert indices.shape == (5, 2)
        assert scores.shape == (5, 4)

    def test_indices_in_range(self):
        gate = DeepSeekGate(d_model=32, n_routed_experts=4, n_activated=2)
        x = torch.randn(5, 32)
        _, indices, _ = gate(x)
        assert indices.min().item() >= 0
        assert indices.max().item() < 4

    def test_top_k_indices_unique_per_token(self):
        gate = DeepSeekGate(d_model=32, n_routed_experts=4, n_activated=2)
        x = torch.randn(5, 32)
        _, indices, _ = gate(x)
        for row in indices:
            assert len(set(row.tolist())) == indices.shape[1]

    def test_sigmoid_weights_renormalized_to_one(self):
        """Sigmoid scoring divides by the sum of selected weights, so each
        token's selected gate weights must sum to exactly route_scale."""
        gate = DeepSeekGate(
            d_model=32, n_routed_experts=4, n_activated=2, score_func="sigmoid"
        )
        x = torch.randn(5, 32)
        weights, _, _ = gate(x)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(5), atol=1e-5)

    def test_softmax_scores_sum_to_one(self):
        gate = DeepSeekGate(
            d_model=32, n_routed_experts=4, n_activated=2, score_func="softmax"
        )
        x = torch.randn(5, 32)
        _, _, scores = gate(x)
        assert torch.allclose(scores.sum(dim=-1), torch.ones(5), atol=1e-5)

    def test_group_limited_routing_shapes(self):
        gate = DeepSeekGate(
            d_model=32,
            n_routed_experts=4,
            n_activated=2,
            n_groups=2,
            topk_groups=1,
        )
        x = torch.randn(3, 32)
        weights, indices, scores = gate(x)
        assert weights.shape == (3, 2)
        assert indices.shape == (3, 2)


# ---------------------------------------------------------------------------
# DeepSeekMoE
# ---------------------------------------------------------------------------


class TestDeepSeekMoE:
    def setup_method(self):
        self.cfg = moda_cfg()
        self.moe = DeepSeekMoE(self.cfg)

    def test_output_shape(self):
        x = torch.randn(B, T, self.cfg.d_model)
        out, _ = self.moe(x)
        assert out.shape == (B, T, self.cfg.d_model)

    def test_balance_loss_none_in_eval(self):
        self.moe.eval()
        x = torch.randn(B, T, self.cfg.d_model)
        _, balance_loss = self.moe(x)
        assert balance_loss is None

    def test_balance_loss_present_in_train(self):
        self.moe.train()
        x = torch.randn(B, T, self.cfg.d_model)
        _, balance_loss = self.moe(x)
        assert balance_loss is not None
        assert balance_loss.dim() == 0

    def test_balance_loss_disabled_when_alpha_zero(self):
        cfg = moda_cfg(moe_balance_alpha=0.0)
        moe = DeepSeekMoE(cfg)
        moe.train()
        x = torch.randn(B, T, cfg.d_model)
        _, balance_loss = moe(x)
        assert balance_loss is None

    def test_shared_experts_always_fire(self):
        # Zero out every routed expert; output should still be nonzero from shared.
        for expert in self.moe.experts:
            for p in expert.parameters():
                p.data.zero_()
        x = torch.randn(B, T, self.cfg.d_model)
        out, _ = self.moe(x)
        assert out.abs().sum() > 0

    def test_gate_receives_gradient_through_balance_loss(self):
        self.moe.train()
        x = torch.randn(B, T, self.cfg.d_model)
        out, balance_loss = self.moe(x)
        (out.sum() + balance_loss).backward()
        assert self.moe.gate.weight.grad is not None


# ---------------------------------------------------------------------------
# MoDAAttention
# ---------------------------------------------------------------------------


class TestMoDAAttention:
    def setup_method(self):
        self.cfg = moda_cfg()
        self.attn = MoDAAttention(self.cfg)
        self.rope = RotaryEmbedding(
            self.cfg.head_dim, self.cfg.max_seq_len, self.cfg.rope_base
        )

    def test_invalid_gqa_ratio_raises(self):
        with pytest.raises(ValueError):
            MoDAAttention(moda_cfg(n_heads_q=4, n_heads_kv=3))

    def test_output_shape_no_depth_cache(self):
        x = torch.randn(B, T, self.cfg.d_model)
        cos, sin = self.rope(T)
        out = self.attn(x, [], [], cos, sin)
        assert out.shape == (B, T, self.cfg.d_model)

    def test_output_shape_with_depth_cache(self):
        x = torch.randn(B, T, self.cfg.d_model)
        cos, sin = self.rope(T)
        Hk, d = self.cfg.n_heads_kv, self.cfg.head_dim
        depth_k = [torch.randn(B, Hk, T, d)]
        depth_v = [torch.randn(B, Hk, T, d)]
        out = self.attn(x, depth_k, depth_v, cos, sin)
        assert out.shape == (B, T, self.cfg.d_model)

    def test_depth_cache_changes_output(self):
        """The depth-KV path must actually be wired into the unified softmax,
        not silently ignored."""
        x = torch.randn(B, T, self.cfg.d_model)
        cos, sin = self.rope(T)
        Hk, d = self.cfg.n_heads_kv, self.cfg.head_dim
        depth_k = [torch.randn(B, Hk, T, d)]
        depth_v = [torch.randn(B, Hk, T, d)]

        out_no_depth = self.attn(x, [], [], cos, sin)
        out_with_depth = self.attn(x, depth_k, depth_v, cos, sin)
        assert not torch.allclose(out_no_depth, out_with_depth, atol=1e-5)

    def test_causal_no_depth_cache(self):
        x = torch.randn(B, T, self.cfg.d_model)
        cos, sin = self.rope(T)
        out = self.attn(x, [], [], cos, sin)

        x_perturbed = x.clone()
        x_perturbed[:, -1, :] += 10.0
        out_perturbed = self.attn(x_perturbed, [], [], cos, sin)

        assert torch.allclose(out[:, :-1], out_perturbed[:, :-1], atol=1e-5)

    def test_causal_with_depth_cache(self):
        x = torch.randn(B, T, self.cfg.d_model)
        cos, sin = self.rope(T)
        Hk, d = self.cfg.n_heads_kv, self.cfg.head_dim
        depth_k = [torch.randn(B, Hk, T, d)]
        depth_v = [torch.randn(B, Hk, T, d)]
        out = self.attn(x, depth_k, depth_v, cos, sin)

        x_perturbed = x.clone()
        x_perturbed[:, -1, :] += 10.0
        out_perturbed = self.attn(x_perturbed, depth_k, depth_v, cos, sin)

        assert torch.allclose(out[:, :-1], out_perturbed[:, :-1], atol=1e-5)

    def test_multiple_depth_layers(self):
        x = torch.randn(B, T, self.cfg.d_model)
        cos, sin = self.rope(T)
        Hk, d = self.cfg.n_heads_kv, self.cfg.head_dim
        depth_k = [torch.randn(B, Hk, T, d) for _ in range(3)]
        depth_v = [torch.randn(B, Hk, T, d) for _ in range(3)]
        out = self.attn(x, depth_k, depth_v, cos, sin)
        assert out.shape == (B, T, self.cfg.d_model)


# ---------------------------------------------------------------------------
# MoDABlock
# ---------------------------------------------------------------------------


class TestMoDABlock:
    def setup_method(self):
        self.cfg = moda_cfg()
        self.block = MoDABlock(self.cfg)
        self.rope = RotaryEmbedding(
            self.cfg.head_dim, self.cfg.max_seq_len, self.cfg.rope_base
        )

    def test_output_shapes(self):
        x = torch.randn(B, T, self.cfg.d_model)
        cos, sin = self.rope(T)
        x_out, k_write, v_write, _ = self.block(x, [], [], cos, sin)
        Hk, d = self.cfg.n_heads_kv, self.cfg.head_dim
        assert x_out.shape == (B, T, self.cfg.d_model)
        assert k_write.shape == (B, Hk, T, d)
        assert v_write.shape == (B, Hk, T, d)

    def test_balance_loss_none_in_eval(self):
        self.block.eval()
        x = torch.randn(B, T, self.cfg.d_model)
        cos, sin = self.rope(T)
        *_, balance_loss = self.block(x, [], [], cos, sin)
        assert balance_loss is None

    def test_two_block_depth_chain(self):
        """Simulates MoDAModel's loop: block 2 reads depth KVs written by block 1."""
        block1 = MoDABlock(self.cfg)
        block2 = MoDABlock(self.cfg)
        x = torch.randn(B, T, self.cfg.d_model)
        cos, sin = self.rope(T)

        x1, k1, v1, _ = block1(x, [], [], cos, sin)
        x2, k2, v2, _ = block2(x1, [k1], [v1], cos, sin)

        assert x2.shape == (B, T, self.cfg.d_model)
        Hk, d = self.cfg.n_heads_kv, self.cfg.head_dim
        assert k2.shape == (B, Hk, T, d)


# ---------------------------------------------------------------------------
# MoDAModel
# ---------------------------------------------------------------------------


class TestMoDAModel:
    def setup_method(self):
        self.cfg = moda_cfg()
        self.model = MoDAModel(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (B, T))

    def test_forward_shape_no_labels(self):
        logits, loss = self.model(self.ids)
        assert logits.shape == (B, T, self.cfg.vocab_size)
        assert loss is None

    def test_forward_no_nan(self):
        logits, _ = self.model(self.ids)
        assert not torch.isnan(logits).any()

    def test_forward_with_labels_returns_scalar_loss(self):
        labels = torch.randint(0, self.cfg.vocab_size, (B, T))
        logits, loss = self.model(self.ids, labels=labels)
        assert loss is not None
        assert loss.dim() == 0

    def test_eval_mode_loss_has_no_balance_term(self):
        """In eval mode DeepSeekMoE returns balance_loss=None for every layer,
        so the total loss must reduce to plain LM cross-entropy without error."""
        self.model.eval()
        labels = torch.randint(0, self.cfg.vocab_size, (B, T))
        logits, loss = self.model(self.ids, labels=labels)
        expected = torch.nn.functional.cross_entropy(
            logits.view(-1, self.cfg.vocab_size), labels.view(-1)
        )
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_exceeds_max_seq_len_raises(self):
        cfg = moda_cfg(max_seq_len=4)
        model = MoDAModel(cfg)
        with pytest.raises(ValueError):
            model(torch.randint(0, cfg.vocab_size, (1, 8)))

    def test_weight_tying(self):
        assert self.model.lm_head.weight is self.model.embed.weight

    def test_num_parameters_positive(self):
        assert self.model.num_parameters() > 0

    def test_num_parameters_trainable_only_matches_total_when_all_trainable(self):
        assert self.model.num_parameters(trainable_only=True) == self.model.num_parameters()

    def test_gradient_flow_through_full_model(self):
        """Mirrors examples/moda_example.py's smoke check: every parameter
        should receive a gradient except the last block's depth-write
        projections, which have no downstream consumer (no later layer
        reads them)."""
        self.model.train()
        labels = torch.randint(0, self.cfg.vocab_size, (B, T))
        _, loss = self.model(self.ids, labels=labels)
        loss.backward()

        last = self.cfg.n_layers - 1
        expected_missing = {
            f"blocks.{last}.k_write.weight",
            f"blocks.{last}.v_write.weight",
        }
        missing = {
            name for name, p in self.model.named_parameters() if p.grad is None
        }
        assert missing == expected_missing

    def test_gate_weight_receives_gradient(self):
        self.model.train()
        labels = torch.randint(0, self.cfg.vocab_size, (B, T))
        _, loss = self.model(self.ids, labels=labels)
        loss.backward()
        assert self.model.blocks[0].moe.gate.weight.grad is not None
