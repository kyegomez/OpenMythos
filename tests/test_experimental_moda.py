"""Tests for the Mixture-of-Depths Attention (MoDA) model.

Covers configuration, forward pass, loss computation, gradient flow, and
determinism for the independent ``open_mythos.experimental.moda`` model.  These tests use
tiny dimensions so they run quickly on CPU.
"""

from __future__ import annotations

import pytest
import torch

from open_mythos.experimental.moda import (
    DeepSeekGate,
    MoDAConfig,
    MoDAModel,
)


# ---------------------------------------------------------------------------
# Shared tiny config so tests stay fast on CPU
# ---------------------------------------------------------------------------

B, T = 2, 4  # batch, sequence length


def _tiny_cfg(**overrides) -> MoDAConfig:
    """Build a tiny MoDAConfig suitable for CPU unit tests."""
    defaults = dict(
        # Transformer / MoDA
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads_q=4,
        n_heads_kv=2,
        head_dim=16,
        max_seq_len=32,
        rope_base=10_000.0,
        attn_dropout=0.0,
        norm_eps=1e-6,
        # MoE
        n_shared_experts=2,
        n_routed_experts=8,
        n_activated_experts=2,
        expert_hidden_dim=32,
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
    def test_constructible(self):
        """Default MoDAConfig() should succeed and expose expected fields."""
        torch.manual_seed(42)
        cfg = MoDAConfig()
        assert cfg.vocab_size == 32_000
        assert cfg.d_model == 2048
        assert cfg.n_layers == 24
        assert cfg.n_heads_q == 16
        assert cfg.n_heads_kv == 8
        assert cfg.head_dim == 128
        # MoE defaults
        assert cfg.n_shared_experts == 2
        assert cfg.n_routed_experts == 64
        assert cfg.n_activated_experts == 6
        assert cfg.expert_hidden_dim == 704
        assert cfg.moe_score_func == "softmax"

    def test_tiny_overrides(self):
        """Config overrides via constructor kwargs should be honoured."""
        torch.manual_seed(42)
        cfg = _tiny_cfg()
        assert cfg.d_model == 64
        assert cfg.n_layers == 2
        assert cfg.vocab_size == 100


# ---------------------------------------------------------------------------
# MoDAModel
# ---------------------------------------------------------------------------


class TestMoDAModel:
    def test_forward_shape(self):
        """Forward without labels returns (logits, None) with logits shape [B, T, V]."""
        torch.manual_seed(42)
        cfg = _tiny_cfg()
        model = MoDAModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (B, T))

        out = model(input_ids)
        # Model returns a tuple (logits, loss); loss is None when labels is None.
        assert isinstance(out, tuple)
        assert len(out) == 2
        logits, loss = out
        assert logits.shape == (B, T, cfg.vocab_size)
        assert loss is None

    def test_labels_returns_loss(self):
        """Providing labels yields a finite, positive scalar loss."""
        torch.manual_seed(42)
        cfg = _tiny_cfg()
        model = MoDAModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (B, T))
        labels = input_ids.clone()

        logits, loss = model(input_ids, labels=labels)

        assert logits.shape == (B, T, cfg.vocab_size)
        assert loss is not None
        assert loss.dim() == 0  # scalar
        assert torch.isfinite(loss)
        assert loss.item() > 0.0

    def test_balance_loss_positive(self):
        """With MoE balance enabled in training mode the total loss is finite & >0.

        We also verify the model is in training mode so the per-block balance
        term is actually accumulated inside the forward pass.
        """
        torch.manual_seed(42)
        cfg = _tiny_cfg(moe_balance_alpha=0.1)  # amplify balance contribution
        model = MoDAModel(cfg)
        model.train()

        input_ids = torch.randint(0, cfg.vocab_size, (B, T))
        labels = input_ids.clone()

        _, loss = model(input_ids, labels=labels)

        assert loss is not None
        assert torch.isfinite(loss)
        assert loss.item() > 0.0

        # Sanity: balance loss component is reachable via the MoE gate.
        # The gate weight must be a Parameter of finite values.
        gate0 = model.blocks[0].moe.gate
        assert isinstance(gate0, DeepSeekGate)
        assert torch.isfinite(gate0.weight).all()

    def test_gradient_flow(self):
        """After backward(), every parameter should have a gradient.

        The smoke test in ``moda.py`` notes that the *last* block's depth-write
        projections legitimately receive no gradient because their outputs are
        never read (no layer depends on the last depth cache entry).  We accept
        that expected exception explicitly.

        Additionally, routed MoE experts only receive gradients when the
        router selects them for at least one token in the forward pass. With a
        tiny batch/seq it is expected that some experts are not picked, so we
        allow a configurable slack on routed-expert grad coverage while still
        requiring every non-routed parameter (embeddings, norms, attention
        projections, shared experts, gates) to receive a gradient.
        """
        torch.manual_seed(42)
        # Use a larger B×T so more tokens exercise more experts on average.
        # Even so, stochastic routing can leave a couple of experts idle per
        # forward; the assertion below tolerates that.
        cfg = _tiny_cfg()
        model = MoDAModel(cfg)
        model.train()

        B_big, T_big = 4, 8
        input_ids = torch.randint(0, cfg.vocab_size, (B_big, T_big))
        labels = input_ids.clone()

        _, loss = model(input_ids, labels=labels)
        assert loss is not None
        loss.backward()

        expected_no_grad = {
            f"blocks.{cfg.n_layers - 1}.k_write.weight",
            f"blocks.{cfg.n_layers - 1}.v_write.weight",
        }

        def _is_routed_expert_param(name: str) -> bool:
            # e.g. blocks.0.moe.experts.2.w1.weight
            return ".moe.experts." in name

        missing = [
            name
            for name, p in model.named_parameters()
            if p.grad is None and name not in expected_no_grad
        ]

        # Routed experts may legitimately miss grads when the router didn't
        # pick them on this batch. Separate these from true coverage gaps.
        missing_routed = [n for n in missing if _is_routed_expert_param(n)]
        missing_other = [n for n in missing if not _is_routed_expert_param(n)]

        # Non-routed params must always receive a gradient.
        assert missing_other == [], (
            "Non-routed parameters missing gradients "
            f"(count={len(missing_other)}): {missing_other}"
        )

        # Routed experts: allow up to ~40% of their params to miss grads due
        # to stochastic top-k routing on a tiny batch. With the CPU-friendly
        # shapes here, typically 0-2 of 8 experts per layer see zero tokens.
        routed_params = [
            n for n, _ in model.named_parameters() if _is_routed_expert_param(n)
        ]
        miss_ratio = len(missing_routed) / max(1, len(routed_params))
        assert miss_ratio <= 0.4, (
            "Too many routed-expert params missing gradients "
            f"(missing={len(missing_routed)}/{len(routed_params)}, "
            f"ratio={miss_ratio:.2f}): {missing_routed}"
        )

        # Spot check — MoE gate must receive a gradient via P_i in balance loss
        gate0_grad = model.blocks[0].moe.gate.weight.grad
        assert gate0_grad is not None
        assert torch.isfinite(gate0_grad).all()

    def test_deterministic_forward(self):
        """Same seed + same inputs ⇒ identical logits across two model builds."""
        cfg = _tiny_cfg()

        torch.manual_seed(42)
        model1 = MoDAModel(cfg)
        model1.eval()
        torch.manual_seed(7)
        input_ids = torch.randint(0, cfg.vocab_size, (B, T))
        with torch.no_grad():
            logits1, _ = model1(input_ids)

        torch.manual_seed(42)
        model2 = MoDAModel(cfg)
        model2.eval()
        torch.manual_seed(7)
        input_ids2 = torch.randint(0, cfg.vocab_size, (B, T))
        with torch.no_grad():
            logits2, _ = model2(input_ids2)

        assert torch.equal(input_ids, input_ids2)
        assert torch.equal(logits1, logits2)

    def test_sequence_too_long_raises(self):
        """Sequences longer than max_seq_len must raise ValueError."""
        torch.manual_seed(42)
        cfg = _tiny_cfg(max_seq_len=4)
        model = MoDAModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (B, cfg.max_seq_len + 1))
        with pytest.raises(ValueError):
            model(input_ids)

    def test_num_parameters_positive(self):
        """num_parameters() returns a positive integer."""
        torch.manual_seed(42)
        cfg = _tiny_cfg()
        model = MoDAModel(cfg)
        n = model.num_parameters()
        assert isinstance(n, int)
        assert n > 0
        assert model.num_parameters(trainable_only=True) == n


# ---------------------------------------------------------------------------
# DeepSeekGate
# ---------------------------------------------------------------------------


class TestDeepSeekGate:
    def test_output_shapes(self):
        """Gate returns weights, indices, and full soft scores with correct shapes."""
        torch.manual_seed(42)
        d_model, n_routed, k = 64, 8, 2
        gate = DeepSeekGate(
            d_model=d_model,
            n_routed_experts=n_routed,
            n_activated=k,
            score_func="softmax",
        )
        n_tokens = 12
        x = torch.randn(n_tokens, d_model)

        weights, indices, scores = gate(x)

        assert weights.shape == (n_tokens, k)
        assert indices.shape == (n_tokens, k)
        assert scores.shape == (n_tokens, n_routed)
        # Softmax rows must sum to ~1
        assert torch.allclose(
            scores.sum(dim=-1), torch.ones(n_tokens), atol=1e-4
        )
        # Indices must be within the expert range
        assert indices.min().item() >= 0
        assert indices.max().item() < n_routed

    def test_sigmoid_gate(self):
        """Sigmoid-variant gate returns weights that re-normalise to 1 per token."""
        torch.manual_seed(42)
        d_model, n_routed, k = 64, 8, 4
        gate = DeepSeekGate(
            d_model=d_model,
            n_routed_experts=n_routed,
            n_activated=k,
            score_func="sigmoid",
            route_scale=1.0,
        )
        n_tokens = 6
        x = torch.randn(n_tokens, d_model)

        weights, indices, scores = gate(x)
        assert weights.shape == (n_tokens, k)
        assert indices.shape == (n_tokens, k)
        assert scores.shape == (n_tokens, n_routed)
        # With route_scale=1.0 and sigmoid normalisation each row sums to 1.
        assert torch.allclose(
            weights.sum(dim=-1), torch.ones(n_tokens), atol=1e-4
        )

    def test_bias_initialises_zero(self):
        """When use_bias=True the bias parameter starts at zeros and is float32."""
        torch.manual_seed(42)
        gate = DeepSeekGate(
            d_model=32,
            n_routed_experts=4,
            n_activated=2,
            use_bias=True,
        )
        assert gate.bias is not None
        assert gate.bias.shape == (4,)
        assert gate.bias.dtype == torch.float32
        assert torch.all(gate.bias == 0.0)


# ---------------------------------------------------------------------------
# MoDA k_write RoPE position guard
# ---------------------------------------------------------------------------


class TestMoDAKWriteRopeGuard:
    """Silent-failure guard: depth-write RoPE binds absolute positions 0..T-1.

    MoDA's depth KV cache is currently rebuilt per forward, so cos/sin
    spanning ``[0, T)`` matches the tokens. A future autoregressive-decode
    refactor that keeps the same cos/sin but feeds T=1 tokens at position
    > 0 would silently apply position-0 rotation — the exact class of bug
    already fixed in main.py's GQA/MLA generate paths.

    The guard in ``MoDABlock.forward`` (open_mythos/experimental/moda.py)
    raises RuntimeError when ``cos.shape[-2] != T``. This test exercises
    that guard.
    """

    def test_matching_cos_sin_accepted(self):
        """Happy path: cos/sin aligned with T must run without raising."""
        from open_mythos.experimental.moda import MoDABlock

        torch.manual_seed(42)
        cfg = _tiny_cfg()
        block = MoDABlock(cfg)
        x = torch.randn(B, T, cfg.d_model)
        head_dim = cfg.head_dim
        cos = torch.randn(1, 1, T, head_dim)
        sin = torch.randn(1, 1, T, head_dim)
        # Empty depth cache at layer 0 — first layer has nothing to read.
        out, k_w, v_w, bal = block(x, [], [], cos, sin)
        assert out.shape == x.shape

    def test_mismatched_cos_sin_raises(self):
        """cos/sin shorter than T must raise RuntimeError with an actionable message."""
        from open_mythos.experimental.moda import MoDABlock

        torch.manual_seed(42)
        cfg = _tiny_cfg()
        block = MoDABlock(cfg)
        x = torch.randn(B, T, cfg.d_model)
        head_dim = cfg.head_dim
        # Pass a cos/sin pair with the wrong T to simulate a future-refactor
        # mistake (e.g. passing the full [0, cfg.max_seq_len) table while the
        # current forward only has T tokens).
        wrong_T = T + 2
        cos = torch.randn(1, 1, wrong_T, head_dim)
        sin = torch.randn(1, 1, wrong_T, head_dim)
        with pytest.raises(RuntimeError, match="MoDA k_write RoPE"):
            block(x, [], [], cos, sin)
