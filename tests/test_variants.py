"""Comprehensive tests for open_mythos/variants.py factory functions."""

import pytest
import torch

from open_mythos.variants import (
    mythos_1b,
    mythos_3b,
    mythos_10b,
    mythos_50b,
    mythos_100b,
    mythos_500b,
    mythos_1t,
)
from open_mythos.main import MythosConfig, OpenMythos

# Ordered from smallest to largest scale.
ALL_FACTORIES = [
    mythos_1b,
    mythos_3b,
    mythos_10b,
    mythos_50b,
    mythos_100b,
    mythos_500b,
    mythos_1t,
]

# Configs that are small enough to actually instantiate on CPU without OOM.
SMALL_FACTORIES = [mythos_1b, mythos_3b]


class TestVariantConfigs:
    """Tests for every variant factory function in variants.py."""

    @pytest.mark.parametrize("factory", ALL_FACTORIES, ids=lambda f: f.__name__)
    def test_each_variant_returns_config(self, factory):
        """All 7 factory functions return a MythosConfig instance."""
        cfg = factory()
        assert isinstance(cfg, MythosConfig)

    @pytest.mark.parametrize("factory", SMALL_FACTORIES, ids=lambda f: f.__name__)
    def test_each_variant_instantiates_model(self, factory):
        """1b and 3b configs can create an OpenMythos model on CPU."""
        cfg = factory()
        model = OpenMythos(cfg)
        assert isinstance(model, torch.nn.Module)
        # Sanity: model should have parameters.
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

    @pytest.mark.parametrize("factory", ALL_FACTORIES, ids=lambda f: f.__name__)
    def test_dim_divisible_by_n_heads(self, factory):
        """dim must be evenly divisible by n_heads."""
        cfg = factory()
        assert cfg.dim % cfg.n_heads == 0, (
            f"{factory.__name__}: dim={cfg.dim} not divisible by n_heads={cfg.n_heads}"
        )

    @pytest.mark.parametrize("factory", ALL_FACTORIES, ids=lambda f: f.__name__)
    def test_n_heads_divisible_by_n_kv_heads(self, factory):
        """n_heads must divide evenly by n_kv_heads (for GQA grouping)."""
        cfg = factory()
        assert cfg.n_heads % cfg.n_kv_heads == 0, (
            f"{factory.__name__}: n_heads={cfg.n_heads} not divisible by "
            f"n_kv_heads={cfg.n_kv_heads}"
        )

    @pytest.mark.parametrize("factory", ALL_FACTORIES, ids=lambda f: f.__name__)
    def test_vocab_size_positive(self, factory):
        """All configs must have a positive vocab_size."""
        cfg = factory()
        assert cfg.vocab_size > 0

    def test_dimensions_increase_with_scale(self):
        """dim must strictly increase from 1b -> 3b -> 10b -> ... -> 1t."""
        dims = [f().dim for f in ALL_FACTORIES]
        for i in range(len(dims) - 1):
            assert dims[i] < dims[i + 1], (
                f"dim did not increase: {ALL_FACTORIES[i].__name__} "
                f"(dim={dims[i]}) >= {ALL_FACTORIES[i+1].__name__} "
                f"(dim={dims[i+1]})"
            )

    def test_expert_count_increases_or_stays(self):
        """Larger models should have n_experts >= the previous scale."""
        expert_counts = [f().n_experts for f in ALL_FACTORIES]
        for i in range(len(expert_counts) - 1):
            assert expert_counts[i] <= expert_counts[i + 1], (
                f"n_experts decreased: {ALL_FACTORIES[i].__name__} "
                f"({expert_counts[i]}) > {ALL_FACTORIES[i+1].__name__} "
                f"({expert_counts[i+1]})"
            )

    @pytest.mark.parametrize("factory", ALL_FACTORIES, ids=lambda f: f.__name__)
    def test_max_loop_iters_positive(self, factory):
        """All configs must have positive max_loop_iters."""
        cfg = factory()
        assert cfg.max_loop_iters > 0

    @pytest.mark.parametrize("factory", ALL_FACTORIES, ids=lambda f: f.__name__)
    def test_attn_type_is_mla(self, factory):
        """All variants use Multi-Latent Attention."""
        cfg = factory()
        assert cfg.attn_type == "mla"

    @pytest.mark.parametrize("factory", ALL_FACTORIES, ids=lambda f: f.__name__)
    def test_act_threshold_valid(self, factory):
        """act_threshold must be in the range (0, 1]."""
        cfg = factory()
        assert 0.0 < cfg.act_threshold <= 1.0, (
            f"{factory.__name__}: act_threshold={cfg.act_threshold} out of (0, 1]"
        )

    @pytest.mark.parametrize("factory", ALL_FACTORIES, ids=lambda f: f.__name__)
    def test_rope_theta_positive(self, factory):
        """All configs must have a positive rope_theta."""
        cfg = factory()
        assert cfg.rope_theta > 0.0
