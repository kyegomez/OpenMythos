"""Tests for quantization and expert offloading modules."""

import torch
import torch.nn as nn
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_mythos.quantization import (
    QuantizedLinear,
    quantize_linear_layer,
    quantize_model,
    get_model_memory_mb,
)
from open_mythos.expert_offloader import ExpertOffloader


class TestQuantizedLinear:
    """Tests for QuantizedLinear module."""

    def test_int4_quantization(self):
        """Test INT4 quantization preserves approximate values."""
        linear = nn.Linear(256, 128, bias=True)
        linear.weight.data.uniform_(-1, 1)
        linear.bias.data.uniform_(-0.1, 0.1)

        ql = QuantizedLinear(linear, bits=4, group_size=64)

        x = torch.randn(1, 16, 256)
        out_original = linear(x)
        out_quantized = ql(x)

        # INT4 is lossy, but should be in the same ballpark
        # Allow up to 20% relative error
        rel_error = (out_original - out_quantized).abs() / (out_original.abs() + 1e-6)
        assert rel_error.mean() < 0.2, f"Relative error too high: {rel_error.mean():.4f}"

    def test_int8_quantization(self):
        """Test INT8 quantization is more accurate than INT4."""
        linear = nn.Linear(256, 128, bias=True)
        linear.weight.data.uniform_(-1, 1)

        ql4 = QuantizedLinear(linear, bits=4, group_size=64)
        ql8 = QuantizedLinear(linear, bits=8, group_size=64)

        x = torch.randn(1, 16, 256)
        out_original = linear(x)
        out_int4 = ql4(x)
        out_int8 = ql8(x)

        err_int4 = (out_original - out_int4).abs().mean()
        err_int8 = (out_original - out_int8).abs().mean()

        # INT8 should be more accurate
        assert err_int8 < err_int4, "INT8 should have lower error than INT4"

    def test_memory_reduction(self):
        """Test that quantization reduces memory."""
        linear = nn.Linear(1024, 1024, bias=False)

        original_bytes = linear.weight.numel() * 2  # FP16 = 2 bytes

        ql4 = QuantizedLinear(linear, bits=4, group_size=128)
        quantized_bytes = sum(
            b.numel() * b.element_size() for b in ql4.buffers()
        )

        # INT4 should be ~4x smaller
        assert quantized_bytes < original_bytes / 2, (
            f"Quantized ({quantized_bytes}B) should be much smaller than "
            f"original ({original_bytes}B)"
        )

    def test_output_shape(self):
        """Test output shape is correct."""
        linear = nn.Linear(256, 128, bias=True)
        ql = QuantizedLinear(linear, bits=4, group_size=64)

        x = torch.randn(2, 32, 256)
        out = ql(x)

        assert out.shape == (2, 32, 128), f"Expected shape (2, 32, 128), got {out.shape}"

    def test_group_size_validation(self):
        """Test invalid group_size raises error."""
        linear = nn.Linear(256, 128)
        with pytest.raises(AssertionError):
            QuantizedLinear(linear, bits=4, group_size=0)

    def test_bits_validation(self):
        """Test invalid bits raises error."""
        linear = nn.Linear(256, 128)
        with pytest.raises(AssertionError):
            QuantizedLinear(linear, bits=3, group_size=64)


class TestQuantizeModel:
    """Tests for model-level quantization."""

    def test_quantize_linear_layer(self):
        """Test single layer quantization."""
        linear = nn.Linear(256, 128)
        ql = quantize_linear_layer(linear, bits=4, group_size=64)
        assert isinstance(ql, QuantizedLinear)

    def test_quantize_model_experts_only(self):
        """Test model quantization only affects expert layers."""
        # Create a minimal model with MoE-like structure
        class FakeMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(64, 128),
                        nn.Linear(128, 64),
                    )
                    for _ in range(4)
                ])
                self.router = nn.Linear(64, 4)

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 64)
                self.moe = FakeMoE()

        model = FakeModel()
        model = quantize_model(model, bits=4, group_size=32, quantize_experts_only=True)

        # Router should NOT be quantized
        assert isinstance(model.moe.router, nn.Linear)
        # Expert FFNs should be quantized
        for expert in model.moe.experts:
            for layer in expert.modules():
                if isinstance(layer, nn.Linear):
                    assert isinstance(layer, QuantizedLinear)


class TestExpertOffloader:
    """Tests for ExpertOffloader."""

    def test_discover_moe_layers(self):
        """Test MoE layer discovery."""
        class FakeMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = nn.ModuleList([nn.Linear(64, 64) for _ in range(8)])
                self.router = nn.Linear(64, 8)

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.moe = FakeMoE()

        model = FakeModel()
        offloader = ExpertOffloader(model, gpu_experts=2, cache_experts=4)

        assert "moe" in offloader.moe_layers

    def test_stats_tracking(self):
        """Test statistics are tracked."""
        class FakeMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = nn.ModuleList([nn.Linear(16, 16) for _ in range(4)])
                self.router = nn.Linear(16, 4)

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.moe = FakeMoE()

        model = FakeModel()
        offloader = ExpertOffloader(model, gpu_experts=2, cache_experts=4)

        stats = offloader.get_stats()
        assert "gpu_hits" in stats
        assert "total_requests" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
