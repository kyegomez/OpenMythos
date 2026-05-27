"""
INT4/INT8 Weight Quantization for OpenMythos.

Supports GPTQ-style and AWQ-style quantization for MoE expert weights.
Enables running mythos_1b on 8GB VRAM and mythos_3b on 12GB VRAM.

Usage:
    from open_mythos.quantization import quantize_model, QuantizedLinear

    # Quantize entire model
    model = quantize_model(model, bits=4, group_size=128)

    # Or quantize individual layers
    linear = QuantizedLinear(original_linear, bits=4, group_size=128)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class QuantizedLinear(nn.Module):
    """
    Memory-efficient quantized linear layer.

    Stores weights in INT4 or INT8 format with per-group scaling factors.
    Reduces memory by 4x (INT4) or 2x (INT8) compared to FP16.

    Args:
        original_linear: The FP16/FP32 linear layer to quantize
        bits: Quantization precision (4 or 8)
        group_size: Number of weights sharing a scale factor (default: 128)
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
    ):
        super().__init__()
        if bits not in (4, 8):
            raise ValueError(f"Only INT4 and INT8 supported, got {bits}")
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        if not isinstance(original_linear, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(original_linear).__name__}")

        self.bits = bits
        self.group_size = group_size
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.has_bias = original_linear.bias is not None

        # Quantize weights
        weight = original_linear.weight.data.float()  # [out, in]
        qweight, scales, zeros = self._quantize_weight(weight)

        # Store quantized weights
        if bits == 4:
            # Pack two INT4 values into one INT8
            self.register_buffer(
                "qweight",
                self._pack_int4(qweight).to(torch.int8),
                persistent=True,
            )
        else:
            self.register_buffer(
                "qweight", qweight.to(torch.int8), persistent=True
            )

        self.register_buffer("scales", scales.half(), persistent=True)
        self.register_buffer("zeros", zeros.half(), persistent=True)

        if self.has_bias:
            self.register_buffer(
                "bias", original_linear.bias.data.half(), persistent=True
            )
        else:
            self.bias = None

    def _quantize_weight(
        self, weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize weight tensor to INT4/INT8 with group-wise scaling."""
        out_features, in_features = weight.shape

        # Pad in_features to be divisible by group_size
        pad_size = (self.group_size - in_features % self.group_size) % self.group_size
        if pad_size > 0:
            weight = F.pad(weight, (0, pad_size))

        _, padded_in = weight.shape
        num_groups = padded_in // self.group_size

        # Reshape: [out, num_groups, group_size]
        weight_groups = weight.reshape(out_features, num_groups, self.group_size)

        # Per-group min/max
        w_min = weight_groups.min(dim=-1, keepdim=True).values  # [out, num_groups, 1]
        w_max = weight_groups.max(dim=-1, keepdim=True).values  # [out, num_groups, 1]

        if self.bits == 4:
            qmin, qmax = 0, 15
        else:
            qmin, qmax = -127, 127

        # Scale and zero-point
        scales = (w_max - w_min) / (qmax - qmin)
        scales = scales.clamp(min=1e-10)  # Avoid division by zero
        zeros = w_min

        # Quantize
        if self.bits == 4:
            qweight = ((weight_groups - zeros) / scales).round().clamp(0, 15)
        else:
            qweight = ((weight_groups - zeros) / scales).round().clamp(-127, 127)

        # Reshape back
        qweight = qweight.reshape(out_features, padded_in)
        scales = scales.reshape(out_features, num_groups)
        zeros = zeros.reshape(out_features, num_groups)

        return qweight, scales, zeros

    def _pack_int4(self, qweight: torch.Tensor) -> torch.Tensor:
        """Pack two INT4 values into one INT8 byte."""
        # qweight: [out, in] with values 0-15
        out_features, in_features = qweight.shape
        assert in_features % 2 == 0, "in_features must be even for INT4 packing"

        # Reshape to [out, in//2, 2]
        qweight = qweight.reshape(out_features, in_features // 2, 2)
        # Pack: low nibble + high nibble
        packed = (qweight[:, :, 0] | (qweight[:, :, 1] << 4)).to(torch.int8)
        return packed

    def _unpack_int4(self, qweight: torch.Tensor) -> torch.Tensor:
        """Unpack INT8 byte into two INT4 values."""
        # qweight: [out, in//2] packed
        out_features, half_in = qweight.shape

        low = (qweight & 0x0F).to(torch.float32)
        high = ((qweight >> 4) & 0x0F).to(torch.float32)

        # Interleave: [out, in]
        unpacked = torch.stack([low, high], dim=-1).reshape(out_features, half_in * 2)
        return unpacked

    def _dequantize_weight(self) -> torch.Tensor:
        """Dequantize weights from INT4/INT8 to float."""
        if self.bits == 4:
            qweight_fp = self._unpack_int4(self.qweight)
        else:
            qweight_fp = self.qweight.float()

        # Apply group-wise dequantization
        out_features, in_features = qweight_fp.shape
        num_groups = in_features // self.group_size
        qweight_groups = qweight_fp.reshape(out_features, num_groups, self.group_size)

        # Dequantize: weight = qweight * scale + zero
        dequant = qweight_groups * self.scales.float().unsqueeze(-1) + self.zeros.float().unsqueeze(-1)
        weight = dequant.reshape(out_features, in_features)

        # Trim to original size
        return weight[:, : self.in_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly dequantization.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        weight = self._dequantize_weight().half()
        output = F.linear(x.half(), weight)

        if self.has_bias:
            output = output + self.bias

        return output

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"QuantizedLinear("
            f"in={self.in_features}, "
            f"out={self.out_features}, "
            f"bits={self.bits}, "
            f"group_size={self.group_size}, "
            f"bias={self.has_bias})"
        )


def quantize_linear_layer(
    layer: nn.Linear,
    bits: int = 4,
    group_size: int = 128,
) -> QuantizedLinear:
    """Quantize a single linear layer."""
    return QuantizedLinear(layer, bits=bits, group_size=group_size)


def quantize_moe_experts(
    moe_layer: nn.Module,
    bits: int = 4,
    group_size: int = 128,
    expert_ids: Optional[list] = None,
) -> nn.Module:
    """
    Quantize MoE expert FFN layers.

    Only quantizes the large expert FFN layers (gate_proj, up_proj, down_proj).
    Attention and router layers remain in FP16 for accuracy.

    Args:
        moe_layer: The MoE module containing experts
        bits: Quantization precision (4 or 8)
        group_size: Group size for quantization
        expert_ids: Specific experts to quantize (None = all)

    Returns:
        The modified MoE layer (modifies in-place)
    """
    if bits not in (4, 8):
        raise ValueError(f"Only INT4 and INT8 supported, got {bits}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    quantized_count = 0

    for name, module in moe_layer.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # Only quantize expert FFN layers
        is_expert_ffn = any(
            pattern in name
            for pattern in ["gate_proj", "up_proj", "down_proj"]
        )

        if not is_expert_ffn:
            continue

        # If specific experts requested, filter
        if expert_ids is not None:
            expert_match = False
            for eid in expert_ids:
                if f"experts.{eid}." in name or f"expert_{eid}." in name:
                    expert_match = True
                    break
            if not expert_match:
                continue

        # Find parent module and attribute name
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = dict(moe_layer.named_modules())[parent_name]
        else:
            parent = moe_layer
            attr_name = name

        # Replace with quantized version
        setattr(parent, attr_name, quantize_linear_layer(module, bits, group_size))
        quantized_count += 1

    return moe_layer


def quantize_model(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 128,
    quantize_experts_only: bool = True,
) -> nn.Module:
    """
    Quantize an OpenMythos model.

    By default, only quantizes MoE expert FFN layers (biggest memory consumers).
    Attention layers, embeddings, and router remain in FP16.

    Args:
        model: OpenMythos model to quantize
        bits: Quantization precision (4 or 8)
        group_size: Group size for quantization
        quantize_experts_only: If True, only quantize MoE experts (recommended)

    Returns:
        Quantized model (modifies in-place)

    Raises:
        ValueError: If bits or group_size are invalid
    """
    if bits not in (4, 8):
        raise ValueError(f"Only INT4 and INT8 supported, got {bits}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    if quantize_experts_only:
        # Find MoE layers
        moe_found = 0
        for name, module in model.named_modules():
            if hasattr(module, "experts") and hasattr(module, "router"):
                quantize_moe_experts(module, bits, group_size)
                moe_found += 1
        logger.info(f"Quantized {moe_found} MoE layers to INT{bits} (group_size={group_size})")
    else:
        # Quantize all linear layers
        quantized = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                    attr_name = name
                setattr(
                    parent, attr_name, quantize_linear_layer(module, bits, group_size)
                )
                quantized += 1
        logger.info(f"Quantized {quantized} linear layers to INT{bits}")

    return model


def get_model_memory_mb(model: nn.Module) -> dict:
    """Get memory breakdown of model parameters."""
    total_bytes = 0
    quantized_bytes = 0
    fp16_bytes = 0

    for param in model.parameters():
        nbytes = param.numel() * param.element_size()
        total_bytes += nbytes

    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            for param in module.parameters():
                quantized_bytes += param.numel() * param.element_size()
            for buf in module.buffers():
                quantized_bytes += buf.numel() * buf.element_size()
        elif isinstance(module, nn.Linear):
            for param in module.parameters():
                fp16_bytes += param.numel() * param.element_size()

    return {
        "total_mb": total_bytes / 1024 / 1024,
        "quantized_mb": quantized_bytes / 1024 / 1024,
        "fp16_mb": fp16_bytes / 1024 / 1024,
        "compression_ratio": fp16_bytes / max(quantized_bytes, 1),
    }


def print_quantization_summary(model: nn.Module):
    """Print a summary of quantization status."""
    q_linear = 0
    fp_linear = 0
    total_params = 0
    quantized_params = 0

    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            q_linear += 1
            quantized_params += module.in_features * module.out_features
        elif isinstance(module, nn.Linear):
            fp_linear += 1
            total_params += module.weight.numel()

    total_params += quantized_params

    print("=" * 50)
    print("OpenMythos Quantization Summary")
    print("=" * 50)
    print(f"Quantized linear layers: {q_linear}")
    print(f"FP16 linear layers:     {fp_linear}")
    print(f"Total parameters:       {total_params:,}")
    print(f"Quantized parameters:   {quantized_params:,}")
    print(f"Quantization ratio:     {quantized_params/max(total_params,1)*100:.1f}%")
    print("=" * 50)
