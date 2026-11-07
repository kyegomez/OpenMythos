"""BitNet b1.58-style ternary weight quantization primitives.

Reference: "The Era of 1-bit LLMs: All Large Language Models are in 1.58
Bits" (arXiv:2402.17764). Weights map to {-1, 0, +1} under an absmean scale
``gamma = mean(|W|)``. The scale is an explicit return value because dropping
it destroys weight-magnitude information entirely; it MUST be persisted next
to the packed payload at export time. Straight-through estimation keeps
gradients flowing to the full-precision master weights during QAT.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from open_mythos.config import MythosConfig

__all__ = ["absmean_quantize", "absmax_act_quant", "BitLinear", "build_linear"]

_ABMEAN_EPS = 1e-8
_ACT_QUANT_EPS = 1e-8


def absmean_quantize(
    weight: Tensor, eps: float = _ABMEAN_EPS
) -> tuple[Tensor, Tensor]:
    """Map a weight tensor to ternary {-1, 0, +1} with absmean scale.

    Returns ``(ternary, gamma)`` where ``gamma = mean(|weight|)`` clamped by
    ``eps``. The forward semantics are ``ternary * gamma``; callers must
    persist ``gamma`` alongside the payload at export time.
    """
    gamma = weight.abs().mean().clamp(min=eps)
    ternary = torch.clamp(torch.round(weight / gamma), -1.0, 1.0)
    return ternary, gamma


def absmax_act_quant(
    x: Tensor, bits: int = 8, group_size: int | None = None
) -> tuple[Tensor, Tensor]:
    """Symmetric absmax activation quantization along the last dimension.

    Splits the final dimension into ``group_size``-wide groups when given
    (the last dimension must be divisible by it). Returns ``(q, scale)``
    with ``x ~= q * scale`` where ``scale`` has shape
    ``x.shape[:-1] + (n_groups,)``.
    """
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}")
    if group_size is None:
        group_size = x.shape[-1]
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    if x.shape[-1] % group_size != 0:
        raise ValueError(
            f"last dim ({x.shape[-1]}) must be divisible by "
            f"group_size ({group_size})"
        )
    qmax = 2 ** (bits - 1) - 1
    grouped = x.reshape(*x.shape[:-1], -1, group_size)
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=_ACT_QUANT_EPS) / qmax
    q = torch.clamp(torch.round(grouped / scale), -qmax, qmax)
    return q.reshape_as(x), scale.reshape(*x.shape[:-1], -1)


class BitLinear(nn.Linear):
    """Linear layer with BitNet b1.58 ternary-weight QAT (W1.58).

    The full-precision ``weight`` is the trainable master copy. Forward
    dequantizes ``ternary * gamma`` and backpropagates through a
    straight-through estimator, so gradients match the unquantized linear
    exactly while the forward sees the ternary approximation.
    ``enabled=False`` runs the plain linear path (parity tests / kill
    switch).

    Subclasses ``nn.Linear`` on purpose: state-dict keys and parameter
    shapes stay identical, so a full-precision checkpoint loads into a
    W1.58 model unchanged (warm-start path), and model-wide ``isinstance``
    init passes cover BitLinear for free.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        enabled: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.enabled = enabled

    def quantized_weight(self) -> tuple[Tensor, Tensor]:
        """Return ``(ternary, gamma)``; ``gamma`` must be exported with the payload."""
        return absmean_quantize(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        if not self.enabled:
            return F.linear(x, self.weight, self.bias)
        ternary, gamma = absmean_quantize(self.weight)
        dequantized = ternary * gamma
        weight = self.weight + (dequantized - self.weight).detach()
        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        mode = "w158" if self.enabled else "off"
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, quant={mode}"
        )


def build_linear(
    cfg: MythosConfig,
    in_features: int,
    out_features: int,
    bias: bool = False,
) -> nn.Linear:
    """Linear factory: ``BitLinear`` under ``quant_mode="w158"``, else plain.

    Quantization placement rule: call this for attention projections, MoE
    experts, and FFN linears. Never for embeddings, the LM head, the MoE
    router, LoRA adapters, or LTI injection — those stay full precision.
    """
    if cfg.quant.mode == "w158":
        return BitLinear(in_features, out_features, bias=bias)
    return nn.Linear(in_features, out_features, bias=bias)
