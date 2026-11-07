from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from open_mythos.config import MythosConfig
from open_mythos.primitives.quant import build_linear


class Expert(nn.Module):
    """
    Single SwiGLU feed-forward expert.

    Implements the gated linear unit variant: output = down(silu(gate(x)) * up(x)).
    Used both as individual routed experts inside MoEFFN and as the standard dense
    FFN in prelude/coda blocks (where expert_dim = dim * 4 // 3).
    """

    def __init__(self, dim: int, expert_dim: int, cfg: Optional[MythosConfig] = None):
        """
        Args:
            dim        -- input and output feature dimension
            expert_dim -- inner (hidden) dimension of the expert
            cfg        -- when given, linears honor cfg.quant.mode (BitLinear
                          under "w158"); None keeps plain nn.Linear
        """
        super().__init__()

        def make_linear(in_features: int, out_features: int) -> nn.Linear:
            if cfg is not None:
                return build_linear(cfg, in_features, out_features)
            return nn.Linear(in_features, out_features, bias=False)

        self.gate = make_linear(dim, expert_dim)
        self.up = make_linear(dim, expert_dim)
        self.down = make_linear(expert_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input of shape (..., dim)
        Returns:
            Tensor of shape (..., dim)
        """
        return self.down(F.silu(self.gate(x)) * self.up(x))
