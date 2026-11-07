"""InjectionProtocol — structural type for recurrent-loop injection modules.

Stage 4.3. Currently satisfied by LTIInjection only. Alternative
injection schemes (identity skip, gated skip, Mamba-style SSM step) would
fit the same (h, e, trans_out) -> h_new signature.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class InjectionProtocol(Protocol):
    def __call__(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        trans_out: torch.Tensor,
    ) -> torch.Tensor: ...
