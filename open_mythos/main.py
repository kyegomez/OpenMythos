"""Backward-compat shim. The real implementations live in:
  - open_mythos.config       — MythosConfig
  - open_mythos.primitives   — RMSNorm, RoPE utils, Expert
  - open_mythos.attention    — GQAttention, MLAttention
  - open_mythos.moe          — MoEFFN
  - open_mythos.blocks       — TransformerBlock
  - open_mythos.recurrence   — LTI/ACT/LoRA/RecurrentBlock
  - open_mythos.model        — OpenMythos
  - open_mythos.kv_cache     — StaticKVCache

Kept so that `from open_mythos.main import X` existing user code keeps working.
Assigning to attributes on this module that were historically defined here
(e.g. `open_mythos.main.apply_rope = spy`) propagates to the implementation
modules that now own those names, preserving monkey-patch semantics.
"""

import sys as _sys
from types import ModuleType as _ModuleType

import torch.nn.functional as F  # noqa: F401 — kept as attribute for monkey-patching

from open_mythos.attention.gqa import GQAttention
from open_mythos.attention.mla import MLAttention
from open_mythos.blocks import TransformerBlock
from open_mythos.config import MythosConfig
from open_mythos.model import OpenMythos
from open_mythos.moe import MoEFFN
from open_mythos.primitives.expert import Expert
from open_mythos.primitives.norm import RMSNorm
from open_mythos.primitives.rope import apply_rope, loop_index_embedding, precompute_rope_freqs
from open_mythos.recurrence.act import ACTHalting
from open_mythos.recurrence.block import RecurrentBlock
from open_mythos.recurrence.lora import LoRAAdapter
from open_mythos.recurrence.lti import LTIInjection

__all__ = [
    "MythosConfig",
    "RMSNorm",
    "precompute_rope_freqs", "apply_rope", "loop_index_embedding",
    "Expert",
    "GQAttention", "MLAttention",
    "MoEFFN",
    "TransformerBlock",
    "LoRAAdapter", "LTIInjection", "ACTHalting", "RecurrentBlock",
    "OpenMythos",
]


# ---------------------------------------------------------------------------
# Legacy monkey-patch propagation.
#
# Before this module was split, callers could write e.g. `mm.apply_rope = spy`
# and have every call site inside the old main.py pick up the replacement,
# because every reference resolved through the module's own namespace. After
# the split the attention/recurrence modules import apply_rope directly, so a
# plain re-export would silently desync from patches made on `main`. We install
# a forwarding module class so attribute writes on `open_mythos.main` also land
# on the submodules that actually own the symbol.
# ---------------------------------------------------------------------------

_FORWARDING_TARGETS: dict[str, tuple[str, ...]] = {
    # Each symbol maps to the list of fully-qualified modules whose
    # attribute of the same name must also be updated.
    "apply_rope": (
        "open_mythos.primitives.rope",
        "open_mythos.attention.gqa",
        "open_mythos.attention.mla",
    ),
    "precompute_rope_freqs": (
        "open_mythos.primitives.rope",
        "open_mythos.model",
    ),
    "loop_index_embedding": (
        "open_mythos.primitives.rope",
    ),
}


class _ForwardingModule(_ModuleType):
    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        for target_name in _FORWARDING_TARGETS.get(name, ()):
            target_mod = _sys.modules.get(target_name)
            if target_mod is not None:
                setattr(target_mod, name, value)


_sys.modules[__name__].__class__ = _ForwardingModule
