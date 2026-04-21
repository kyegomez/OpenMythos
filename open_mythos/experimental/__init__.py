"""
Experimental / research-line modules.

These components are **not** part of the canonical OpenMythos architecture
(Prelude / Recurrent / Coda with MLA + DeepSeek-MoE) exposed at the package
root. They live here to be importable for research without polluting the
public API surface, and their contracts (names, signatures, behavior) are
explicitly unstable.

Included:
    - MoDA (Mixture-of-Depths Attention): a depth-aware attention variant
      that attends across layer depth in addition to sequence position,
      fused with DeepSeek-style MoE FFNs.

Stability: **no guarantees**. Import at your own risk; APIs here may change
or disappear in any commit. Do not build production training configs that
depend on this subpackage.
"""

from open_mythos.experimental.moda import (
    DeepSeekExpert,
    DeepSeekGate,
    DeepSeekMoE,
    MoDAAttention,
    MoDABlock,
    MoDAConfig,
    MoDAModel,
    RMSNorm,
    RotaryEmbedding,
)

__all__ = [
    "MoDAConfig",
    "MoDAModel",
    "MoDABlock",
    "MoDAAttention",
    "DeepSeekExpert",
    "DeepSeekGate",
    "DeepSeekMoE",
    "RMSNorm",
    "RotaryEmbedding",
]
