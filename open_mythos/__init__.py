"""OpenMythos package root.

Heavy, torch-dependent symbols (the model, variants) are imported *lazily* via
PEP 562 ``__getattr__`` so that torch-free subpackages — notably
``open_mythos.dj`` — can be imported and used without pulling in the full model
stack. ``from open_mythos import OpenMythos`` still works exactly as before; the
import just happens on first attribute access.
"""

from importlib import import_module
from typing import TYPE_CHECKING

# name -> submodule it lives in. Resolved lazily on first access.
_LAZY_EXPORTS = {
    # open_mythos.main
    "ACTHalting": "open_mythos.main",
    "Expert": "open_mythos.main",
    "GQAttention": "open_mythos.main",
    "LoRAAdapter": "open_mythos.main",
    "LTIInjection": "open_mythos.main",
    "MLAttention": "open_mythos.main",
    "MoEFFN": "open_mythos.main",
    "MythosConfig": "open_mythos.main",
    "OpenMythos": "open_mythos.main",
    "RecurrentBlock": "open_mythos.main",
    "RMSNorm": "open_mythos.main",
    "TransformerBlock": "open_mythos.main",
    "apply_rope": "open_mythos.main",
    "loop_index_embedding": "open_mythos.main",
    "precompute_rope_freqs": "open_mythos.main",
    # open_mythos.tokenizer
    "MythosTokenizer": "open_mythos.tokenizer",
    # open_mythos.variants
    "mythos_1b": "open_mythos.variants",
    "mythos_3b": "open_mythos.variants",
    "mythos_10b": "open_mythos.variants",
    "mythos_50b": "open_mythos.variants",
    "mythos_100b": "open_mythos.variants",
    "mythos_500b": "open_mythos.variants",
    "mythos_1t": "open_mythos.variants",
}

if TYPE_CHECKING:  # keep static analysis / IDEs happy
    from open_mythos.main import (  # noqa: F401
        ACTHalting,
        Expert,
        GQAttention,
        LoRAAdapter,
        LTIInjection,
        MLAttention,
        MoEFFN,
        MythosConfig,
        OpenMythos,
        RecurrentBlock,
        RMSNorm,
        TransformerBlock,
        apply_rope,
        loop_index_embedding,
        precompute_rope_freqs,
    )
    from open_mythos.tokenizer import MythosTokenizer  # noqa: F401
    from open_mythos.variants import (  # noqa: F401
        mythos_1b,
        mythos_1t,
        mythos_3b,
        mythos_10b,
        mythos_50b,
        mythos_100b,
        mythos_500b,
    )


def __getattr__(name: str):
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value  # cache so subsequent access is direct
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))


__all__ = [
    "MythosConfig",
    "RMSNorm",
    "GQAttention",
    "MLAttention",
    "Expert",
    "MoEFFN",
    "LoRAAdapter",
    "TransformerBlock",
    "LTIInjection",
    "ACTHalting",
    "RecurrentBlock",
    "OpenMythos",
    "precompute_rope_freqs",
    "apply_rope",
    "loop_index_embedding",
    "mythos_1b",
    "mythos_3b",
    "mythos_10b",
    "mythos_50b",
    "mythos_100b",
    "mythos_500b",
    "mythos_1t",
    "MythosTokenizer",
]
