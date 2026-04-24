from importlib import import_module

from open_mythos.config import MythosConfig
from open_mythos.tokenizer import MythosTokenizer, get_vocab_size, load_tokenizer
from open_mythos.variants import (
    mythos_1b,
    mythos_1t,
    mythos_3b,
    mythos_10b,
    mythos_50b,
    mythos_100b,
    mythos_500b,
)

_MAIN_EXPORTS = {
    "ACTHalting",
    "Expert",
    "GQAttention",
    "LoRAAdapter",
    "LTIInjection",
    "MLAttention",
    "MoEFFN",
    "OpenMythos",
    "RecurrentBlock",
    "RMSNorm",
    "TransformerBlock",
    "apply_rope",
    "loop_index_embedding",
    "precompute_rope_freqs",
}
_MAIN_MODULE = None

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
    "load_tokenizer",
    "get_vocab_size",
    "MythosTokenizer",
]


def __getattr__(name: str):
    if name in _MAIN_EXPORTS:
        global _MAIN_MODULE
        if _MAIN_MODULE is None:
            _MAIN_MODULE = import_module("open_mythos.main")
        value = getattr(_MAIN_MODULE, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'open_mythos' has no attribute {name!r}")
