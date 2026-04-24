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
        from open_mythos.main import (
            ACTHalting,
            Expert,
            GQAttention,
            LoRAAdapter,
            LTIInjection,
            MLAttention,
            MoEFFN,
            OpenMythos,
            RecurrentBlock,
            RMSNorm,
            TransformerBlock,
            apply_rope,
            loop_index_embedding,
            precompute_rope_freqs,
        )

        return {
            "ACTHalting": ACTHalting,
            "Expert": Expert,
            "GQAttention": GQAttention,
            "LoRAAdapter": LoRAAdapter,
            "LTIInjection": LTIInjection,
            "MLAttention": MLAttention,
            "MoEFFN": MoEFFN,
            "OpenMythos": OpenMythos,
            "RecurrentBlock": RecurrentBlock,
            "RMSNorm": RMSNorm,
            "TransformerBlock": TransformerBlock,
            "apply_rope": apply_rope,
            "loop_index_embedding": loop_index_embedding,
            "precompute_rope_freqs": precompute_rope_freqs,
        }[name]
    raise AttributeError(f"module 'open_mythos' has no attribute {name!r}")
