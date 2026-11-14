from open_mythos.main import (
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
from open_mythos.variants import (
    mythos_1b,
    mythos_1t,
    mythos_3b,
    mythos_10b,
    mythos_50b,
    mythos_100b,
    mythos_500b,
)

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


def __getattr__(name: str):
    """Lazily import the tokenizer so the core model works without `transformers`.

    MythosTokenizer wraps a HuggingFace AutoTokenizer, pulling in the heavy
    `transformers` dependency. Importing it lazily (PEP 562) means
    `import open_mythos` and the model classes stay usable in environments that
    only install torch.
    """
    if name == "MythosTokenizer":
        from open_mythos.tokenizer import MythosTokenizer

        return MythosTokenizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
