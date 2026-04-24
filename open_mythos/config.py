from dataclasses import asdict, dataclass


@dataclass
class MythosConfig:
    """
    Hyperparameter configuration for OpenMythos.

    Core:
        vocab_size      -- token vocabulary size
        dim             -- model hidden dimension
        n_heads         -- number of query attention heads
        n_kv_heads      -- number of key/value heads (GQA; ignored by MLA)
        max_seq_len     -- maximum sequence length for RoPE precomputation
        max_loop_iters  -- default recurrent loop depth T at inference
        prelude_layers  -- number of standard transformer layers before the loop
        coda_layers     -- number of standard transformer layers after the loop

    Attention (attn_type selects between the two):
        attn_type       -- "gqa" for Grouped Query Attention, "mla" for Multi-Latent Attention
        kv_lora_rank    -- [MLA] compressed KV latent dimension stored in the cache
        q_lora_rank     -- [MLA] compressed Q latent dimension
        qk_rope_head_dim-- [MLA] per-head dims that receive RoPE
        qk_nope_head_dim-- [MLA] per-head dims without positional encoding
        v_head_dim      -- [MLA] per-head value dimension

    MoE FFN (used inside the recurrent block):
        n_experts       -- total number of routed expert FFNs
        n_shared_experts-- number of always-active shared experts
        n_experts_per_tok-- top-K experts selected per token by the router
        expert_dim      -- hidden dimension inside each fine-grained expert

    Other:
        act_threshold   -- ACT halting threshold (cumulative probability to stop looping)
        rope_theta      -- RoPE base frequency
        lora_rank       -- rank of the per-loop depth-wise LoRA adapter
    """

    vocab_size: int = 32000
    dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4  # GQA: fewer KV heads than Q heads
    max_seq_len: int = 4096
    max_loop_iters: int = 16  # T — recurrent depth at inference
    prelude_layers: int = 2
    coda_layers: int = 2
    # Attention type: "gqa" | "mla"
    attn_type: str = "mla"
    # MLA params (only used when attn_type="mla")
    kv_lora_rank: int = 512  # compressed KV latent cached instead of full K/V
    q_lora_rank: int = 1536  # compressed Q latent dim
    qk_rope_head_dim: int = 64  # per-head dims that receive RoPE
    qk_nope_head_dim: int = 128  # per-head dims without RoPE
    v_head_dim: int = 128  # per-head value dim
    # MoE
    n_experts: int = 64
    n_shared_experts: int = 2
    n_experts_per_tok: int = 4  # top-K routed
    expert_dim: int = 512  # fine-grained: dim // (n_experts // n_experts_per_tok)
    # ACT halting
    act_threshold: float = 0.99
    # RoPE
    rope_theta: float = 500000.0
    # LoRA depth adaptation
    lora_rank: int = 16
    # Maximum tokens to generate per forward pass
    max_output_tokens: int = 4096
    # Dropout (set 0.0 to disable; 0.1 is standard for pretraining)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.attn_type not in {"gqa", "mla"}:
            raise ValueError(
                f"Unsupported attn_type {self.attn_type!r}; expected 'gqa' or 'mla'"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a plain-Python config dictionary for serialization."""
        return asdict(self)

    def runtime_profile(self) -> dict[str, object]:
        """
        Describe the stable runtime-facing capabilities of this config.

        GatesOfMythos can use this to validate routing decisions without
        inspecting model internals or loading a heavyweight instance first.
        """
        return {
            "model_name": "OpenMythos",
            "attn_type": self.attn_type,
            "supports_kv_cache": True,
            "supports_incremental_decode": True,
            "uses_moe": True,
            "uses_act_halting": True,
            "uses_lti_injection": True,
            "max_context_tokens": self.max_seq_len,
            "max_loop_iters": self.max_loop_iters,
            "max_output_tokens": self.max_output_tokens,
            "cache_layout": {
                "prelude": [f"prelude_{i}" for i in range(self.prelude_layers)],
                "recurrent": "recurrent_loop_{t}",
                "coda": [f"coda_{i}" for i in range(self.coda_layers)],
            },
            "attention_backend": (
                "multi_latent" if self.attn_type == "mla" else "grouped_query"
            ),
        }
