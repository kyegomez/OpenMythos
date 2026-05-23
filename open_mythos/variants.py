from open_mythos.main import MythosConfig

# ---------------------------------------------------------------------------
# TINY_CONFIG — unit testing + CPU smoke testing only
# ---------------------------------------------------------------------------


def mythos_tiny() -> MythosConfig:
    """Tiny config for unit tests and CPU smoke tests.

    Targets roughly 1-3M parameters so that tests complete in seconds on
    any machine without a GPU.  Not suitable for training or inference on
    real text — use mythos_1b() or larger for that.

    Architecture choices:
    - dim=128, n_heads=4 (head_dim=32, fits comfortably in CPU cache)
    - MLA attention with minimal lora ranks
    - n_experts=2, n_shared_experts=1, n_experts_per_tok=1 (minimal MoE)
    - recurrent_iters (max_loop_iters)=2
    - prelude_layers=1, coda_layers=1
    - vocab_size=1000, max_seq_len=64
    """
    return MythosConfig(
        vocab_size=1000,
        dim=128,
        n_heads=4,
        n_kv_heads=4,
        max_seq_len=64,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        attn_type="mla",
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
        n_experts=2,
        n_shared_experts=1,
        n_experts_per_tok=1,
        expert_dim=32,
        act_threshold=0.99,
        rope_theta=10000.0,
        lora_rank=4,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# TINY_CONFIG singleton — import directly for convenience in tests
# ---------------------------------------------------------------------------

TINY_CONFIG = mythos_tiny()

# Parameter budget breakdown per variant:
#   total ≈ embed + prelude/coda dense blocks + recurrent MLA + MoE
#   MoE   = 3 * dim * expert_dim * (n_experts + n_shared * n_experts_per_tok)
# expert_dim is solved from the residual budget after all other terms.


def mythos_1b() -> MythosConfig:
    """1B parameter config. Small research/fine-tuning model. dim=2048, 64 experts, 16 loop iters, 4k context."""
    return MythosConfig(
        vocab_size=32000,
        dim=2048,
        n_heads=16,
        n_kv_heads=4,
        max_seq_len=4096,
        max_loop_iters=16,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=256,
        q_lora_rank=512,
        qk_rope_head_dim=32,
        qk_nope_head_dim=64,
        v_head_dim=64,
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=2048,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
    )


def mythos_3b() -> MythosConfig:
    """3B parameter config. Compact inference model. dim=3072, 64 experts, 16 loop iters, 4k context."""
    return MythosConfig(
        vocab_size=32000,
        dim=3072,
        n_heads=24,
        n_kv_heads=6,
        max_seq_len=4096,
        max_loop_iters=16,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=384,
        q_lora_rank=768,
        qk_rope_head_dim=32,
        qk_nope_head_dim=96,
        v_head_dim=96,
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=4096,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
    )


def mythos_10b() -> MythosConfig:
    """10B parameter config. Mid-scale general model. dim=4096, 128 experts, 24 loop iters, 8k context."""
    return MythosConfig(
        vocab_size=32000,
        dim=4096,
        n_heads=32,
        n_kv_heads=8,
        max_seq_len=8192,
        max_loop_iters=24,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=1024,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=128,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=5632,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=16,
    )


def mythos_50b() -> MythosConfig:
    """50B parameter config. Large reasoning model. dim=6144, 256 experts, 32 loop iters, 8k context."""
    return MythosConfig(
        vocab_size=32000,
        dim=6144,
        n_heads=48,
        n_kv_heads=8,
        max_seq_len=8192,
        max_loop_iters=32,
        prelude_layers=3,
        coda_layers=3,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=256,
        n_shared_experts=4,
        n_experts_per_tok=4,
        expert_dim=9728,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=32,
    )


def mythos_100b() -> MythosConfig:
    """100B parameter config. Frontier-class model. dim=8192, 256 experts, 32 loop iters, 1M context, 128k output."""
    return MythosConfig(
        vocab_size=32000,
        dim=8192,
        n_heads=64,
        n_kv_heads=8,
        max_seq_len=1000000,
        max_loop_iters=32,
        prelude_layers=4,
        coda_layers=4,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=2048,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=256,
        n_shared_experts=4,
        n_experts_per_tok=8,
        expert_dim=13568,
        act_threshold=0.99,
        rope_theta=1000000.0,
        lora_rank=64,
        max_output_tokens=131072,
    )


def mythos_500b() -> MythosConfig:
    """500B parameter config. Ultra-scale MoE model. dim=12288, 512 experts, 48 loop iters, 1M context, 128k output."""
    return MythosConfig(
        vocab_size=100000,
        dim=12288,
        n_heads=96,
        n_kv_heads=16,
        max_seq_len=1000000,
        max_loop_iters=48,
        prelude_layers=4,
        coda_layers=4,
        attn_type="mla",
        kv_lora_rank=1024,
        q_lora_rank=3072,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=512,
        n_shared_experts=8,
        n_experts_per_tok=8,
        expert_dim=23040,
        act_threshold=0.99,
        rope_theta=1000000.0,
        lora_rank=128,
        max_output_tokens=131072,
    )


def mythos_1t() -> MythosConfig:
    """1T parameter config. Maximum scale. dim=16384, 512 experts, 64 loop iters, 1M context, 128k output."""
    return MythosConfig(
        vocab_size=100000,
        dim=16384,
        n_heads=128,
        n_kv_heads=16,
        max_seq_len=1000000,
        max_loop_iters=64,
        prelude_layers=6,
        coda_layers=6,
        attn_type="mla",
        kv_lora_rank=1024,
        q_lora_rank=4096,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=512,
        n_shared_experts=8,
        n_experts_per_tok=8,
        expert_dim=34560,
        act_threshold=0.99,
        rope_theta=2000000.0,
        lora_rank=256,
        max_output_tokens=131072,
    )
