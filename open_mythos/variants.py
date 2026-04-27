from open_mythos.main import MythosConfig

# Parameter budget breakdown per variant:
#   total ≈ embed + prelude/coda dense blocks + recurrent MLA + MoE
#   MoE   = 3 * dim * expert_dim * (n_experts + n_shared * n_experts_per_tok)
# expert_dim is solved from the residual budget after all other terms.
#
# MiniMax-M2.7 reference config (see minimax_m2_config below):
#   Architecture source: MiniMax-01 technical report (MiniMaxAI/MiniMax-M2.7 on HuggingFace)
#   230B total params, ~7B active per token, 48Q/8KV heads, head_dim=128, 32 routed experts


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


def minimax_m2_config() -> MythosConfig:
    """OpenMythos configuration matching the MiniMax-M2.7 architecture.

    MiniMax-M2.7 (MiniMaxAI/MiniMax-M2.7 on HuggingFace) is a 230B-parameter sparse
    MoE language model.  Key structural dimensions expressed in OpenMythos terms:

    * ``dim = 6144``  — matches MiniMax-M2.7's hidden size
    * ``n_heads = 48`` / ``n_kv_heads = 8``  — 6× GQA ratio (48 query heads, 8 KV heads)
    * ``qk_rope_head_dim = 64``, ``qk_nope_head_dim = 64``, ``v_head_dim = 128``
      — the per-head dimensions satisfy MiniMax-M2.7's head_dim = 128 for both
      the Q/K attention key size (rope + nope = 128) and the value projection (128)
    * ``n_experts = 32``, ``n_shared_experts = 2``, ``n_experts_per_tok = 4``
      — mirrors MiniMax-M2.7's fine-grained MoE design (32 routed + 2 always-on)
    * ``max_seq_len = 40960``  — MiniMax-M2.7's 40K-token native context window
    * ``rope_theta = 10_000_000``  — high RoPE base for long-context stability

    Use this config to instantiate an OpenMythos model whose structural dimensions
    reflect the MiniMax-M2.7 design point.  Weights are randomly initialized; load
    ``MiniMaxAI/MiniMax-M2.7`` from HuggingFace for production inference.

    Example::

        from open_mythos.variants import minimax_m2_config
        from open_mythos.main import OpenMythos
        import torch

        cfg = minimax_m2_config()
        model = OpenMythos(cfg)
        ids = torch.randint(0, cfg.vocab_size, (1, 16))
        logits = model(ids, n_loops=4)
        print(logits.shape)  # (1, 16, 200064)
    """
    return MythosConfig(
        vocab_size=200064,
        dim=6144,
        n_heads=48,
        n_kv_heads=8,
        max_seq_len=40960,
        max_loop_iters=16,
        prelude_layers=4,
        coda_layers=4,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        qk_nope_head_dim=64,
        v_head_dim=128,
        n_experts=32,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=8192,
        act_threshold=0.99,
        rope_theta=10_000_000.0,
        lora_rank=32,
        max_output_tokens=4096,
    )
