from open_mythos.main import MythosConfig

# Parameter budget breakdown per variant:
#   total ≈ embed + prelude/coda dense blocks + recurrent MLA + MoE
#   MoE   = 3 * dim * expert_dim * (n_experts + n_shared * n_experts_per_tok)
# expert_dim is solved from the residual budget after all other terms.
#
# Stability scaling strategy:
#   The original non-linear loop scaling (1B:16, 10B:24, 50B:32, 1T:64) creates
#   compounding instabilities at high iteration counts:
#   - Residual drift: transformer_out accumulates without norm control over 64 steps
#   - Norm explosion: residual stream grows as O(sqrt(T)), ~8x at T=64
#   - Gradient circulation: 64 unrolled steps through shared weights amplifies
#     dominant Jacobian eigenvectors exponentially
#   - Representation collapse: insufficient per-iteration differentiation
#
#   Mitigations applied to all variants with T >= 20:
#   1. recurrence_dropout: linear stochastic depth — later iterations dropped more
#      often, breaking gradient circulation and forcing useful computation at depth
#   2. use_learned_iter_embed: learned embeddings differentiate ALL dim channels
#      per iteration (vs sinusoidal's dim//8), preventing collapse
#   3. ponder_weight: ACT ponder cost regularization penalizes excessive steps,
#      encouraging early halting for easy tokens
#   4. state_norm_eps: post-injection RMSNorm prevents norm explosion
#   5. mup_enabled: μP initialization for width-stable training dynamics
#
#   Loop count scaling is kept sub-linear in log(params):
#     1B→16, 3B→16, 10B→24, 50B→32, 100B→32, 500B→40, 1T→48
#   The 1T variant is reduced from 64→48 loops. At 48 loops with full
#   stabilization, the model achieves comparable effective depth while keeping
#   gradient path length manageable. Depth extrapolation to 64+ loops at
#   inference remains supported via the learned embedding clamping.


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
        recurrence_dropout=0.0,  # T=16 is stable without dropout
        ponder_weight=0.005,
        use_learned_iter_embed=True,
        mup_enabled=False,
        mup_base_width=2048,
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
        recurrence_dropout=0.0,  # T=16 is stable without dropout
        ponder_weight=0.005,
        use_learned_iter_embed=True,
        mup_enabled=False,
        mup_base_width=2048,
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
        recurrence_dropout=0.05,  # mild stochastic depth at T=24
        ponder_weight=0.01,
        use_learned_iter_embed=True,
        mup_enabled=True,
        mup_base_width=2048,
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
        recurrence_dropout=0.10,  # moderate stochastic depth at T=32
        ponder_weight=0.015,
        use_learned_iter_embed=True,
        mup_enabled=True,
        mup_base_width=2048,
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
        recurrence_dropout=0.10,  # moderate stochastic depth at T=32
        ponder_weight=0.015,
        use_learned_iter_embed=True,
        mup_enabled=True,
        mup_base_width=2048,
    )


def mythos_500b() -> MythosConfig:
    """500B parameter config. Ultra-scale MoE model. dim=12288, 512 experts, 40 loop iters, 1M context, 128k output.

    Loop iterations reduced from 48 to 40 vs original. At T=40 with full
    stabilization (state norm + stochastic depth + learned embeddings + ponder
    cost), effective reasoning depth is comparable while gradient path length
    is 17% shorter. Depth extrapolation to 64+ loops at inference is unaffected.
    """
    return MythosConfig(
        vocab_size=100000,
        dim=12288,
        n_heads=96,
        n_kv_heads=16,
        max_seq_len=1000000,
        max_loop_iters=40,
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
        recurrence_dropout=0.15,  # strong stochastic depth at T=40
        ponder_weight=0.02,
        use_learned_iter_embed=True,
        mup_enabled=True,
        mup_base_width=2048,
    )


def mythos_1t() -> MythosConfig:
    """1T parameter config. Maximum scale. dim=16384, 512 experts, 48 loop iters, 1M context, 128k output.

    Loop iterations reduced from 64 to 48 vs original. The original 64-loop
    design triggers compounding pathologies at dim=16384:
    - Residual drift: 64 unnormalized transformer_out terms compound
    - Norm explosion: residual stream grows ~8x its initial magnitude
    - Gradient circulation: 64 unrolled steps through shared weights amplifies
      dominant Jacobian eigenvectors exponentially
    - Representation collapse: sinusoidal embedding only covers 12.5% of channels

    At T=48 with full stabilization:
    - Post-injection RMSNorm bounds residual magnitude
    - Stochastic depth (0.15) breaks gradient circulation at later iterations
    - Learned embeddings differentiate all 16384 channels per iteration
    - Ponder cost (0.02) encourages early halting for easy tokens
    - μP initialization ensures stable training dynamics

    Depth extrapolation to 64+ loops at inference is preserved via learned
    embedding clamping (last trained embedding reused for t > 47).
    """
    return MythosConfig(
        vocab_size=100000,
        dim=16384,
        n_heads=128,
        n_kv_heads=16,
        max_seq_len=1000000,
        max_loop_iters=48,
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
        recurrence_dropout=0.15,  # strong stochastic depth at T=48
        ponder_weight=0.025,
        use_learned_iter_embed=True,
        mup_enabled=True,
        mup_base_width=2048,
    )
