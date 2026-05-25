"""
Experiment config for OpenMythos loop-scaling validation.

Two model variants with identical param count/compute:
- looped:   max_loop_iters=8   (the OpenMythos architecture)
- baseline: max_loop_iters=1   (equivalent to a plain transformer)

Training data is FineWeb-Edu sample-10BT (streaming); we train on ~1B tokens.
"""

from dataclasses import dataclass, field, asdict
from open_mythos.main import MythosConfig


def mythos_150m(max_loop_iters: int = 8) -> MythosConfig:
    """
    ~150M parameter config tuned for a single H100.

    With max_loop_iters=8 and MoE (16 experts, top-2), activated params per
    token ≈ 80M; total params ≈ 150M. The looped block is run 8 times so the
    effective compute per forward matches a ~8x deeper plain transformer.
    """
    return MythosConfig(
        vocab_size=50257,
        dim=768,
        n_heads=12,
        n_kv_heads=4,
        max_seq_len=1024,
        max_loop_iters=max_loop_iters,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=192,
        q_lora_rank=384,
        qk_rope_head_dim=32,
        qk_nope_head_dim=32,
        v_head_dim=32,
        n_experts=16,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=1536,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=16,
        dropout=0.0,
    )


@dataclass
class TrainConfig:
    # Model
    max_loop_iters: int = 8
    run_name: str = "looped_8"

    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    tokenizer: str = "gpt2"
    seq_len: int = 1024

    # Training (H100 can fit B=32 directly; adjust to 16 for A40-class GPUs)
    batch_size: int = 32
    grad_accum_steps: int = 1    # 32 * 1024 = 32,768 tokens/step
    max_steps: int = 15000       # 15k * 32k = ~490M tokens
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Logging & checkpointing
    log_every: int = 20
    eval_every: int = 2000
    ckpt_every: int = 5000
    output_dir: str = "/workspace/runs"

    # Precision
    dtype: str = "bfloat16"

    # Eval
    eval_seq_len: int = 1024
    eval_batch_size: int = 8
    eval_num_batches: int = 50

    def to_dict(self):
        return asdict(self)
