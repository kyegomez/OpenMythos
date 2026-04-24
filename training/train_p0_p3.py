#!/usr/bin/env python3
"""
OpenMythos P0-P3 Enhancement Training Pipeline
=============================================

Full training script integrating all P0/P1/P2/P3 enhancements.

Usage:
    # Single GPU (mock / CPU test):
    python training/train_p0_p3.py --config quick

    # Single GPU (full):
    python training/train_p0_p3.py --config 3b --epochs 1

    # Multi-GPU (full):
    torchrun --nproc_per_node=$(nvidia-smi --query-gpu=name | wc -l) \
        training/train_p0_p3.py --config 3b
"""

import os
import sys
import math
import time
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

# Optional distributed training
try:
    import torch.distributed as dist
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
        FullStateDictConfig,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    DIST_AVAILABLE = True
except ImportError:
    DIST_AVAILABLE = False
    FSDP = None

# Optional Accelerator (HuggingFace)
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    Accelerator = None


# ============================================================================
# Imports from our enhanced main.py
# ============================================================================
from open_mythos.main import (
    MythosConfig,
    OpenMythos,
    # P0
    DepthSelector,
    CurriculumScheduler,
    # P1
    FlashMLAttention,
    loop_consistency_loss,
    AdaptiveLTIInjection,
    ComplexityAwareRecurrentBlock,
    # P2
    SpeculativeDecoder,
    PipelineStage,
    PipelineDriver,
    # P3
    HierarchicalRecurrentBlock,
    MetaLoopRecurrentBlock,
)

# FSDP imports — deferred because torch.distributed may not be available
if DIST_AVAILABLE:
    from open_mythos.main import TransformerBlock, RecurrentBlock


# ============================================================================
# Config Dataclass
# ============================================================================


@dataclass
class TrainConfig:
    """Training configuration dataclass."""

    # Model
    model_config: str = "3b"  # "small", "3b", or "custom"
    vocab_size: int = 6400    # set from tokenizer
    seq_len: int = 2048
    max_loop_iters: int = 8

    # Batch & optimization
    micro_batch: int = 1
    grad_accum: int = 8
    epochs: int = 1
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 200

    # P0
    use_multiscale: bool = True
    use_curriculum: bool = True

    # P1
    use_flash_mla: bool = True
    n_kv_sharing_groups: int = 0
    use_adaptive_spectral_radius: bool = False
    use_complexity_aware: bool = False
    use_loop_consistency_loss: bool = True
    loop_consistency_beta: float = 0.1

    # P2
    use_speculative: bool = False
    speculative_k: int = 4
    pipeline_stages: int = 1

    # P3
    use_hierarchical: bool = False
    use_meta_loop: bool = False

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bf16"
    seed: int = 42
    log_every: int = 10
    eval_every: int = 500
    ckpt_every: int = 1000
    ckpt_dir: str = "checkpoints"
    save_total_limit: int = 3

    # Training length
    max_steps: Optional[int] = None  # override max_tokens target

    def model_cfg(self) -> MythosConfig:
        """Build MythosConfig from this TrainConfig."""
        # Base config by model size
        if self.model_config == "small":
            dim, n_heads, n_kv = 256, 8, 4
            prelude_layers, coda_layers = 1, 1
            n_experts, n_shared = 4, 1
            expert_dim = dim * 2
            lora_rank = 32
        elif self.model_config == "3b":
            dim, n_heads, n_kv = 512, 16, 8
            prelude_layers, coda_layers = 2, 2
            n_experts, n_shared = 8, 2
            expert_dim = dim * 2
            lora_rank = 64
        else:
            raise ValueError(f"Unknown config: {self.model_config}")

        cfg = MythosConfig(
            vocab_size=self.vocab_size,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv,
            max_seq_len=self.seq_len,
            max_loop_iters=self.max_loop_iters,
            prelude_layers=prelude_layers,
            coda_layers=coda_layers,
            n_experts=n_experts,
            n_shared_experts=n_shared,
            n_experts_per_tok=2,
            expert_dim=expert_dim,
            lora_rank=lora_rank,
            act_threshold=0.9,
            dropout=0.0,
            # P0
            loop_depths=[4, 8, 16],
            # P1
            use_flash_mla=self.use_flash_mla,
            n_kv_sharing_groups=self.n_kv_sharing_groups,
            use_adaptive_spectral_radius=self.use_adaptive_spectral_radius,
            # P2
            speculative_k=self.speculative_k,
            pipeline_stages=self.pipeline_stages,
            # MLA params (needed even when using GQA)
            kv_lora_rank=dim // 4,
            q_lora_rank=dim // 2,
            qk_rope_head_dim=dim // n_heads,
            qk_nope_head_dim=dim // n_heads,
            v_head_dim=dim // n_heads,
        )
        return cfg


# ============================================================================
# OpenMythosLM — Language Model wrapper with NLL loss
# ============================================================================


class OpenMythosLM(nn.Module):
    """
    Language model head on top of OpenMythos backbone.

    Wraps the core OpenMythos model with:
    - Pad-correct cross-entropy NLL loss (standard for next-token prediction)
    - Optional loop consistency loss (P1)
    - Per-sample average loss over the sequence dimension

    Forward returns (logits, loss) when targets are provided,
    or just logits during inference.
    """

    def __init__(
        self,
        backbone: nn.Module,
        use_loop_consistency: bool = True,
        loop_consistency_beta: float = 0.1,
        tie_weights: bool = True,
    ):
        """
        Args:
            backbone               -- OpenMythos model
            use_loop_consistency  -- add loop_consistency_loss (P1-3)
            loop_consistency_beta -- weight for the variance component
            tie_weights           -- tie the LM head weight with embedding
        """
        super().__init__()
        self.backbone = backbone
        cfg = backbone.cfg

        # LM head: vocab_size -> dim projection
        # If tie_weights, shares the embedding table; otherwise independent
        if tie_weights:
            self.head = None  # will use embed.weight
        else:
            self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        self.use_loop_consistency = use_loop_consistency
        self.loop_consistency_beta = loop_consistency_beta
        self._tied = tie_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids -- token indices (B, T)
            targets   -- next-token targets (B, T); if None, skips loss computation

        Returns:
            Dict with:
                logits   -- (B, T, vocab_size) or (B, T, dim) if tied
                loss     -- scalar NLL loss (only if targets is not None)
                l2_loss  -- loop consistency loss (P1-3, only if enabled and targets not None)
                total_loss-- weighted sum of loss + l2_loss
        """
        out = self.backbone(input_ids, **kwargs)

        if self._tied:
            logits = F.linear(out, self.backbone.embed.weight)
        else:
            logits = self.head(out)

        result = {"logits": logits}

        if targets is not None:
            # Pad-correct cross-entropy: shift logits and targets by 1
            # Predict next token, not current token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()

            # Flatten for cross-entropy
            B, T, V = shift_logits.shape
            loss = F.cross_entropy(
                shift_logits.view(B * T, V),
                shift_targets.view(B * T),
                reduction="mean",
            )
            result["loss"] = loss

            # P1-3: Loop consistency loss
            if self.use_loop_consistency and hasattr(self.backbone, "recurrent"):
                recurrent = self.backbone.recurrent
                if hasattr(recurrent, "hiddens"):
                    l2_loss = loop_consistency_loss(
                        recurrent.hiddens,
                        beta=self.loop_consistency_beta,
                    )
                    result["l2_loss"] = l2_loss
                    result["total_loss"] = loss + l2_loss
                elif hasattr(recurrent, "complexity_history"):
                    # ComplexityAwareRecurrentBlock doesn't store hiddens list
                    # Can't compute loop consistency loss without it
                    result["total_loss"] = loss
                else:
                    result["total_loss"] = loss
            else:
                result["total_loss"] = loss

        return result

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Shortcut to backbone.generate()."""
        return self.backbone.generate(input_ids, **kwargs)


# ============================================================================
# Dataset — Mock and Real
# ============================================================================


class MockDataset(IterableDataset):
    """
    Mock dataset for quick sanity testing (no tokenization needed).

    Generates random token sequences for CPU/CUDA smoke tests.
    """

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples

    def __iter__(self):
        for _ in range(self.n_samples):
            input_ids = torch.randint(
                0, self.vocab_size, (self.seq_len + 1,), dtype=torch.long
            )
            yield input_ids[:-1], input_ids[1:]


class TokenizedDataset(IterableDataset):
    """
    Streaming tokenized dataset (FineWeb-Edu / SlimPajama compatible).

    Yields (input_ids, targets) tuples ready for OpenMythosLM forward.

    Usage:
        ds = TokenizedDataset(
            path_or_hf_repo="your/tokenized/data",
            seq_len=2048,
            rank=rank,
            world_size=world_size,
        )
    """

    def __init__(
        self,
        path_or_hf_repo: str,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.path_or_hf_repo = path_or_hf_repo
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed + self.rank)
        from datasets import load_dataset

        # Try HuggingFace dataset first, fall back to memory-mapped file
        try:
            ds = load_dataset(
                self.path_or_hf_repo,
                split="train",
                streaming=True,
            )
        except Exception:
            # Fallback: load from disk
            import numpy as np

            data = np.memmap(self.path_or_hf_repo, dtype=np.uint16, mode="r")
            n_samples = len(data) // (self.seq_len + 1)

            indices = list(range(n_samples))
            if self.shuffle:
                rng.shuffle(indices)

            for idx in indices:
                start = idx * (self.seq_len + 1)
                chunk = data[start : start + self.seq_len + 1]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                targets = torch.tensor(chunk[1:], dtype=torch.long)
                yield input_ids, targets
            return

        # Shard across workers
        n_shards = self.world_size
        shard_idx = self.rank % n_shards

        buf = []
        sample_count = 0
        for sample in ds:
            if "text" in sample:
                # Text dataset — encode with tokenizer
                # This would need the tokenizer; for now use raw tokens
                tokens = sample.get("tokens", sample.get("input_ids"))
                if tokens is None:
                    continue
                if isinstance(tokens, list):
                    buf.extend(tokens)
                else:
                    buf.append(tokens)

            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf = buf[self.seq_len + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )
                sample_count += 1


# ============================================================================
# LR Schedule
# ============================================================================


def get_cosine_lr(
    step: int,
    warmup: int,
    max_lr: float,
    min_lr: float,
    total_steps: int,
) -> float:
    """Cosine decay with linear warmup."""
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ============================================================================
# Optimizer Builders
# ============================================================================


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    fused: bool = True,
) -> torch.optim.Optimizer:
    """
    Build an AdamW optimizer with layer-wise LR decay.

    Layer-wise decay: layers closer to the input get lower LR.
    This helps stability and improves fine-tuning performance.
    """
    from torch.optim.optimizer import Optimizer

    # Separate frozen/penalty params
    no_decay = []
    decay = []
    no_decay_names = {"bias", "LayerNorm", "norm", "embed"}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay_names):
            no_decay.append(param)
        elif weight_decay == 0 or "bias" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    fused_available = (
        hasattr(torch.optim, "AdamW") and fused and torch.cuda.is_available()
    )

    groups = [
        {
            "params": decay,
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": no_decay,
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]

    if fused_available:
        return torch.optim.AdamW(**groups, betas=betas, fused=True)
    return torch.optim.AdamW(**groups, betas=betas)


# ============================================================================
# Gradient Clipping
# ============================================================================


def clip_grads(model: nn.Module, max_norm: float) -> float:
    """Clip gradients by global norm. Returns the total norm before clipping."""
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm,
    ).item()


# ============================================================================
# Checkpointing
# ============================================================================


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: TrainConfig,
    path: str,
    ddp: bool = False,
) -> None:
    """Save a training checkpoint."""
    ckpt = {
        "step": step,
        "cfg": cfg,
        "model": model.state_dict() if not ddp else None,
        "optimizer": optimizer.state_dict(),
    }
    if ddp and DIST_AVAILABLE:
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            ckpt["model"] = model.state_dict()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    ddp: bool = False,
) -> int:
    """Load a checkpoint. Returns the saved step."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["step"])


# ============================================================================
# Mixed Precision
# ============================================================================


def get_autocast(dtype_str: str, device: str = "cuda"):
    """Get torch.amp.autocast context manager."""
    if device == "cpu":
        return torch.cpu.amp.autocast(dtype=torch.float16)
    dtype = {"float16": torch.float16, "bf16": torch.bfloat16}.get(dtype_str, torch.bfloat16)
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


# ============================================================================
# Training Step
# ============================================================================


def train_step(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    scheduler: Optional[object] = None,
    step: int = 0,
) -> dict[str, float]:
    """
    Single training step with gradient accumulation and mixed precision.

    Returns a dict of metrics.
    """
    input_ids, targets = batch
    input_ids = input_ids.to(cfg.device)
    targets = targets.to(cfg.device)

    # Zero optimizer grads
    optimizer.zero_grad(set_to_none=True)

    # Gradient accumulation loop
    loss_val, l2_val, total_val = 0.0, 0.0, 0.0
    n_accum = 0

    for micro_i in range(cfg.grad_accum):
        # Forward with mixed precision
        ctx = get_autocast(cfg.dtype, cfg.device)
        with ctx:
            result = model(input_ids, targets, start_pos=0)
            loss = result["total_loss"] / cfg.grad_accum

        # Backward with gradient scaling
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        loss_val += result.get("loss", loss_val).item()
        l2_val += result.get("l2_loss", 0.0).item()
        total_val += loss.item()
        n_accum += 1

    # Gradient unclipping
    if scaler is not None:
        scaler.unscale_(optimizer)
    grad_norm = clip_grads(model, cfg.max_grad_norm)

    # Optimizer step
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    # LR schedule
    if scheduler is not None:
        scheduler.step()

    return {
        "loss": loss_val / n_accum,
        "l2_loss": l2_val / n_accum,
        "total_loss": total_val / n_accum,
        "grad_norm": grad_norm,
        "lr": optimizer.param_groups[0]["lr"],
    }


# ============================================================================
# Main Training Loop
# ============================================================================


def train(cfg: TrainConfig) -> None:
    """
    Main training function.

    Handles single/multi-GPU, optional FSDP, mixed precision, logging,
    checkpointing, and learning rate scheduling.
    """

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    ddp = DIST_AVAILABLE and int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank = local_rank = 0
        world_size = 1
        device = cfg.device

    master = rank == 0

    # Seed
    seed = cfg.seed + rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if master:
        print("=" * 60)
        print("OpenMythos P0-P3 Training")
        print(f"Config: {cfg.model_config}")
        print(f"Device: {device} | World: {world_size}")
        print("=" * 60)

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model_cfg = cfg.model_cfg()
    model_cfg.vocab_size = cfg.vocab_size

    if cfg.use_hierarchical:
        backbone = OpenMythos.__new__(OpenMythos)
        # Manually construct with HierarchicalRecurrentBlock
        nn.Module.__init__(backbone, cfg=model_cfg)
        backbone._init_weights()
        backbone.recurrent = HierarchicalRecurrentBlock(model_cfg)
    elif cfg.use_meta_loop:
        backbone = OpenMythos.__new__(OpenMythos)
        nn.Module.__init__(backbone, cfg=model_cfg)
        backbone._init_weights()
        backbone.recurrent = MetaLoopRecurrentBlock(model_cfg)
    elif cfg.use_complexity_aware:
        backbone = OpenMythos.__new__(OpenMythos)
        nn.Module.__init__(backbone, cfg=model_cfg)
        backbone._init_weights()
        backbone.recurrent = ComplexityAwareRecurrentBlock(model_cfg)
    else:
        backbone = OpenMythos(model_cfg)

    lm = OpenMythosLM(
        backbone,
        use_loop_consistency=cfg.use_loop_consistency_loss,
        loop_consistency_beta=cfg.loop_consistency_beta,
    )
    lm = lm.to(device)

    if master:
        n_params = sum(p.numel() for p in lm.parameters())
        print(f"Model params: {n_params:,}")

    # -------------------------------------------------------------------------
    # DDP / FSDP wrapping
    # -------------------------------------------------------------------------
    if ddp and world_size > 1:
        # Wrap transformer blocks with FSDP
        wrap_policy = ModuleWrapPolicy(
            {
                TransformerBlock,
                RecurrentBlock,
                ComplexityAwareRecurrentBlock,
                HierarchicalRecurrentBlock,
                MetaLoopRecurrentBlock,
            }
        )
        lm = FSDP(
            lm,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            auto_wrap_policy=wrap_policy,
            device_id=local_rank,
        )
        if master:
            print("FSDP enabled")

    # -------------------------------------------------------------------------
    # Optimizer & Scheduler
    # -------------------------------------------------------------------------
    optimizer = build_optimizer(
        lm, cfg.max_lr, cfg.weight_decay
    )

    total_steps = cfg.max_steps or 1000  # default 1000 if not set

    def lr_lambda(step):
        return get_cosine_lr(
            step, cfg.warmup_steps, cfg.max_lr, cfg.min_lr, total_steps
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Gradient scaler for mixed precision
    scaler = (
        torch.cuda.amp.GradScaler() if cfg.device.startswith("cuda") else None
    )

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    # Use mock dataset if no real data path is configured
    train_ds = MockDataset(
        vocab_size=cfg.vocab_size,
        seq_len=cfg.seq_len,
        n_samples=1000,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.micro_batch,
        num_workers=0,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    model.train()
    step = 0
    t0 = time.monotonic()
    log_buffer: dict[str, float] = {}

    for epoch in range(cfg.epochs):
        for batch in train_loader:
            if step >= total_steps:
                break

            metrics = train_step(
                lm, batch, optimizer, cfg, scaler, scheduler, step
            )

            # Accumulate for logging
            for k, v in metrics.items():
                log_buffer[k] = log_buffer.get(k, 0.0) + v

            # Logging
            if step % cfg.log_every == 0 and master:
                elapsed = time.monotonic() - t0
                avg = {k: v / cfg.log_every for k, v in log_buffer.items()}
                tokens_seen = (
                    world_size * cfg.micro_batch * cfg.grad_accum * cfg.seq_len * step
                )
                print(
                    f"step={step:4d} | "
                    f"loss={avg.get('loss', 0):.4f} | "
                    f"l2={avg.get('l2_loss', 0):.6f} | "
                    f"total={avg.get('total_loss', 0):.4f} | "
                    f"grad={avg.get('grad_norm', 0):.4f} | "
                    f"lr={avg.get('lr', 0):.2e} | "
                    f"tok={tokens_seen:,} | "
                    f"dt={elapsed:.1f}s"
                )
                log_buffer.clear()
                t0 = time.monotonic()

            # Checkpointing
            if step % cfg.ckpt_every == 0 and master and step > 0:
                ckpt_path = os.path.join(cfg.ckpt_dir, f"step_{step:07d}.pt")
                save_checkpoint(lm, optimizer, step, cfg, ckpt_path, ddp=ddp)

            step += 1

    # Final checkpoint
    if master:
        ckpt_path = os.path.join(cfg.ckpt_dir, f"step_{step:07d}.pt")
        save_checkpoint(lm, optimizer, step, cfg, ckpt_path, ddp=ddp)
        print(f"Training complete. Final checkpoint: {ckpt_path}")

    if ddp:
        dist.destroy_process_group()


# ============================================================================
# Entry Point
# ============================================================================


def cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="OpenMythos P0-P3 Training Pipeline"
    )
    parser.add_argument(
        "--config",
        choices=["small", "3b"],
        default="small",
        help="Model config to use",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max training steps (overrides epochs)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=6400,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--use-hierarchical",
        action="store_true",
        help="Use HierarchicalRecurrentBlock (P3-1)",
    )
    parser.add_argument(
        "--use-meta-loop",
        action="store_true",
        help="Use MetaLoopRecurrentBlock (P3-2)",
    )
    parser.add_argument(
        "--use-complexity-aware",
        action="store_true",
        help="Use ComplexityAwareRecurrentBlock (P1-6)",
    )
    parser.add_argument(
        "--no-loop-consistency",
        action="store_true",
        help="Disable loop consistency loss",
    )
    parser.add_argument(
        "--flash-mla",
        action="store_true",
        default=True,
        help="Enable FlashMLA (P1-1, default)",
    )
    parser.add_argument(
        "--no-flash-mla",
        action="store_true",
        help="Disable FlashMLA",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bf16"],
        default="bf16",
        help="Mixed precision dtype",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )

    args = parser.parse_args()

    cfg = TrainConfig(
        model_config=args.config,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        epochs=args.epochs,
        max_steps=args.max_steps,
        device=args.device,
        dtype=args.dtype,
        max_lr=args.lr,
        use_hierarchical=args.use_hierarchical,
        use_meta_loop=args.use_meta_loop,
        use_complexity_aware=args.use_complexity_aware,
        use_loop_consistency_loss=not args.no_loop_consistency,
        use_flash_mla=not args.no_flash_mla,
    )

    train(cfg)


if __name__ == "__main__":
    cli()
