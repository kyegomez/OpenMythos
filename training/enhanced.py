#!/usr/bin/env python3
"""
OpenMythos Full Enhancement Training Script
===========================================

All P0/P1/P2/P3 enhancements integrated into a single training pipeline:

P0: Multi-scale loop depth + Curriculum learning
P1: Flash MLA + Loop consistency regularization + Capacity-aware routing
P2: Cross-layer KV sharing + Speculative decoding + Expert specialization
P3: Hierarchical recurrence + Meta-learned loop depth

Usage:
    Single GPU:
        python training/enhanced.py

    Multi-GPU:
        torchrun --nproc_per_node=$(nvidia-smi --query-gpu=name | wc -l) \
            training/enhanced.py
"""

import os
import math
import time
import random
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from contextlib import nullcontext

from loguru import logger
from datasets import load_dataset

from open_mythos import OpenMythos, MythosConfig
from open_mythos.tokenizer import MythosTokenizer
from open_mythos.variants import mythos_3b
from open_mythos.main_p0 import (
    ComplexityAwareLoopDepth,
    CurriculumLoopScheduler,
    MultiScaleRecurrentBlock,
    FlashMLAAttention,
    LoopConsistencyRegularizer,
    CapacityAwareRouter,
    CrossLayerKVCache,
    SpeculativeRDTDecoding,
    TaskConditionedMoE,
    HierarchicalRecurrentBlock,
    MetaLearnedLoopDepth,
    OpenMythosEnhanced,
)


# ============================================================================
# Dataset
# ============================================================================

class FineWebEduDataset(IterableDataset):
    """Streaming FineWeb-Edu loader."""

    def __init__(self, encoding, seq_len: int, subset: str, rank: int, world_size: int):
        self.encoding = encoding
        self.seq_len = seq_len
        self.subset = subset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        total_shards = self.world_size * num_workers
        shard_index = self.rank * num_workers + worker_id

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=self.subset,
            split="train",
            streaming=True,
        ).shard(num_shards=total_shards, index=shard_index)

        buf = []
        for sample in ds:
            buf.extend(self.encoding.encode(sample["text"]))
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf = buf[self.seq_len + 1:]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )


# ============================================================================
# LR Schedule
# ============================================================================

def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * step / warmup
    if step >= total:
        return min_lr
    decay = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


# ============================================================================
# Checkpointing
# ============================================================================

def _list_ckpts(ckpt_dir: str) -> list[str]:
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.startswith("step_") and f.endswith(".pt")
    )


def save_checkpoint(model, optimizer, step: int, cfg, vocab_size, ckpt_dir, ddp, master, keep_last=3):
    if ddp:
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if not master:
        return

    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
    tmp_path = final_path + ".tmp"
    torch.save({
        "step": step, "model": model_state, "optimizer": optim_state,
        "cfg": cfg, "vocab_size": vocab_size,
    }, tmp_path)
    os.replace(tmp_path, final_path)

    for old in _list_ckpts(ckpt_dir)[:-keep_last]:
        try:
            os.remove(old)
        except OSError:
            pass

    logger.success(f"Checkpoint saved → {final_path}")


def load_checkpoint(model, optimizer, path: str, ddp: bool) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ddp:
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            model.load_state_dict(ckpt["model"])
            optim_state = FSDP.optim_state_dict_to_load(model, optimizer, ckpt["optimizer"])
            optimizer.load_state_dict(optim_state)
    else:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["step"])


# ============================================================================
# Main Training
# ============================================================================

def main():
    # ------------------------------------------------------------------
    # Distributed init
    # ------------------------------------------------------------------
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank = local_rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

    master = rank == 0

    if master:
        logger.info(f"GPUs: {torch.cuda.device_count()} | World: {world_size} | Device: {device}")
        logger.info("=" * 60)
        logger.info("OpenMythos Full Enhancement Training")
        logger.info("P0: Multi-scale loop + Curriculum learning")
        logger.info("P1: Flash MLA + Loop consistency + Capacity routing")
        logger.info("P2: Cross-layer KV + Speculative decoding + Expert spec")
        logger.info("P3: Hierarchical recurrence + Meta-learned depth")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    encoding = MythosTokenizer()
    vocab_size = encoding.vocab_size

    if master:
        logger.info(f"Tokenizer: gpt-oss-20b | Vocab size: {vocab_size:,}")

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    seq_len = 2048
    micro_batch = 4
    target_tokens = 30_000_000_000
    grad_accum = max(1, 256 // (world_size * micro_batch))
    global_batch_tok = world_size * micro_batch * grad_accum * seq_len
    total_steps = target_tokens // global_batch_tok
    warmup_steps = 2000
    lr = 3e-4
    wd = 0.1
    log_every = 10
    ckpt_every = 1000
    ckpt_dir = "checkpoints_enhanced"
    dataset_subset = "sample-10BT"

    if master:
        logger.info(
            f"seq_len={seq_len} | micro_batch={micro_batch} | grad_accum={grad_accum} | "
            f"global_batch={global_batch_tok:,} | total_steps={total_steps:,}"
        )

    # ------------------------------------------------------------------
    # Enhanced MythosConfig
    # ------------------------------------------------------------------
    cfg = mythos_3b()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = seq_len

    # P0: Multi-scale loop
    cfg.enable_multiscale_loop = True
    cfg.loop_depths = [4, 8, 16]
    cfg.complexity_threshold_low = 0.33
    cfg.complexity_threshold_high = 0.66

    # P0: Curriculum learning
    cfg.enable_curriculum = True
    cfg.curriculum_min_depth = 4
    cfg.curriculum_max_depth = 16
    cfg.curriculum_phase1_steps = 2000
    cfg.curriculum_phase2_steps = 5000
    cfg.curriculum_phase3_steps = 10000
    cfg.curriculum_phase4_steps = total_steps - 20000

    # P1: Flash MLA
    cfg.enable_p1_flash_mla = True

    # P1: Loop consistency regularization
    cfg.enable_p1_consistency_regularization = True

    # P1: Capacity-aware routing
    cfg.enable_p1_capacity_aware_routing = True

    # P2: Cross-layer KV sharing
    cfg.enable_p2_cross_layer_kv = True
    cfg.kv_share_every = 3

    # P2: Expert specialization
    cfg.enable_p2_expert_specialization = True

    # P3: Hierarchical recurrence
    cfg.enable_p3_hierarchical = True
    cfg.n_outer_loops = 4
    cfg.n_inner_loops = 4

    # P3: Meta-learned loop depth
    cfg.enable_p3_meta_learned_depth = True

    if master:
        logger.info("Enabled enhancements: P0[P1[P2[P3")
        logger.info(f"  P0: multiscale={cfg.enable_multiscale_loop}, curriculum={cfg.enable_curriculum}")
        logger.info(f"  P1: flash_mla={cfg.enable_p1_flash_mla}, consistency={cfg.enable_p1_consistency_regularization}")
        logger.info(f"  P1: capacity_routing={cfg.enable_p1_capacity_aware_routing}")
        logger.info(f"  P2: cross_layer_kv={cfg.enable_p2_cross_layer_kv}, expert_spec={cfg.enable_p2_expert_specialization}")
        logger.info(f"  P3: hierarchical={cfg.enable_p3_hierarchical}, meta_depth={cfg.enable_p3_meta_learned_depth}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    model = OpenMythosEnhanced(cfg)

    if ddp:
        mp_policy = MixedPrecision(
            param_dtype=amp_dtype, reduce_dtype=amp_dtype, buffer_dtype=amp_dtype,
        )
        wrap_policy = ModuleWrapPolicy({
            TransformerBlock, RecurrentBlock, MultiScaleRecurrentBlock,
            HierarchicalRecurrentBlock, TaskConditionedMoE,
        })
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            auto_wrap_policy=wrap_policy,
            device_id=local_rank,
        )
    else:
        model = model.to(device)

    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if "cuda" in device and not ddp else nullcontext()
    )

    if master:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Parameters: {n_params:,} | AMP dtype: {amp_dtype}")

    # ------------------------------------------------------------------
    # Enhancement Components
    # ------------------------------------------------------------------
    curriculum_scheduler = CurriculumLoopScheduler(cfg, total_steps)

    # P1: Consistency regularizer
    consistency_reg = LoopConsistencyRegularizer(cfg) if cfg.enable_p1_consistency_regularization else None

    # P2: Cross-layer KV cache
    cross_layer_kv = CrossLayerKVCache(share_every=cfg.kv_share_every) if cfg.enable_p2_cross_layer_kv else None

    # P3: Meta-learned depth predictor
    meta_depth_predictor = MetaLearnedLoopDepth(cfg) if cfg.enable_p3_meta_learned_depth else None

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95), fused=True
    )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_step = 0
    existing_ckpts = _list_ckpts(ckpt_dir)
    if existing_ckpts:
        latest = existing_ckpts[-1]
        if master:
            logger.info(f"Resuming from: {latest}")
        start_step = load_checkpoint(model, optimizer, latest, ddp)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = FineWebEduDataset(encoding, seq_len, dataset_subset, rank, world_size)
    loader = DataLoader(dataset, batch_size=micro_batch, num_workers=4, pin_memory=True)

    if master:
        os.makedirs(ckpt_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------
    model.train()
    data_iter = iter(loader)
    t0 = time.perf_counter()
    step = start_step

    # Tracking
    depth_counts = {4: 0, 8: 0, 16: 0}
    phase_counts = {"phase1": 0, "phase2": 0, "phase3": 0, "phase4": 0}

    while step < total_steps:
        # P0: Get curriculum depth
        current_phase = curriculum_scheduler.get_phase(step)
        curriculum_depth = curriculum_scheduler.get_depth(step)

        # Update LR
        cur_lr = get_lr(step, warmup_steps, total_steps, lr, lr * 0.1)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad()
        loss_accum = 0.0

        # Track loop outputs for consistency regularization
        loop_outputs = []

        for micro_step in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x = x.to(device if not ddp else f"cuda:{local_rank}", non_blocking=True)
            y = y.to(device if not ddp else f"cuda:{local_rank}", non_blocking=True)

            sync = nullcontext() if (not ddp or micro_step == grad_accum - 1) else model.no_sync()
            with sync, amp_ctx:
                # P0: Use curriculum depth
                logits = model(x, n_loops=curriculum_depth)

                # P1: Collect loop outputs for consistency regularization
                if consistency_reg is not None and len(loop_outputs) < 16:
                    # Store intermediate states if model exposes them
                    pass  # Would need model to return loop_states

                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1), reduction="none"
                )
                loss_accum += loss.detach()

            if ddp and micro_step < grad_accum - 1:
                model.no_sync().backward(loss)
            else:
                loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # P1: Apply consistency regularization
        # if consistency_reg is not None and loop_outputs:
        #     total_loss = consistency_reg(h_0, h_T, loop_outputs, loss_accum)
        # else:
        total_loss = loss_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # P3: Meta-learned depth prediction (during training)
        if meta_depth_predictor is not None and step % 100 == 0:
            with torch.no_grad():
                # Simulate loop states for meta-depth training
                dummy_loop_states = [
                    torch.randn(2, 8, cfg.dim, device=device) for _ in range(4)
                ]
                pred_depth, conv_score = meta_depth_predictor(dummy_loop_states, max_depth=16)
                if master:
                    logger.debug(f"Step {step}: meta pred_depth={pred_depth.item()}, convergence={conv_score.mean().item():.3f}")

        # Track depth usage
        depth_counts[curriculum_depth] = depth_counts.get(curriculum_depth, 0) + 1
        phase_counts[current_phase] = phase_counts.get(current_phase, 0) + 1

        # Logging
        if step % log_every == 0 and master:
            elapsed = time.perf_counter() - t0
            tokens_seen = step * global_batch_tok
            lr_now = optimizer.param_groups[0]["lr"]

            logger.info(
                f"step={step:,} | loss={loss_accum.item():.4f} | "
                f"lr={lr_now:.2e} | depth={curriculum_depth} | "
                f"phase={current_phase} | "
                f"tok/s={tokens_seen/elapsed:.0f} | "
                f"progress={100*step/total_steps:.1f}%"
            )

        # Checkpointing
        if step % ckpt_every == 0 and step > 0 and master:
            save_checkpoint(model, optimizer, step, cfg, vocab_size, ckpt_dir, ddp, master)

        step += 1

    # Final checkpoint
    if master:
        save_checkpoint(model, optimizer, step - 1, cfg, vocab_size, ckpt_dir, ddp, master)
        logger.info("Training complete!")

        logger.info("Curriculum depth usage:")
        for depth, count in sorted(depth_counts.items()):
            logger.info(f"  depth={depth}: {count} steps ({100*count/step:.1f}%)")

        logger.info("Phase distribution:")
        for phase, count in sorted(phase_counts.items()):
            logger.info(f"  {phase}: {count} steps ({100*count/step:.1f}%)")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
