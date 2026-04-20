#!/usr/bin/env python3
"""
OpenMythos pretraining on FineWeb-Edu with FSDP + AdamW.

Single GPU:
    python training/3b_fine_web_edu.py

Multi-GPU:
    torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") training/3b_fine_web_edu.py
"""

import os
import math
import time
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

from datasets import load_dataset

from open_mythos import OpenMythos
from open_mythos.main import TransformerBlock, RecurrentBlock
from open_mythos.variants import mythos_3b
from open_mythos.tokenizer import MythosTokenizer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FineWebEduDataset(IterableDataset):
    def __init__(
        self,
        encoding,
        seq_len: int,
        subset: str,
        rank: int,
        world_size: int,
        shard_offset: int = 0,
    ):
        self.encoding = encoding
        self.seq_len = seq_len
        self.subset = subset
        self.rank = rank
        self.world_size = world_size
        self.shard_offset = shard_offset

    def __iter__(self):
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        total_shards = self.world_size * num_workers
        shard_index = (self.rank * num_workers + worker_id + self.shard_offset) % total_shards

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
                buf = buf[self.seq_len + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------


def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * step / warmup
    if step >= total:
        return min_lr
    decay = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    val_loader: DataLoader,
    steps: int,
    vocab_size: int,
    amp_ctx,
    ddp: bool,
    device: str,
    local_rank: int,
) -> float:
    """Estimate validation loss over a small number of micro-batches."""
    model_was_training = model.training
    model.eval()

    val_iter = iter(val_loader)
    losses = []
    for _ in range(steps):
        try:
            x, y = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            x, y = next(val_iter)

        x = x.to(device if not ddp else f"cuda:{local_rank}", non_blocking=True)
        y = y.to(device if not ddp else f"cuda:{local_rank}", non_blocking=True)

        with amp_ctx:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        losses.append(loss.detach())

    loss_tensor = torch.stack(losses).mean()
    if ddp:
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor = loss_tensor / dist.get_world_size()

    if model_was_training:
        model.train()
    return float(loss_tensor.item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
        print(f"GPUs: {torch.cuda.device_count()}  |  World size: {world_size}  |  Device: {device}")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    encoding = MythosTokenizer()
    vocab_size = encoding.vocab_size

    if master:
        print(f"Tokenizer: gpt-oss-20b  |  Vocab size: {vocab_size:,}")

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
    val_every = 200
    val_steps = 20
    ckpt_dir = "checkpoints"
    dataset_subset = "sample-10BT"  # → sample-100BT or "default" for full run

    if master:
        print(
            f"seq_len={seq_len} | micro_batch={micro_batch} | grad_accum={grad_accum}\n"
            f"global_batch_tokens={global_batch_tok:,} | total_steps={total_steps:,}"
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    cfg = mythos_3b()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = seq_len

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    model = OpenMythos(cfg)

    if ddp:
        mp_policy = MixedPrecision(
            param_dtype=amp_dtype,
            reduce_dtype=amp_dtype,
            buffer_dtype=amp_dtype,
        )
        wrap_policy = ModuleWrapPolicy({TransformerBlock, RecurrentBlock})
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
            if "cuda" in device
            else nullcontext()
        )

    # FSDP handles its own mixed precision; only need autocast for single-GPU
    amp_ctx = nullcontext() if ddp else amp_ctx  # type: ignore[possibly-undefined]

    if master:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}  |  AMP dtype: {amp_dtype}")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95), fused=True
    )
    use_grad_scaler = (amp_dtype == torch.float16) and ("cuda" in device) and (not ddp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    # ------------------------------------------------------------------
    # Dataset + DataLoader
    # ------------------------------------------------------------------
    dataset = FineWebEduDataset(encoding, seq_len, dataset_subset, rank, world_size)
    loader = DataLoader(dataset, batch_size=micro_batch, num_workers=4, pin_memory=True)
    val_dataset = FineWebEduDataset(
        encoding, seq_len, dataset_subset, rank, world_size, shard_offset=1
    )
    val_loader = DataLoader(
        val_dataset, batch_size=micro_batch, num_workers=2, pin_memory=True
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    if master:
        os.makedirs(ckpt_dir, exist_ok=True)

    model.train()
    data_iter = iter(loader)
    t0 = time.perf_counter()
    step = 0

    while step < total_steps:
        cur_lr = get_lr(step, warmup_steps, total_steps, lr, lr * 0.1)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x = x.to(device if not ddp else f"cuda:{local_rank}", non_blocking=True)
            y = y.to(device if not ddp else f"cuda:{local_rank}", non_blocking=True)

            sync = (
                nullcontext()
                if (not ddp or micro_step == grad_accum - 1)
                else model.no_sync()
            )
            with sync, amp_ctx:
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                )
                loss = loss / grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_accum += loss.item()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        step += 1

        if master and step % log_every == 0:
            dt = time.perf_counter() - t0
            tok_per_sec = global_batch_tok * log_every / dt
            tokens_seen = step * global_batch_tok
            print(
                f"step {step:6d}/{total_steps} | loss {loss_accum:.4f} "
                f"| lr {cur_lr:.2e} | {tok_per_sec / 1e6:.2f}M tok/s "
                f"| {tokens_seen / 1e9:.1f}B tokens seen"
            )
            t0 = time.perf_counter()

        if step % val_every == 0:
            val_loss = evaluate_loss(
                model=model,
                val_loader=val_loader,
                steps=val_steps,
                vocab_size=vocab_size,
                amp_ctx=amp_ctx,
                ddp=ddp,
                device=device,
                local_rank=local_rank,
            )
            if master:
                val_ppl = math.exp(min(val_loss, 20.0))
                print(
                    f"validation | step {step:6d}/{total_steps} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}"
                )

        if master and step % ckpt_every == 0:
            path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
            if ddp:
                with FSDP.state_dict_type(
                    model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                ):
                    model_state = model.state_dict()
            else:
                model_state = model.state_dict()
            torch.save(
                {
                    "step": step,
                    "model": model_state,
                    "optimizer": optimizer.state_dict(),
                    "cfg": cfg,
                    "vocab_size": vocab_size,
                },
                path,
            )
            print(f"Checkpoint saved → {path}")

    if ddp:
        dist.destroy_process_group()

    if master:
        print("Training complete.")


if __name__ == "__main__":
    main()
