#!/usr/bin/env python3
"""
OpenMythos pretraining on FineWeb-Edu with Muon optimizer.

Single GPU:
    python train.py

Multi-GPU (auto-detects GPU count):
    torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") train.py
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from contextlib import nullcontext

from datasets import load_dataset

from open_mythos import OpenMythos
from open_mythos.variants import mythos_3b
from open_mythos.tokenizer import MythosTokenizer
from torch.optim import Muon


def build_optimizers(model: nn.Module, muon_lr: float, adamw_lr: float, wd: float):
    """Muon for 2D weight matrices; AdamW for embeddings, norms, biases."""
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            p.ndim >= 2
            and "embed" not in name
            and "norm" not in name
            and "scale" not in name
        ):
            muon_params.append(p)
        else:
            adamw_params.append(p)
    muon = Muon(muon_params, lr=muon_lr)
    adamw = torch.optim.AdamW(
        adamw_params, lr=adamw_lr, weight_decay=wd, betas=(0.9, 0.95), fused=True
    )
    return muon, adamw


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FineWebEduDataset(IterableDataset):
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

        # shard first by DDP rank, then by dataloader worker
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # ------------------------------------------------------------------
    # Distributed init — works for single GPU (python train.py)
    # and multi-GPU (torchrun --nproc_per_node=N train.py)
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
        n_gpu = torch.cuda.device_count()
        print(
            f"GPUs detected: {n_gpu}  |  World size: {world_size}  |  Device: {device}"
        )

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
    micro_batch = 4  # sequences per GPU per grad-accum step
    target_tokens = 30_000_000_000  # 30B token run
    grad_accum = max(1, 256 // (world_size * micro_batch))
    global_batch_tok = world_size * micro_batch * grad_accum * seq_len
    total_steps = target_tokens // global_batch_tok
    warmup_steps = 2000
    muon_lr = 0.02
    adamw_lr = 3e-4
    wd = 0.1
    log_every = 10
    ckpt_every = 1000
    ckpt_dir = "checkpoints"
    dataset_subset = "sample-10BT"  # → sample-100BT or "default" for full run

    if master:
        print(
            f"seq_len={seq_len} | micro_batch={micro_batch} | grad_accum={grad_accum}\n"
            f"global_batch_tokens={global_batch_tok:,} | total_steps={total_steps:,}"
        )

    # ------------------------------------------------------------------
    # Model — override vocab_size to match tokenizer
    # ------------------------------------------------------------------
    cfg = mythos_3b()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = seq_len

    model = OpenMythos(cfg).to(device)

    # Mixed precision
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_supported else torch.float16
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if "cuda" in device
        else nullcontext()
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    if master:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}  |  AMP dtype: {amp_dtype}")

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if ddp else model

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    muon, adamw = build_optimizers(raw_model, muon_lr, adamw_lr, wd)

    # ------------------------------------------------------------------
    # Dataset + DataLoader
    # ------------------------------------------------------------------
    dataset = FineWebEduDataset(encoding, seq_len, dataset_subset, rank, world_size)
    loader = DataLoader(dataset, batch_size=micro_batch, num_workers=4, pin_memory=True)

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
        cur_muon_lr = get_lr(step, warmup_steps, total_steps, muon_lr, muon_lr * 0.1)
        cur_adamw_lr = get_lr(step, warmup_steps, total_steps, adamw_lr, adamw_lr * 0.1)
        for g in muon.param_groups:
            g["lr"] = cur_muon_lr
        for g in adamw.param_groups:
            g["lr"] = cur_adamw_lr

        muon.zero_grad()
        adamw.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Defer DDP gradient sync until the last micro-step
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

            scaler.scale(loss).backward()
            loss_accum += loss.item()

        # Unscale, clip, step both optimizers
        scaler.unscale_(muon)
        scaler.unscale_(adamw)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(muon)
        scaler.step(adamw)
        scaler.update()

        step += 1

        if master and step % log_every == 0:
            dt = time.perf_counter() - t0
            tok_per_sec = global_batch_tok * log_every / dt
            tokens_seen = step * global_batch_tok
            print(
                f"step {step:6d}/{total_steps} | loss {loss_accum:.4f} "
                f"| muon_lr {cur_muon_lr:.2e} | adamw_lr {cur_adamw_lr:.2e} "
                f"| {tok_per_sec / 1e6:.2f}M tok/s | {tokens_seen / 1e9:.1f}B tokens seen"
            )
            t0 = time.perf_counter()

        if master and step % ckpt_every == 0:
            path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
            torch.save(
                {
                    "step": step,
                    "model": raw_model.state_dict(),
                    "muon": muon.state_dict(),
                    "adamw": adamw.state_dict(),
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
