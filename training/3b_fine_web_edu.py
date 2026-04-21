#!/usr/bin/env python3
"""
OpenMythos pretraining on FineWeb-Edu with FSDP + AdamW.

Single GPU:
    python training/3b_fine_web_edu.py

Multi-GPU:
    torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") training/3b_fine_web_edu.py

Hardening vs. the upstream reference trainer:
    - All RNGs seeded (python / numpy / torch / cuda) and the seed is
      checkpointed so a resume reproduces the same data stream given the
      same shard position.
    - Checkpoint carries RNG state, AMP scaler state, and torch + cuda
      versions — enough to survive a node swap mid-run.
    - Graceful SIGTERM / SIGINT handling: drains the current microbatch
      loop, saves a final atomic checkpoint, then tears down NCCL. No
      half-written `.tmp` files, no hang.
    - NaN / Inf loss guard per microstep. A bad batch zeros its grad
      contribution; if the whole accumulation window is bad we skip
      optimizer.step() but still advance the step counter so LR schedule
      and logging stay monotonic.
    - ShardedGradScaler on the fp16 path (Volta / Pascal); bf16 path
      (Ampere+) runs with FSDP MixedPrecision and no scaler, which is
      the officially supported combination.
    - File logging with loguru rotation (100 MB + 7-day retention) so
      long runs don't fill the disk with a single multi-GB log.
    - EOS injected between packed documents via
      `encoding.encode_with_eos()` so the model never sees a
      cross-document attention window without a boundary marker.
    - `model.update_router_biases(ddp=ddp)` called after every
      successful `optimizer.step()` — this is what actually drives the
      aux-loss-free load balancing; without it, `router_bias` stays at
      zeros forever.
    - Micro-batch loss accumulated on-device; single `.item()` per step.
"""

import math
import os
import random
import signal
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import load_dataset
from loguru import logger
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from open_mythos import OpenMythos
from open_mythos.main import RecurrentBlock, TransformerBlock
from open_mythos.tokenizer import MythosTokenizer
from open_mythos.variants import mythos_3b


# ---------------------------------------------------------------------------
# Determinism / RNG
# ---------------------------------------------------------------------------


DEFAULT_SEED = 1337


def seed_everything(seed: int) -> None:
    """
    Seed every RNG we touch so two runs with the same seed draw the same
    microbatches (given an unchanged dataset shard order).

    This is not a promise of bit-exact model outputs — cuBLAS / cuDNN
    kernels pick different algorithms depending on workspace allocator
    state, and FSDP all-reduce order is non-deterministic under NCCL.
    What it does buy is a reproducible *data path* and a reproducible
    initialization — the two things that actually cause "why did my
    loss curve change" regressions.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def snapshot_rng() -> dict:
    """Collect every RNG's state so a checkpoint can resume reproducibly."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: Optional[dict]) -> None:
    """Inverse of `snapshot_rng`; tolerates missing keys from older ckpts."""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FineWebEduDataset(IterableDataset):
    """
    Streaming FineWeb-Edu loader yielding fixed-length (input, target) pairs.

    FineWeb-Edu is trillions of tokens, so `streaming=True` pulls shards on
    demand instead of materializing to disk. Sharding is two-dimensional —
    `world_size` ranks × `num_workers` DataLoader workers per rank — and each
    `(rank, worker_id)` deterministically owns one shard of the global stream.
    That gives disjoint coverage without any cross-process coordination.

    Documents are encoded with `encode_with_eos` so concatenated docs have
    an EOS token between them — the model sees explicit boundaries instead
    of silently attending across unrelated documents (which inflates loss
    and teaches spurious long-range dependencies).

    Streaming datasets are not seekable, so a resumed run re-enters its shard
    from the beginning. Acceptable at pretraining scale: the chance of
    re-playing the same tokens before the run ends is negligible versus the
    cost of a true resumable loader.
    """

    def __init__(
        self,
        encoding: MythosTokenizer,
        seq_len: int,
        subset: str,
        rank: int,
        world_size: int,
    ):
        """
        Args:
            encoding   -- tokenizer exposing ``encode_with_eos(str) -> list[int]``
            seq_len    -- context length; every yielded pair has this many tokens
            subset     -- FineWeb-Edu config name (e.g. "sample-10BT", "default")
            rank       -- global rank of this process within the distributed job
            world_size -- total number of distributed processes
        """
        self.encoding = encoding
        self.seq_len = seq_len
        self.subset = subset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        """
        Yield ``(input_ids, target_ids)`` tensors of length ``seq_len`` forever.

        Inputs and targets are shifted by one for next-token prediction —
        ``target[i] == input[i + 1]``. Documents are concatenated into a rolling
        buffer and sliced into fixed-length chunks, packing short docs together
        and splitting long ones. This keeps every step at the same shape,
        which under FSDP avoids recompute from variable-length inputs and
        removes the need for a pad-aware attention mask.
        """
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

        buf: list[int] = []
        for sample in ds:
            text = sample.get("text")
            ids = self.encoding.encode_with_eos(text) if text else []
            if not ids:
                continue
            buf.extend(ids)
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
    """
    Linear warmup → half-cosine decay to ``min_lr``.

    Standard language-model pretraining schedule. The warmup phase prevents
    Adam's second-moment estimate from collapsing to a huge LR in the first
    few steps when gradients are noisy. The cosine tail lets the model make
    small, increasingly conservative updates near the end of training rather
    than crashing to ``min_lr`` at a fixed step.

    Behavior by region:
        step < warmup                 → linear ramp 0 → max_lr
        warmup ≤ step < total         → cosine decay max_lr → min_lr
        step ≥ total                  → clamped at min_lr (safety for
                                        off-by-one step counters at the end
                                        of training)
    """
    if step < warmup:
        return max_lr * step / max(1, warmup)
    if step >= total:
        return min_lr
    decay = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


CKPT_SUFFIX = ".pt"
CKPT_PREFIX = "step_"


def _list_ckpts(ckpt_dir: str) -> list[str]:
    """
    Return checkpoint paths in ``ckpt_dir`` sorted oldest → newest.

    Relies on the zero-padded ``step_{0000000}.pt`` filename convention so
    lexicographic sort matches chronological order. Changing the filename
    format elsewhere without updating the pad width would silently break
    both ``keep_last`` pruning and resume-latest on startup, since both pick
    the last element of this list.
    """
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.startswith(CKPT_PREFIX) and f.endswith(CKPT_SUFFIX)
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[ShardedGradScaler],
    step: int,
    cfg,
    vocab_size: int,
    seed: int,
    ckpt_dir: str,
    ddp: bool,
    master: bool,
    keep_last: int = 3,
) -> None:
    """
    Gather full model + optimizer state, write atomically, prune old files.

    Under FSDP both states are collected inside a single FULL_STATE_DICT
    context so the optim-state tensors bind to fully-unsharded parameters;
    mixing contexts between model and optimizer has caused silent divergence
    on resume in past torch versions. The temp-file + os.replace write means
    a kill mid-save leaves the previous checkpoint intact instead of a
    truncated .pt file. Non-master ranks participate in the FSDP gather
    (otherwise the collective would hang) but exit before touching disk.

    The checkpoint also carries the RNG state snapshot, the AMP scaler
    state (when applicable), and the training seed — enough for a fresh
    process on a different node to resume with deterministic data order.
    """
    if ddp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
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
    final_path = os.path.join(ckpt_dir, f"{CKPT_PREFIX}{step:07d}{CKPT_SUFFIX}")
    tmp_path = final_path + ".tmp"

    payload = {
        "step": step,
        "model": model_state,
        "optimizer": optim_state,
        "cfg": cfg,
        "vocab_size": vocab_size,
        "seed": seed,
        "rng": snapshot_rng(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)

    # Best-effort fsync of the directory so the rename is durable across
    # a crash/power-loss — torch.save already fsyncs the file itself.
    try:
        dir_fd = os.open(ckpt_dir, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError as exc:
        logger.warning(f"Directory fsync failed (non-fatal) on {ckpt_dir}: {exc}")

    for old in _list_ckpts(ckpt_dir)[:-keep_last]:
        try:
            os.remove(old)
        except OSError as exc:
            logger.warning(f"Failed to prune old checkpoint {old}: {exc}")

    logger.success(f"Checkpoint saved → {final_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[ShardedGradScaler],
    path: str,
    ddp: bool,
) -> int:
    """
    Restore model + optimizer + scaler + RNG from disk, returning the step
    to resume at.

    Every rank reads the file (``rank0_only=False`` on load) so FSDP has
    access to the full state on each rank — the complement to the
    ``rank0_only=True`` save path. Must mirror save's single-context
    pattern; splitting the model and optimizer loads across two
    ``state_dict_type`` blocks has historically produced optimizer state
    bound to the wrong shard shapes.

    ``weights_only=False`` is required because the checkpoint contains the
    pickled ``cfg`` dataclass — flip to ``weights_only=True`` only if you
    separate config out.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if ddp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            model.load_state_dict(ckpt["model"])
            optim_state = FSDP.optim_state_dict_to_load(
                model=model,
                optim=optimizer,
                optim_state_dict=ckpt["optimizer"],
            )
            optimizer.load_state_dict(optim_state)
    else:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    restore_rng(ckpt.get("rng"))

    return int(ckpt["step"])


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


class ShutdownFlag:
    """
    Cooperative shutdown request set by SIGTERM / SIGINT handlers.

    Using a class attribute (rather than a bare global) because loguru's
    multiprocessing fork tracker sometimes re-imports modules in child
    processes, which would reset a module-level global. A class attribute
    on a module-level singleton survives that.
    """

    requested = False

    @classmethod
    def request(cls, signum: int, frame) -> None:
        """
        Signal handler: mark the flag and log once. The training loop
        polls this between microsteps and exits cleanly with a final
        atomic checkpoint. A second signal falls through to default
        handling (hard kill) — so a stuck rank can always be force-killed.
        """
        if cls.requested:
            logger.warning(f"Second signal {signum} received — falling through")
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        cls.requested = True
        logger.warning(f"Signal {signum} received — draining to final checkpoint")


def install_signal_handlers() -> None:
    """
    Install SIGTERM + SIGINT handlers on the main rank / main thread only.

    Data-loader worker processes inherit these handlers too, which is
    harmless: they don't have a model to checkpoint, and the flag they
    would flip lives in a different address space.
    """
    signal.signal(signal.SIGTERM, ShutdownFlag.request)
    signal.signal(signal.SIGINT, ShutdownFlag.request)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def configure_logging(log_dir: str, master: bool, rank: int) -> None:
    """
    Attach a rotating file sink to loguru on every rank.

    Rank 0 keeps the default stderr sink; non-master ranks silence stderr
    to avoid interleaved output chaos, but still write a per-rank file log
    so post-mortem debugging sees every rank's view. 100 MB per file and
    7-day retention is enough to cover a multi-week pretraining run while
    keeping total log volume bounded.
    """
    os.makedirs(log_dir, exist_ok=True)

    if not master:
        # Replace stderr with a null sink so non-master ranks don't spam the
        # terminal; per-rank file sink still captures everything.
        logger.remove()

    logger.add(
        os.path.join(log_dir, f"train.rank{rank}.log"),
        rotation="100 MB",
        retention="7 days",
        compression="gz",
        enqueue=True,  # thread-safe + signal-safe
        level="INFO",
        backtrace=True,
        diagnose=False,  # diagnose=True can leak tensor contents into logs
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """
    End-to-end pretraining entry point.

    Order matters: distributed init must run before any CUDA allocation, the
    tokenizer must exist before the model is built (vocab_size flows into
    cfg), and FSDP must wrap the model before the optimizer is constructed
    (FSDP re-flattens parameters, so an optimizer built on the unwrapped
    model would track stale param objects). Resume then loads state into the
    already-constructed optimizer in-place.
    """
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

    log_dir = "logs"
    configure_logging(log_dir, master, rank)
    install_signal_handlers()

    if master:
        logger.info(
            f"GPUs: {torch.cuda.device_count()}  |  "
            f"World size: {world_size}  |  Device: {device}"
        )
        logger.info(f"Torch: {torch.__version__} | CUDA: {torch.version.cuda}")

    # ------------------------------------------------------------------
    # RNG seeding — uses per-rank offset so each rank's in-process RNG
    # starts from a different state (torch.manual_seed is local), but the
    # numeric seed stored in the checkpoint is the same base value.
    # ------------------------------------------------------------------
    seed = int(os.environ.get("OPENMYTHOS_SEED", DEFAULT_SEED))
    seed_everything(seed + rank)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    encoding = MythosTokenizer()
    vocab_size = encoding.vocab_size
    eos_id = encoding.eos_token_id

    if master:
        logger.info(
            f"Tokenizer: gpt-oss-20b  |  Vocab size: {vocab_size:,}  |  "
            f"EOS id: {eos_id}"
        )
        if eos_id is None:
            logger.warning(
                "Tokenizer has no EOS token — documents will be concatenated "
                "without a boundary marker (not ideal for pretraining quality)"
            )

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
    grad_clip = 1.0
    log_every = 10
    ckpt_every = 1000
    ckpt_dir = "checkpoints"
    dataset_subset = "sample-10BT"  # → sample-100BT or "default" for full run

    if master:
        logger.info(
            f"seq_len={seq_len} | micro_batch={micro_batch} | "
            f"grad_accum={grad_accum} | "
            f"global_batch_tokens={global_batch_tok:,} | "
            f"total_steps={total_steps:,} | seed={seed}"
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    cfg = mythos_3b()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = seq_len
    # Re-validate after mutating vocab_size / max_seq_len — the default
    # variant values pass, but an operator who edits them at the CLI
    # gets a clean error here instead of a mid-step crash.
    cfg.__post_init__()

    if master:
        logger.info(f"Config: {asdict(cfg)}")

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    use_scaler = (amp_dtype == torch.float16) and torch.cuda.is_available()

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
        amp_ctx = nullcontext()
    else:
        model = model.to(device)
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
            if "cuda" in device
            else nullcontext()
        )

    if master:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Parameters: {n_params:,}  |  AMP dtype: {amp_dtype}  |  "
            f"Scaler: {'on' if use_scaler else 'off'}"
        )

    # ------------------------------------------------------------------
    # Optimizer + GradScaler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95), fused=True
    )

    # ShardedGradScaler = FSDP-aware unscale / all-reduce inf detection.
    # Only needed when training in fp16; bf16 has enough dynamic range to
    # forgo loss scaling entirely. On CPU / single-GPU we still use the
    # sharded variant — it degenerates to a normal GradScaler but keeps
    # the call sites identical.
    scaler = ShardedGradScaler(enabled=use_scaler) if use_scaler else None

    # ------------------------------------------------------------------
    # Resume from latest checkpoint (if any)
    # ------------------------------------------------------------------
    start_step = 0
    existing_ckpts = _list_ckpts(ckpt_dir)
    if existing_ckpts:
        latest = existing_ckpts[-1]
        if master:
            logger.info(f"Resuming from checkpoint: {latest}")
        start_step = load_checkpoint(model, optimizer, scaler, latest, ddp)
        if master:
            logger.success(f"Resumed at step {start_step}")

    # ------------------------------------------------------------------
    # Dataset + DataLoader
    # ------------------------------------------------------------------
    dataset = FineWebEduDataset(encoding, seq_len, dataset_subset, rank, world_size)
    loader = DataLoader(
        dataset,
        batch_size=micro_batch,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    if master:
        os.makedirs(ckpt_dir, exist_ok=True)

    model.train()
    data_iter = iter(loader)
    t0 = time.perf_counter()
    step = start_step
    target_device = f"cuda:{local_rank}" if ddp else device
    skipped_steps = 0

    try:
        while step < total_steps:
            if ShutdownFlag.requested:
                if master:
                    logger.warning(
                        f"Shutdown requested at step {step}, breaking loop"
                    )
                break

            cur_lr = get_lr(step, warmup_steps, total_steps, lr, lr * 0.1)
            for g in optimizer.param_groups:
                g["lr"] = cur_lr

            optimizer.zero_grad(set_to_none=True)

            # Accumulate micro-batch loss on-device; single .item() at the
            # end of the accumulation window keeps the GPU fully async.
            loss_accum = torch.zeros((), device=target_device, dtype=torch.float32)
            bad_microsteps = 0

            for micro_step in range(grad_accum):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    x, y = next(data_iter)

                x = x.to(target_device, non_blocking=True)
                y = y.to(target_device, non_blocking=True)

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

                # NaN / Inf guard: a single corrupt sample (or a spike in
                # the MoE routing) can produce a non-finite loss. Propagating
                # it into backward() poisons every param's grad and — worse
                # — the Adam second-moment buffers, which is unrecoverable
                # without rewinding to a prior checkpoint. Detect it here
                # and skip the backward entirely.
                if not torch.isfinite(loss):
                    bad_microsteps += 1
                    logger.warning(
                        f"non-finite loss at step {step}.{micro_step} "
                        f"(value={loss.item()}) — skipping backward for this microstep"
                    )
                    continue

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss_accum = loss_accum + loss.detach()

            # If every microstep was non-finite, skip the optimizer step
            # entirely — grads are either empty (set_to_none=True above) or
            # untouched, and advancing Adam on them would taint the moment
            # buffers. LR/step still tick so the schedule stays monotonic.
            if bad_microsteps == grad_accum:
                skipped_steps += 1
                step += 1
                if master:
                    logger.error(
                        f"step {step}: ALL {grad_accum} microsteps non-finite — "
                        f"skipping optimizer.step() "
                        f"(total skipped: {skipped_steps})"
                    )
                continue

            # FSDP shards parameters, so `nn.utils.clip_grad_norm_` would clip
            # against each rank's local norm and miss the cross-shard gather.
            # FSDP.clip_grad_norm_ computes the true global norm and returns it.
            if scaler is not None:
                # Unscale in-place so clip_grad_norm_ sees true-magnitude grads.
                scaler.unscale_(optimizer)

            if ddp:
                grad_norm = model.clip_grad_norm_(grad_clip)
            else:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            # One more non-finite check, this time on the global grad norm —
            # an Inf/NaN here means clip didn't rescue us and stepping would
            # taint Adam state. ShardedGradScaler.step() handles this for the
            # fp16 path, but we enforce it uniformly.
            if not torch.isfinite(grad_norm):
                skipped_steps += 1
                step += 1
                if master:
                    logger.error(
                        f"step {step}: non-finite grad_norm={float(grad_norm)} — "
                        f"skipping optimizer.step() (total skipped: {skipped_steps})"
                    )
                if scaler is not None:
                    scaler.update()  # keep the scaler bookkeeping consistent
                continue

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # Drive the aux-loss-free router bias update. Without this call,
            # the router_bias buffer stays at zeros forever and the balancing
            # mechanism is inert — a silent correctness bug. `ddp=ddp` asks
            # the OpenMythos wrapper to all-reduce expert_load across ranks
            # before each local update.
            base_model = model.module if ddp and hasattr(model, "module") else model
            if hasattr(base_model, "update_router_biases"):
                base_model.update_router_biases(ddp=ddp)

            step += 1

            if master and step % log_every == 0:
                dt = time.perf_counter() - t0
                tok_per_sec = global_batch_tok * log_every / max(dt, 1e-9)
                tokens_seen = step * global_batch_tok
                loss_val = float(loss_accum.detach().item())
                logger.info(
                    f"step {step:6d}/{total_steps} | loss {loss_val:.4f} "
                    f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e} "
                    f"| {tok_per_sec / 1e6:.2f}M tok/s "
                    f"| {tokens_seen / 1e9:.1f}B tokens seen "
                    f"| skipped={skipped_steps}"
                )
                t0 = time.perf_counter()

            if step % ckpt_every == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scaler,
                    step,
                    cfg,
                    vocab_size,
                    seed,
                    ckpt_dir,
                    ddp,
                    master,
                )
    except Exception:
        # Log the traceback from every rank so a crash from one rank is
        # recoverable post-mortem; then re-raise so the exit code is non-zero.
        logger.exception(f"Training crashed at step {step}")
        raise
    finally:
        # Final save covers two cases: normal completion when
        # `step` isn't aligned to ckpt_every, *and* SIGTERM-driven early
        # exit. Either way, we want the most recent state on disk.
        try:
            if step > start_step and (
                step % ckpt_every != 0 or ShutdownFlag.requested
            ):
                save_checkpoint(
                    model,
                    optimizer,
                    scaler,
                    step,
                    cfg,
                    vocab_size,
                    seed,
                    ckpt_dir,
                    ddp,
                    master,
                )
        except Exception:
            logger.exception("Final checkpoint save failed")

        if ddp:
            # Barrier so no rank exits while another is still finishing its
            # checkpoint gather — avoids NCCL "process group destroyed" noise.
            try:
                dist.barrier()
            finally:
                dist.destroy_process_group()

    if master:
        if ShutdownFlag.requested:
            logger.warning(f"Training stopped early at step {step} by signal")
        else:
            logger.success(f"Training complete at step {step}")

    # Non-zero exit on SIGTERM so a supervisor (k8s, slurm) sees the job
    # as interrupted rather than successfully completed.
    if ShutdownFlag.requested:
        sys.exit(130)


if __name__ == "__main__":
    main()
