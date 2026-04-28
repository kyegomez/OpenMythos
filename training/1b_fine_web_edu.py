#!/usr/bin/env python3
"""
OpenMythos 1B pretraining on FineWeb-Edu with FSDP + AdamW + optional ClearML.

Supports both the original ACT recipe and the new stochastic-depth recipe
(Option B) via the `recurrent_mode` hyperparameter in main(). Checkpoints
are compatible across modes.

Single GPU:
    python training/1b_fine_web_edu.py

Multi-GPU:
    torchrun --nproc_per_node=N training/1b_fine_web_edu.py

Dataset: expects FineWeb-Edu parquet files at DATASET_PATH (see docs/datasets.md
for preparation instructions). Uses direct pyarrow parquet reading rather than
the HuggingFace `datasets` streaming iterator (~17,000x faster for local files).

Environment variables (optional):
    DATASET_PATH     -- local path to FineWeb-Edu parquet files (required)
    OUTPUT_DIR       -- checkpoint + log directory (default: ./output/experiments)
    TARGET_TOKENS    -- token budget in billions (default: 10)
    HF_TOKEN         -- HuggingFace token, for tokenizer download

ClearML tracking (optional — set all three to enable):
    CLEARML_API_HOST
    CLEARML_API_ACCESS_KEY
    CLEARML_API_SECRET_KEY
    CLEARML_PROJECT  -- ClearML project name (default: openmythos)
    EXPERIMENT_NAME  -- ClearML task name (default: 1b-fine-web-edu)
"""

import os
import math
import random
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from loguru import logger
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

import glob as _glob
import pyarrow.parquet as pq
from datasets import load_dataset

from open_mythos import OpenMythos
from open_mythos.main import TransformerBlock, RecurrentBlock
from open_mythos.variants import mythos_1b
from open_mythos.tokenizer import MythosTokenizer

# ---------------------------------------------------------------------------
# ClearML (lazy — only initialized on rank 0)
# ---------------------------------------------------------------------------

_clearml_task = None
_clearml_logger = None


def init_clearml(cfg, training_hparams: dict, timeout: int = 30):
    """Initialize ClearML tracking on rank 0. No-op if unreachable or missing."""
    global _clearml_task, _clearml_logger
    import signal

    def _timeout_handler(signum, frame):
        raise TimeoutError("ClearML init timed out")

    try:
        from clearml import Task

        project = os.environ.get("CLEARML_PROJECT", "openmythos")
        task_name = os.environ.get("EXPERIMENT_NAME", "1b-fine-web-edu")

        # Task.init can hang if the ClearML server is unreachable (e.g.,
        # the network is restricted). Use a SIGALRM timeout to fail fast.
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
        try:
            _clearml_task = Task.init(project_name=project, task_name=task_name)
            _clearml_task.connect(vars(cfg), name="model_config")
            _clearml_task.connect(training_hparams, name="training_hparams")
            _clearml_logger = _clearml_task.get_logger()
            logger.info(f"ClearML initialized: project={project}, task={task_name}")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except Exception as e:
        logger.warning(
            f"ClearML init failed (training continues without tracking): {e}"
        )


def log_clearml(series: str, value: float, step: int):
    """Report a scalar to ClearML if available."""
    if _clearml_logger is not None:
        _clearml_logger.report_scalar("train", series, iteration=step, value=value)


def log_clearml_text(title: str, text: str):
    """Log text to ClearML if available."""
    if _clearml_logger is not None:
        _clearml_logger.report_text(f"## {title}\n\n{text}")


def register_clearml_artifact(name: str, path: str):
    """Register a file artifact in ClearML if available."""
    if _clearml_task is not None:
        _clearml_task.upload_artifact(name, artifact_object=path)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FineWebEduDataset(IterableDataset):
    """
    FineWeb-Edu loader yielding fixed-length (input, target) pairs.

    Supports two modes:
      - Local parquet: loads from a directory of .parquet files (no internet needed)
      - Streaming: pulls shards on demand from HuggingFace (requires internet)

    Documents are concatenated into a rolling buffer and sliced into
    fixed-length chunks. Sharding is two-dimensional: world_size ranks x
    num_workers DataLoader workers per rank.
    """

    def __init__(
        self,
        encoding,
        seq_len: int,
        rank: int,
        world_size: int,
        dataset_path: str = "",
        dataset_subset: str = "sample-10BT",
    ):
        self.encoding = encoding
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.dataset_path = dataset_path
        self.dataset_subset = dataset_subset

    def _get_parquet_files(self, shard_index: int, total_shards: int) -> list[str]:
        """Return the subset of parquet files assigned to this shard."""
        all_files = sorted(_glob.glob(os.path.join(self.dataset_path, "*.parquet")))
        if not all_files:
            raise FileNotFoundError(f"No .parquet files found in {self.dataset_path}")
        return [f for i, f in enumerate(all_files) if i % total_shards == shard_index]

    def _iter_parquet(self, shard_index: int, total_shards: int):
        """Read local parquet files directly via pyarrow. Loops infinitely."""
        files = self._get_parquet_files(shard_index, total_shards)
        if not files:
            return

        buf: list[int] = []
        while True:
            for parquet_path in files:
                table = pq.read_table(parquet_path, columns=["text"])
                text_column = table.column("text")
                del table

                for text_value in text_column:
                    text = text_value.as_py()
                    if text:
                        buf.extend(self.encoding.encode(text))
                        while len(buf) >= self.seq_len + 1:
                            chunk = buf[: self.seq_len + 1]
                            buf = buf[self.seq_len + 1 :]
                            yield (
                                torch.tensor(chunk[:-1], dtype=torch.long),
                                torch.tensor(chunk[1:], dtype=torch.long),
                            )
                del text_column

    def _iter_streaming(self, shard_index: int, total_shards: int):
        """HuggingFace streaming fallback (requires internet)."""
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=self.dataset_subset,
            split="train",
            streaming=True,
        ).shard(num_shards=total_shards, index=shard_index)

        buf: list[int] = []
        for sample in ds:
            buf.extend(self.encoding.encode(sample["text"]))
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf = buf[self.seq_len + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )

    def __iter__(self):
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        total_shards = self.world_size * num_workers
        shard_index = self.rank * num_workers + worker_id

        if self.dataset_path:
            yield from self._iter_parquet(shard_index, total_shards)
        else:
            yield from self._iter_streaming(shard_index, total_shards)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup -> cosine decay
# ---------------------------------------------------------------------------


def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * step / warmup
    if step >= total:
        return min_lr
    decay = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _list_ckpts(ckpt_dir: str) -> list[str]:
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.startswith("step_") and f.endswith(".pt")
    )


def save_checkpoint(
    model,
    optimizer,
    step: int,
    cfg,
    vocab_size: int,
    ckpt_dir: str,
    ddp: bool,
    master: bool,
    keep_last: int = 3,
) -> None:
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
    final_path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
    tmp_path = final_path + ".tmp"
    torch.save(
        {
            "step": step,
            "model": model_state,
            "optimizer": optim_state,
            "cfg": cfg,
            "vocab_size": vocab_size,
        },
        tmp_path,
    )
    os.replace(tmp_path, final_path)

    for old in _list_ckpts(ckpt_dir)[:-keep_last]:
        try:
            os.remove(old)
        except OSError as exc:
            logger.warning(f"Failed to prune old checkpoint {old}: {exc}")

    logger.success(f"Checkpoint saved -> {final_path}")


def load_checkpoint(model, optimizer, path: str, ddp: bool) -> int:
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

    return int(ckpt["step"])


# ---------------------------------------------------------------------------
# Post-training generation test
# ---------------------------------------------------------------------------


GENERATION_PROMPTS = [
    "The purpose of education is",
    "In the beginning, there was",
    "The most important scientific discovery",
]


def run_generation_test(cfg, ckpt_dir, encoding, device: str):
    """
    Reconstruct a raw model from the latest checkpoint and generate text.

    Under FSDP, calling model.module.generate() while parameters are still
    sharded across ranks produces incorrect output or deadlocks.  Instead,
    this function loads the fully-gathered checkpoint (saved by rank 0) into
    a fresh, unwrapped model on a single GPU after the process group has
    been torn down.  Safe for both single-GPU and post-FSDP scenarios.
    """
    logger.info("Running post-training generation test...")

    ckpts = _list_ckpts(ckpt_dir)
    if not ckpts:
        logger.warning("No checkpoint found — skipping generation test.")
        return

    ckpt = torch.load(ckpts[-1], map_location=device, weights_only=False)
    raw_model = OpenMythos(cfg)
    raw_model.load_state_dict(ckpt["model"])
    raw_model = raw_model.to(device)
    raw_model.eval()

    results = []
    for prompt_text in GENERATION_PROMPTS:
        tokens = encoding.encode(prompt_text)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            output_ids = raw_model.generate(
                input_ids,
                max_new_tokens=128,
                temperature=0.8,
                top_k=40,
            )

        generated = encoding.decode(output_ids[0].tolist())
        result = f"**Prompt:** {prompt_text}\n**Generated:** {generated}\n"
        results.append(result)
        logger.info(f"\n{result}")

    all_results = "\n---\n".join(results)
    log_clearml_text("Generation Samples", all_results)


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
        logger.info(
            f"GPUs: {torch.cuda.device_count()}  |  World size: {world_size}  |  Device: {device}"
        )

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    encoding = MythosTokenizer()
    vocab_size = encoding.vocab_size

    if master:
        logger.info(f"Tokenizer: gpt-oss-20b  |  Vocab size: {vocab_size:,}")

    # ------------------------------------------------------------------
    # Hyperparameters (env-var configurable with defaults)
    # ------------------------------------------------------------------
    # Recurrent-depth training recipe (Option A: ACT, Option B: stochastic depth).
    # Change recurrent_mode to "act" to use the original ACT halting recipe.
    recurrent_mode = "stochastic_depth"  # "act" or "stochastic_depth"
    stochastic_depth_min = 1
    stochastic_depth_max = 32

    seq_len = 2048
    micro_batch = 1
    target_tokens_b = int(os.environ.get("TARGET_TOKENS", "10"))
    target_tokens = target_tokens_b * 1_000_000_000
    grad_accum = max(1, 16 // (world_size * micro_batch))
    global_batch_tok = world_size * micro_batch * grad_accum * seq_len
    total_steps = target_tokens // global_batch_tok
    warmup_steps = 2000
    lr = 3e-4
    min_lr = 3e-5
    wd = 0.1
    log_every = 1
    ckpt_every = 1000
    output_dir = os.environ.get(
        "OUTPUT_DIR", "./output/experiments"
    )
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    dataset_path = os.environ.get(
        "DATASET_PATH", "./data/fineweb-edu"
    )
    dataset_subset = "sample-10BT"

    training_hparams = {
        "seq_len": seq_len,
        "micro_batch": micro_batch,
        "target_tokens": target_tokens,
        "grad_accum": grad_accum,
        "global_batch_tok": global_batch_tok,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "lr": lr,
        "min_lr": min_lr,
        "weight_decay": wd,
        "log_every": log_every,
        "ckpt_every": ckpt_every,
        "output_dir": output_dir,
        "dataset_path": dataset_path,
        "dataset_subset": dataset_subset,
        "world_size": world_size,
        "recurrent_mode": recurrent_mode,
        "stochastic_depth_min": stochastic_depth_min,
        "stochastic_depth_max": stochastic_depth_max,
    }

    if master:
        logger.info(
            f"seq_len={seq_len} | micro_batch={micro_batch} | grad_accum={grad_accum} | "
            f"global_batch_tokens={global_batch_tok:,} | total_steps={total_steps:,} | "
            f"target_tokens={target_tokens_b}B"
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    cfg = mythos_1b()
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

    amp_ctx = nullcontext() if ddp else amp_ctx  # type: ignore[possibly-undefined]

    if master:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Parameters: {n_params:,}  |  AMP dtype: {amp_dtype}")

    if master:
        if recurrent_mode == "stochastic_depth":
            logger.info(
                f"Recurrent mode: stochastic_depth "
                f"(n_loops sampled uniformly from [{stochastic_depth_min}, {stochastic_depth_max}])"
            )
        else:
            logger.info(
                f"Recurrent mode: act (n_loops = cfg.max_loop_iters = {cfg.max_loop_iters})"
            )

    # ------------------------------------------------------------------
    # ClearML init (after model is built so we can log config)
    # ------------------------------------------------------------------
    if master:
        init_clearml(cfg, training_hparams)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
        betas=(0.9, 0.95),
        fused="cuda" in device,
    )

    # ------------------------------------------------------------------
    # Resume from latest checkpoint (if any)
    # ------------------------------------------------------------------
    start_step = 0
    existing_ckpts = _list_ckpts(ckpt_dir)
    if existing_ckpts:
        latest = existing_ckpts[-1]
        if master:
            logger.info(f"Resuming from checkpoint: {latest}")
        start_step = load_checkpoint(model, optimizer, latest, ddp)
        if master:
            logger.success(f"Resumed at step {start_step}")

    # ------------------------------------------------------------------
    # Dataset + DataLoader
    # ------------------------------------------------------------------
    dataset = FineWebEduDataset(
        encoding,
        seq_len,
        rank,
        world_size,
        dataset_path=dataset_path,
        dataset_subset=dataset_subset,
    )
    loader = DataLoader(dataset, batch_size=micro_batch, num_workers=4, pin_memory=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    if master:
        os.makedirs(ckpt_dir, exist_ok=True)

    model.train()
    data_iter = iter(loader)
    t0 = time.perf_counter()
    step = start_step

    while step < total_steps:
        cur_lr = get_lr(step, warmup_steps, total_steps, lr, min_lr)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad()
        loss_accum = 0.0

        # Sample n_loops once per optimizer step. With FSDP/DDP, all ranks must
        # run the same number of recurrent iterations to avoid all-gather
        # ordering mismatch (same bug class as the ACT early-exit deadlock in
        # commit 6c5659c). Broadcast from rank 0 so all ranks agree.
        if recurrent_mode == "stochastic_depth":
            if master:
                n_loops_this_step = random.randint(
                    stochastic_depth_min, stochastic_depth_max
                )
            else:
                n_loops_this_step = 0
            if ddp:
                nl_tensor = torch.tensor(
                    [n_loops_this_step], device=device, dtype=torch.int64
                )
                dist.broadcast(nl_tensor, src=0)
                n_loops_this_step = int(nl_tensor.item())
            bypass_act_this_step = True
        else:
            n_loops_this_step = None
            bypass_act_this_step = False

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
                logits = model(
                    x,
                    n_loops=n_loops_this_step,
                    bypass_act=bypass_act_this_step,
                )
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                )
                loss = loss / grad_accum

            loss.backward()
            loss_accum += loss.item()

        if ddp:
            grad_norm = model.clip_grad_norm_(1.0)
        else:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        step += 1

        if master and step % log_every == 0:
            dt = time.perf_counter() - t0
            tok_per_sec = global_batch_tok * log_every / dt
            tokens_seen = step * global_batch_tok

            n_loops_display = (
                n_loops_this_step
                if n_loops_this_step is not None
                else cfg.max_loop_iters
            )
            logger.info(
                f"step {step:6d}/{total_steps} | loss {loss_accum:.4f} "
                f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e} "
                f"| {tok_per_sec / 1e6:.2f}M tok/s "
                f"| {tokens_seen / 1e9:.1f}B tokens seen "
                f"| mode={recurrent_mode} n_loops={n_loops_display}"
            )

            log_clearml("loss", loss_accum, step)
            log_clearml("grad_norm", float(grad_norm), step)
            log_clearml("lr", cur_lr, step)
            log_clearml("throughput_mtok_s", tok_per_sec / 1e6, step)
            log_clearml("tokens_seen_B", tokens_seen / 1e9, step)
            log_clearml("n_loops", float(n_loops_display), step)

            t0 = time.perf_counter()

        if step % ckpt_every == 0:
            save_checkpoint(
                model, optimizer, step, cfg, vocab_size, ckpt_dir, ddp, master
            )

    # Final checkpoint
    if step > start_step and step % ckpt_every != 0:
        save_checkpoint(model, optimizer, step, cfg, vocab_size, ckpt_dir, ddp, master)

    # ------------------------------------------------------------------
    # Tear down distributed process group before generation
    # ------------------------------------------------------------------
    if ddp:
        dist.barrier()
        dist.destroy_process_group()

    # ------------------------------------------------------------------
    # Post-training generation test (rank 0 only)
    # ------------------------------------------------------------------
    # Reconstruct a fresh model from the checkpoint so we don't need
    # FSDP — the process group is already torn down at this point.
    if master:
        gen_device = device if not ddp else f"cuda:{local_rank}"
        run_generation_test(cfg, ckpt_dir, encoding, gen_device)

    if master:
        logger.success("Training complete.")


if __name__ == "__main__":
    main()
