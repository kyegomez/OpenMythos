"""OpenMythos Training Utilities (100x Enhanced Edition).

Provides a complete, production-ready training loop with:
  - Mixed-precision training (bf16/fp16/fp32)
  - Cosine LR schedule with linear warmup
  - Gradient clipping and accumulation
  - Checkpoint save/resume
  - WandB / TensorBoard logging
  - Distributed training (DDP) support
  - Online loss tracking and progress display
  - Dataset iterator helpers
"""
from __future__ import annotations

import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

try:
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    _HAS_DIST = True
except ImportError:
    _HAS_DIST = False


# ---------------------------------------------------------------------------
# Training Config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """
    Full training configuration for OpenMythos.

    Optimizer:
        lr              -- peak learning rate
        weight_decay    -- AdamW weight decay
        beta1, beta2    -- AdamW betas
        eps             -- AdamW epsilon
        grad_clip       -- max gradient norm (0 = disabled)

    Schedule:
        warmup_steps    -- linear LR warmup steps
        total_steps     -- total training steps
        lr_min_ratio    -- final LR = lr * lr_min_ratio (cosine decay floor)

    Batching:
        batch_size      -- tokens per step (effective = batch_size * grad_accum)
        grad_accum      -- gradient accumulation steps
        seq_len         -- sequence length

    Precision:
        dtype           -- "bf16" | "fp16" | "fp32"

    Checkpointing:
        save_dir        -- directory for checkpoints
        save_every      -- save every N steps
        keep_last       -- keep only the last N checkpoints
        resume_from     -- checkpoint path to resume from

    Logging:
        log_every       -- log loss every N steps
        use_wandb       -- enable Weights & Biases logging
        wandb_project   -- W&B project name
        use_tensorboard -- enable TensorBoard logging
        log_dir         -- TensorBoard log directory
    """
    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 2000
    total_steps: int = 100_000
    lr_min_ratio: float = 0.1

    # Batching
    batch_size: int = 32
    grad_accum: int = 1
    seq_len: int = 2048

    # Precision
    dtype: str = "bf16"  # "bf16" | "fp16" | "fp32"

    # Checkpointing
    save_dir: str = "checkpoints"
    save_every: int = 1000
    keep_last: int = 3
    resume_from: Optional[str] = None

    # Logging
    log_every: int = 10
    use_wandb: bool = False
    wandb_project: str = "open-mythos"
    use_tensorboard: bool = False
    log_dir: str = "runs"

    def get_torch_dtype(self) -> torch.dtype:
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.dtype]


# ---------------------------------------------------------------------------
# LR Schedule
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_ratio: float = 0.1,
) -> LambdaLR:
    """
    Cosine LR schedule with linear warmup.

    Peak LR is reached at warmup_steps, then decays to min_ratio * peak_lr
    following a cosine curve. This matches GPT-3 / LLaMA training recipes.

    Args:
        optimizer    -- the optimizer to schedule
        warmup_steps -- number of linear warmup steps
        total_steps  -- total training steps
        min_ratio    -- final LR / peak LR ratio

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Mixed-precision context
# ---------------------------------------------------------------------------

@contextmanager
def autocast_ctx(dtype: str, device: torch.device):
    """
    Context manager for mixed-precision training.

    Args:
        dtype  -- "bf16", "fp16", or "fp32"
        device -- torch device (autocast only active on CUDA)
    """
    if dtype == "fp32" or device.type != "cuda":
        yield
    else:
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        with torch.autocast(device_type="cuda", dtype=torch_dtype):
            yield


# ---------------------------------------------------------------------------
# Gradient scaler factory
# ---------------------------------------------------------------------------

def make_scaler(dtype: str) -> Optional[torch.cuda.amp.GradScaler]:
    """Return a GradScaler for fp16 training, or None for bf16/fp32."""
    if dtype == "fp16" and torch.cuda.is_available():
        return torch.cuda.amp.GradScaler()
    return None


# ---------------------------------------------------------------------------
# Optimizer builder
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> AdamW:
    """
    Build AdamW optimizer with weight decay applied only to weight matrices
    (not biases, norms, or embeddings) — following Chinchilla / LLaMA recipes.

    Args:
        model -- the model to optimize
        cfg   -- TrainingConfig

    Returns:
        Configured AdamW optimizer
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No decay for: biases, norm weights, embeddings, 1D params
        if param.ndim < 2 or any(k in name for k in ("bias", "norm", "embed", "ln")):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(
        param_groups,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        fused=torch.cuda.is_available(),  # faster fused kernel when available
    )


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Manages saving and loading of training checkpoints.

    Keeps the last `keep_last` checkpoints and tracks training state
    including step count, optimizer state, and scheduler state.
    """

    def __init__(self, save_dir: Union[str, Path], keep_last: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self._saved: List[Path] = []

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[LambdaLR] = None,
        scaler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """
        Save a full training checkpoint.

        Args:
            step      -- current training step
            model     -- model (handles DDP wrapper transparently)
            optimizer -- optimizer state
            scheduler -- LR scheduler state
            scaler    -- GradScaler state (fp16 only)
            metrics   -- optional dict of scalar metrics to include

        Returns:
            Path to the saved checkpoint
        """
        # Unwrap DDP if needed
        raw_model = model.module if hasattr(model, "module") else model

        ckpt = {
            "step": step,
            "model_state_dict": raw_model.state_dict(),
            "config": raw_model.cfg.__dict__ if hasattr(raw_model, "cfg") else {},
            "optimizer_state_dict": optimizer.state_dict(),
            "version": "1.0.0-enhanced",
        }
        if scheduler is not None:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None:
            ckpt["scaler_state_dict"] = scaler.state_dict()
        if metrics:
            ckpt["metrics"] = metrics

        path = self.save_dir / f"step_{step:08d}.pt"
        torch.save(ckpt, path)
        self._saved.append(path)
        print(f"[Checkpoint] Saved → {path} ({path.stat().st_size / 1e6:.1f} MB)")

        # Evict old checkpoints
        while len(self._saved) > self.keep_last:
            old = self._saved.pop(0)
            if old.exists():
                old.unlink()
                print(f"[Checkpoint] Evicted {old.name}")

        return path

    def load(
        self,
        path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[LambdaLR] = None,
        scaler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> int:
        """
        Load a checkpoint and restore model/optimizer/scheduler state.

        Args:
            path      -- checkpoint file path
            model     -- model to restore into
            optimizer -- optimizer to restore (optional)
            scheduler -- LR scheduler to restore (optional)
            scaler    -- GradScaler to restore (optional)
            device    -- target device

        Returns:
            Training step at which checkpoint was saved
        """
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        step = ckpt.get("step", 0)
        print(f"[Checkpoint] Resumed from step {step} ← {path}")
        return step

    def latest(self) -> Optional[Path]:
        """Return the path of the most recently saved checkpoint, if any."""
        candidates = sorted(self.save_dir.glob("step_*.pt"))
        return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Metrics tracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    """Rolling average metrics tracker for training loss and perplexity."""

    def __init__(self, window: int = 100):
        self.window = window
        self._data: Dict[str, List[float]] = {}

    def update(self, **kwargs: float) -> None:
        """Add scalar metric values."""
        for k, v in kwargs.items():
            if k not in self._data:
                self._data[k] = []
            self._data[k].append(float(v))
            if len(self._data[k]) > self.window:
                self._data[k].pop(0)

    def mean(self, key: str) -> float:
        """Return rolling mean of a metric."""
        vals = self._data.get(key, [])
        return sum(vals) / len(vals) if vals else 0.0

    def last(self, key: str) -> float:
        """Return the last recorded value of a metric."""
        vals = self._data.get(key, [])
        return vals[-1] if vals else 0.0

    def summary(self) -> Dict[str, float]:
        """Return a dict of rolling means for all tracked metrics."""
        return {k: self.mean(k) for k in self._data}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Production-ready trainer for OpenMythos.

    Features:
      - Mixed-precision training (bf16/fp16/fp32)
      - Cosine LR with warmup
      - Gradient accumulation and clipping
      - Automatic checkpoint save/resume
      - WandB + TensorBoard integration
      - DDP-aware (rank-0 only logging/saving)

    Usage::

        from open_mythos import OpenMythos, MythosConfig
        from open_mythos.training import Trainer, TrainingConfig

        model = OpenMythos(MythosConfig())
        train_cfg = TrainingConfig(lr=3e-4, total_steps=10000)
        trainer = Trainer(model, train_cfg)
        trainer.fit(data_iterator)  # yields (input_ids, labels) batches
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_main = not _HAS_DIST or not dist.is_initialized() or dist.get_rank() == 0

        # Move model
        self.model = model.to(self.device)

        # Wrap in DDP if distributed
        if _HAS_DIST and dist.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.model = DDP(self.model, device_ids=[local_rank])

        # Optimizer, scheduler, scaler
        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, cfg.warmup_steps, cfg.total_steps, cfg.lr_min_ratio
        )
        self.scaler = make_scaler(cfg.dtype)

        # Checkpoint manager
        self.ckpt_mgr = CheckpointManager(cfg.save_dir, cfg.keep_last)

        # Metrics
        self.metrics = MetricsTracker(window=100)
        self.step = 0

        # Optional logging backends
        self._wandb = None
        self._tb_writer = None
        if self.is_main:
            if cfg.use_wandb:
                try:
                    import wandb
                    self._wandb = wandb
                    wandb.init(project=cfg.wandb_project, config=cfg.__dict__)
                except ImportError:
                    print("[Trainer] wandb not installed — skipping W&B logging")
            if cfg.use_tensorboard:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    self._tb_writer = SummaryWriter(log_dir=cfg.log_dir)
                except ImportError:
                    print("[Trainer] tensorboard not installed — skipping TB logging")

        # Resume if requested
        if cfg.resume_from:
            self.step = self.ckpt_mgr.load(
                cfg.resume_from, self.model, self.optimizer, self.scheduler, self.scaler, self.device
            )
        elif (latest := self.ckpt_mgr.latest()) is not None:
            print(f"[Trainer] Auto-resuming from {latest}")
            self.step = self.ckpt_mgr.load(
                latest, self.model, self.optimizer, self.scheduler, self.scaler, self.device
            )

    def fit(
        self,
        data_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]],
        eval_fn: Optional[Callable[[], Dict[str, float]]] = None,
        eval_every: int = 500,
    ) -> None:
        """
        Run the full training loop.

        Args:
            data_iter  -- iterator yielding (input_ids, labels) tensors
            eval_fn    -- optional callable returning eval metrics dict
            eval_every -- run eval_fn every N steps
        """
        self.model.train()
        accum_loss = 0.0
        t0 = time.time()

        self.optimizer.zero_grad(set_to_none=True)

        while self.step < self.cfg.total_steps:
            for micro_step in range(self.cfg.grad_accum):
                try:
                    input_ids, labels = next(data_iter)
                except StopIteration:
                    print("[Trainer] Data iterator exhausted — stopping.")
                    return

                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                with autocast_ctx(self.cfg.dtype, self.device):
                    _, loss = self.model(input_ids, labels=labels)
                    loss = loss / self.cfg.grad_accum

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_loss += loss.item()

            # Gradient step
            if self.scaler is not None:
                if self.cfg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.step += 1
            self.metrics.update(loss=accum_loss, lr=self.scheduler.get_last_lr()[0])
            accum_loss = 0.0

            # Logging
            if self.is_main and self.step % self.cfg.log_every == 0:
                elapsed = time.time() - t0
                avg_loss = self.metrics.mean("loss")
                ppl = math.exp(min(avg_loss, 20))
                lr_now = self.scheduler.get_last_lr()[0]
                print(
                    f"step {self.step:>8d}/{self.cfg.total_steps} "
                    f"| loss {avg_loss:.4f} | ppl {ppl:.1f} "
                    f"| lr {lr_now:.2e} | {elapsed:.1f}s"
                )
                t0 = time.time()
                self._log_metrics({"train/loss": avg_loss, "train/ppl": ppl, "train/lr": lr_now})

            # Eval
            if eval_fn is not None and self.step % eval_every == 0:
                eval_metrics = eval_fn()
                if self.is_main:
                    self._log_metrics({f"eval/{k}": v for k, v in eval_metrics.items()})
                    print(f"[Eval step {self.step}] " + " | ".join(f"{k}={v:.4f}" for k, v in eval_metrics.items()))
                self.model.train()

            # Checkpoint
            if self.is_main and self.step % self.cfg.save_every == 0:
                self.ckpt_mgr.save(
                    self.step, self.model, self.optimizer, self.scheduler, self.scaler,
                    metrics=self.metrics.summary()
                )

        # Final checkpoint
        if self.is_main:
            self.ckpt_mgr.save(
                self.step, self.model, self.optimizer, self.scheduler, self.scaler,
                metrics=self.metrics.summary()
            )
        print(f"[Trainer] Training complete at step {self.step}.")

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to W&B and/or TensorBoard."""
        if self._wandb is not None:
            self._wandb.log(metrics, step=self.step)
        if self._tb_writer is not None:
            for k, v in metrics.items():
                self._tb_writer.add_scalar(k, v, self.step)

    def close(self) -> None:
        """Clean up logging resources."""
        if self._wandb is not None:
            self._wandb.finish()
        if self._tb_writer is not None:
            self._tb_writer.close()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def simple_token_iterator(
    token_ids: torch.Tensor,
    seq_len: int,
    batch_size: int,
    device: Optional[torch.device] = None,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Infinite iterator over a flat token tensor, yielding (input, label) pairs.

    Tiles the token array to produce infinite batches for training.
    Labels are input_ids shifted left by 1 with the last position set to -100.

    Args:
        token_ids  -- flat 1-D tensor of token IDs
        seq_len    -- sequence length per sample
        batch_size -- number of sequences per batch
        device     -- target device

    Yields:
        (input_ids, labels) each of shape (batch_size, seq_len)
    """
    n = len(token_ids)
    stride = seq_len * batch_size
    pos = 0

    while True:
        # Ensure we have enough tokens
        if pos + stride + 1 > n:
            pos = 0

        chunk = token_ids[pos: pos + stride + 1]
        pos += stride

        x = chunk[:-1].view(batch_size, seq_len)
        y = chunk[1:].view(batch_size, seq_len)

        if device is not None:
            x, y = x.to(device), y.to(device)

        yield x, y


def compute_perplexity(
    model: nn.Module,
    token_ids: torch.Tensor,
    seq_len: int,
    batch_size: int = 4,
    device: Optional[torch.device] = None,
    n_loops: int = 8,
) -> float:
    """
    Compute perplexity of a model on a flat token tensor.

    Args:
        model      -- OpenMythos model
        token_ids  -- flat 1-D tensor of evaluation tokens
        seq_len    -- sequence length for chunking
        batch_size -- batch size for evaluation
        device     -- target device
        n_loops    -- recurrent loop depth

    Returns:
        Perplexity (float)
    """
    model.eval()
    device = device or next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(token_ids) - seq_len, seq_len * batch_size):
            batch_tokens = []
            for j in range(batch_size):
                start = i + j * seq_len
                if start + seq_len + 1 > len(token_ids):
                    break
                batch_tokens.append(token_ids[start: start + seq_len + 1])
            if not batch_tokens:
                break

            batch = torch.stack(batch_tokens).to(device)
            x, y = batch[:, :-1], batch[:, 1:]
            _, loss = model(x, n_loops=n_loops, labels=y)
            n_toks = y.numel()
            total_loss += loss.item() * n_toks
            total_tokens += n_toks

    return math.exp(total_loss / max(total_tokens, 1))
