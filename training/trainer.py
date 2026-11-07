"""Training-loop orchestration for OpenMythos pretraining.

Extracted from training/3b_fine_web_edu.py at Stage 4.5. The ``Trainer``
class wraps the monolithic main-loop body so the entry point can stay
thin and an eval hook can be injected without mutating the loop itself.

Checkpoint I/O lives in ``training.checkpoint``; Trainer.save/load
delegate there. Keeping them separate lets this module stay under the
400-line project ceiling.
"""
from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext
from typing import Callable, Optional

import torch.distributed as dist
import torch.nn as nn
from loguru import logger

from training.checkpoint import _list_ckpts, load_checkpoint, save_checkpoint
from training.constants import USE_GRAD_CHECKPOINT
from training.schedule import get_lr

# _list_ckpts re-exported so callers doing `from training.trainer import ...`
# still resolve — preserves run_3b.py's single-line import (≤60 line budget).
__all__ = [
    "Trainer",
    "_list_ckpts",
    "load_checkpoint",
    "save_checkpoint",
    "apply_grad_checkpoint_if_enabled",
]


# ---------------------------------------------------------------------------
# Gradient checkpointing wrap (P3.8 completion)
# ---------------------------------------------------------------------------


def apply_grad_checkpoint_if_enabled(model: nn.Module) -> int:
    """Wrap every ``RecurrentBlock.forward`` in activation checkpointing.

    No-op when ``training.constants.USE_GRAD_CHECKPOINT`` is False (default),
    preserving the exact pre-P3.8 forward. When True, replaces each
    ``RecurrentBlock``'s forward with a ``torch.utils.checkpoint.checkpoint``
    call using ``use_reentrant=False`` (required for correctness under FSDP
    and gradient accumulation; the reentrant variant silently loses grads on
    non-root autograd paths).

    Must be called BEFORE FSDP wrapping so the checkpointed forward is the
    unit that FSDP sees. Applying after FSDP wrap would checkpoint the
    FlatParam view and break the FSDP collective.

    Args:
        model -- raw (un-wrapped) ``OpenMythos`` or any ``nn.Module`` tree
                 containing ``RecurrentBlock`` instances. Mutated in place.

    Returns:
        Number of ``RecurrentBlock`` modules whose forward was replaced;
        0 when the flag is off or no such modules exist.
    """
    if not USE_GRAD_CHECKPOINT:
        return 0

    # Lazy import: avoids pulling the recurrent block in for users who only
    # build a bare model without ever invoking this helper.
    from open_mythos.recurrence.block import RecurrentBlock
    from torch.utils.checkpoint import checkpoint

    def _make_wrapped(orig_forward):
        def wrapped(*args, **kwargs):
            # use_reentrant=False is mandatory — FSDP + reentrant checkpoint
            # silently drops the backward pass for fan-in RecurrentBlock uses
            # such as the ACT residual add.
            return checkpoint(orig_forward, *args, use_reentrant=False, **kwargs)

        return wrapped

    wrapped_count = 0
    for module in model.modules():
        if isinstance(module, RecurrentBlock):
            # Intentional instance-level forward shadow to wrap in checkpoint.
            # mypy rejects method assignment in general; the runtime behaviour
            # is well-defined and required by the activation-checkpointing
            # pattern.
            module.forward = _make_wrapped(module.forward)  # type: ignore[method-assign]
            wrapped_count += 1

    if wrapped_count:
        logger.info(
            f"Gradient checkpointing enabled on {wrapped_count} RecurrentBlock(s)"
        )
    return wrapped_count


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Encapsulates the FSDP training loop + checkpoint scheduling.

    Stage 4.5. Moved from training/3b_fine_web_edu.py::main() body. The class
    owns the stateful handles (model, optimizer, dataloader, AMP context) and
    exposes ``run``, ``save``, ``load`` as the public surface. An optional
    ``eval_hook`` callable is invoked as ``eval_hook(step, self)`` at each
    optimizer step so validation can be wired in without touching the loop.
    """

    def __init__(
        self,
        cfg,
        model,
        optimizer,
        dataloader,
        *,
        device: str,
        local_rank: int,
        rank: int,
        world_size: int,
        ddp: bool,
        master: bool,
        amp_ctx,
        vocab_size: int,
        ckpt_dir: str,
        total_steps: int,
        log_every: int,
        ckpt_every: int,
        grad_accum_steps: int,
        warmup_steps: int,
        max_lr: float,
        min_lr_ratio: float = 0.1,
        grad_clip: float = 1.0,
        global_batch_tok: int | None = None,
        eval_hook: Optional[Callable[[int, "Trainer"], None]] = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.ddp = ddp
        self.master = master
        self.amp_ctx = amp_ctx
        self.vocab_size = vocab_size
        self.ckpt_dir = ckpt_dir
        self.total_steps = total_steps
        self.log_every = log_every
        self.ckpt_every = ckpt_every
        self.grad_accum_steps = grad_accum_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr_ratio = min_lr_ratio
        self.grad_clip = grad_clip
        self.global_batch_tok = global_batch_tok
        self.eval_hook = eval_hook

    # -- checkpoint helpers ------------------------------------------------

    def save(self, step: int) -> None:
        """Persist current model + optimizer state; master-only disk write."""
        save_checkpoint(
            self.model,
            self.optimizer,
            step,
            self.cfg,
            self.vocab_size,
            self.ckpt_dir,
            self.ddp,
            self.master,
        )

    def load(self, path: str) -> int:
        """Restore from ``path`` into this trainer's model/optimizer in-place."""
        return load_checkpoint(
            self.model, self.optimizer, path, self.ddp, cfg=self.cfg
        )

    # -- main loop ---------------------------------------------------------

    def run(self, start_step: int = 0) -> None:
        """Run the training loop from ``start_step`` to ``self.total_steps``.

        Body copied from 3b_fine_web_edu.py::main(). References to local
        variables are swapped for ``self.xxx``; the optional eval hook is
        invoked after each optimizer step so held-out loss can be computed
        at a fixed cadence without perturbing grad-accum timing.
        """
        if self.master:
            os.makedirs(self.ckpt_dir, exist_ok=True)

        self.model.train()
        data_iter = iter(self.dataloader)
        t0 = time.perf_counter()
        step = start_step

        min_lr = self.max_lr * self.min_lr_ratio
        target_device = (
            self.device if not self.ddp else f"cuda:{self.local_rank}"
        )

        while step < self.total_steps:
            cur_lr = get_lr(
                step,
                self.warmup_steps,
                self.total_steps,
                self.max_lr,
                min_lr,
            )
            for g in self.optimizer.param_groups:
                g["lr"] = cur_lr

            self.optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0

            for micro_step in range(self.grad_accum_steps):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    x, y = next(data_iter)

                x = x.to(target_device, non_blocking=True)
                y = y.to(target_device, non_blocking=True)

                sync = (
                    nullcontext()
                    if (not self.ddp or micro_step == self.grad_accum_steps - 1)
                    else self.model.no_sync()
                )
                with sync, self.amp_ctx:
                    # P3.8 complete: gradient checkpointing is applied at
                    # model-construction time via
                    # ``apply_grad_checkpoint_if_enabled`` (see trainer module
                    # top); no per-step wrap needed here.
                    logits = self.model(x)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, self.vocab_size), y.view(-1)
                    )
                    loss = loss / self.grad_accum_steps

                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    raise RuntimeError(
                        f"Non-finite loss at step {step}: {loss_value}"
                    )

                loss.backward()
                loss_accum += loss_value

            # FSDP shards parameters, so `nn.utils.clip_grad_norm_` would clip
            # against each rank's local norm and miss the cross-shard gather.
            # FSDP.clip_grad_norm_ computes the true global norm and returns it.
            if self.ddp:
                grad_norm = self.model.clip_grad_norm_(self.grad_clip)
            else:
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip
                )
            self.optimizer.step()
            step += 1

            if self.master and step % self.log_every == 0:
                dt = time.perf_counter() - t0
                if self.global_batch_tok is not None:
                    tok_per_sec = self.global_batch_tok * self.log_every / dt
                    tokens_seen = step * self.global_batch_tok
                    logger.info(
                        f"step {step:6d}/{self.total_steps} | loss {loss_accum:.4f} "
                        f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e} "
                        f"| {tok_per_sec / 1e6:.2f}M tok/s "
                        f"| {tokens_seen / 1e9:.1f}B tokens seen"
                    )
                else:
                    logger.info(
                        f"step {step:6d}/{self.total_steps} | loss {loss_accum:.4f} "
                        f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e}"
                    )
                t0 = time.perf_counter()

            if step % self.ckpt_every == 0:
                self.save(step)

            if self.eval_hook is not None:
                self.eval_hook(step, self)

        # Final checkpoint — total_steps may not be divisible by ckpt_every,
        # so without this the tail of the run is lost if the schedule doesn't
        # align.
        if step > start_step and step % self.ckpt_every != 0:
            self.save(step)

        if self.ddp:
            # Barrier so no rank exits while another is still finishing its
            # checkpoint gather — avoids NCCL "process group destroyed" noise.
            dist.barrier()
            dist.destroy_process_group()

        if self.master:
            logger.success("Training complete.")
