"""Validation loop + periodic eval hook.

Stage 4.5 addition. Held-out loss on a small fixed slice of FineWeb-Edu
provides a cheap signal for early-stopping / overfit detection without
requiring a separate eval pipeline. The hook is a no-op on non-master
ranks so grad-collection isn't perturbed by rank-0-only work.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from training.data import FineWebEduDataset, _seed_worker


def build_validation_loader(
    encoding,
    rank: int,
    world_size: int,
    seq_len: int,
    subset: str = "sample-10BT",
    micro_batch: int = 4,
    num_workers: int = 2,
) -> DataLoader:
    """DataLoader over a streamed slice of FineWeb-Edu for held-out eval.

    FineWebEduDataset shards deterministically on ``(rank, worker_id)``, so
    running validation on a different ``(rank, world_size)`` pair than training
    would serve a disjoint shard. We keep the shard identical to training for
    a smoke-level eval — the goal is a cheap signal on model progress, not a
    rigorous held-out partition. Upgrading to a true held-out split belongs in
    a future revision where the dataset class itself learns about splits.

    Args:
        encoding    -- tokenizer exposing ``.encode``
        rank        -- current distributed rank (used for sharding)
        world_size  -- total distributed ranks
        seq_len     -- context length for validation examples
        subset      -- FineWeb-Edu config name (same default as training)
        micro_batch -- per-step validation batch size
        num_workers -- DataLoader workers; kept low since eval is infrequent

    Returns:
        DataLoader iterating over ``(input_ids, target_ids)`` pairs.
    """
    dataset = FineWebEduDataset(encoding, seq_len, subset, rank, world_size)
    return DataLoader(
        dataset,
        batch_size=micro_batch,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
    )


def compute_val_loss(
    model,
    loader: DataLoader,
    device: str | torch.device,
    vocab_size: int,
    num_batches: int = 10,
) -> float:
    """Run ``num_batches`` forward passes on ``loader``; return mean CE loss.

    Leaves ``model.training`` unchanged on exit so this can be called mid-loop
    without surprising the training code.
    """
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= num_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size), y.reshape(-1)
            )
            total += float(loss.item())
            count += 1
    if was_training:
        model.train()
    return total / max(count, 1)


class PeriodicEvalHook:
    """Invoke validation loss every ``every_n_steps`` steps (master rank only).

    Designed to plug into ``Trainer(eval_hook=...)``: called as
    ``hook(step, trainer)`` after each optimizer step, short-circuits on
    non-schedule steps so the hot path has minimal overhead.
    """

    def __init__(
        self,
        every_n_steps: int,
        val_loader: DataLoader,
        device: str | torch.device,
        vocab_size: int,
        num_batches: int = 10,
        logger_obj=None,
    ) -> None:
        self.every_n_steps = every_n_steps
        self.val_loader = val_loader
        self.device = device
        self.vocab_size = vocab_size
        self.num_batches = num_batches
        self.logger_obj = logger_obj

    def __call__(self, step: int, trainer) -> None:
        if step == 0 or step % self.every_n_steps != 0:
            return
        if not getattr(trainer, "master", True):
            return
        val_loss = compute_val_loss(
            trainer.model,
            self.val_loader,
            self.device,
            self.vocab_size,
            num_batches=self.num_batches,
        )
        msg = f"[eval] step={step} val_loss={val_loss:.4f}"
        if self.logger_obj is not None:
            self.logger_obj.info(msg)
        else:
            print(msg)
