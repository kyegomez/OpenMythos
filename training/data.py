"""Dataset + DataLoader machinery for OpenMythos pretraining.

Extracted from training/3b_fine_web_edu.py at Stage 4.5. Exposes the
streaming FineWeb-Edu loader, the worker-seeding helper used for
determinism, the vocab-size compatibility check, and a new
``build_dataloader`` factory that encapsulates the DataLoader
construction ``main()`` previously did inline.
"""
from __future__ import annotations

import random
from collections import deque

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from datasets import load_dataset


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

    Streaming datasets are not seekable, so a resumed run re-enters its shard
    from the beginning. Acceptable at pretraining scale: the chance of
    re-playing the same tokens before the run ends is negligible versus the
    cost of a true resumable loader.
    """

    def __init__(self, encoding, seq_len: int, subset: str, rank: int, world_size: int):
        """
        Args:
            encoding   -- tokenizer exposing `.encode(str) -> list[int]`
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
        Yield `(input_ids, target_ids)` tensors of length `seq_len` forever.

        Inputs and targets are shifted by one for next-token prediction —
        `target[i] == input[i + 1]`. Documents are concatenated into a rolling
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

        # deque gives O(1) popleft per token vs. O(n) for list-slice reassignment
        buf: deque[int] = deque()
        chunk_len = self.seq_len + 1
        for sample in ds:
            buf.extend(self.encoding.encode(sample["text"]))
            while len(buf) >= chunk_len:
                chunk = [buf.popleft() for _ in range(chunk_len)]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def _seed_worker(worker_id: int) -> None:
    """
    Seed every DataLoader worker deterministically from the run's base seed.

    Each worker derives its own stream from `base_seed + worker_id`, giving
    reproducible shuffling across runs while keeping workers decorrelated
    from each other.
    """
    base_seed = torch.initial_seed() % (2**32)
    worker_seed = (base_seed + worker_id) % (2**32)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# ---------------------------------------------------------------------------
# Config compatibility helpers
# ---------------------------------------------------------------------------


def _validate_vocab_compatibility(encoding, cfg) -> None:
    """
    Guard against tokenizer/model-config vocab_size drift.

    Belt-and-suspenders check: the training entry point reconstructs `cfg` via
    `dataclasses.replace` with the tokenizer's vocab size, so in the common
    path these two values are equal by construction. This helper defends
    against future refactors where the two could diverge (e.g. cfg built
    from a checkpoint, tokenizer swapped out mid-run).

    Args:
        encoding -- tokenizer exposing `.vocab_size`
        cfg      -- model config exposing `.vocab_size`

    Raises:
        ValueError if the two vocab sizes disagree; the message contains both
        values so a mismatch is obvious in logs.
    """
    if encoding.vocab_size != cfg.vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size ({encoding.vocab_size}) does not match "
            f"model cfg.vocab_size ({cfg.vocab_size})"
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def build_dataloader(
    encoding,
    seq_len: int,
    subset: str,
    rank: int,
    world_size: int,
    micro_batch: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Construct the training DataLoader with deterministic worker seeding.

    Factors the DataLoader wiring that ``main()`` previously did inline so
    callers don't have to remember to pass ``worker_init_fn`` — forgetting it
    silently breaks training-run reproducibility.

    Args:
        encoding    -- tokenizer used by the underlying dataset
        seq_len     -- context length per training example
        subset      -- FineWeb-Edu config name (e.g. "sample-10BT")
        rank        -- distributed rank owning this loader
        world_size  -- total distributed world size
        micro_batch -- per-step batch size before grad-accum
        num_workers -- DataLoader worker processes
        pin_memory  -- host-pinned buffers for faster device transfer

    Returns:
        A DataLoader that yields ``(input_ids, target_ids)`` tensor pairs.
    """
    dataset = FineWebEduDataset(encoding, seq_len, subset, rank, world_size)
    return DataLoader(
        dataset,
        batch_size=micro_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
    )
