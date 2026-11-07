"""Checkpoint I/O for OpenMythos pretraining.

Extracted from ``training/trainer.py`` to keep both modules under the
400-line project ceiling. Pure relocation — the three functions and
their docstrings are unchanged from the pre-extraction version, only
the containing module moved.

- ``_list_ckpts``: scan a directory for ``step_*.pt`` files.
- ``save_checkpoint``: gather full FSDP state and write atomically.
- ``load_checkpoint``: restore + vocab_size compatibility check.
"""
from __future__ import annotations

import os

import torch
from loguru import logger
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)


def _list_ckpts(ckpt_dir: str) -> list[str]:
    """
    Return checkpoint paths in `ckpt_dir` sorted oldest → newest.

    Relies on the zero-padded `step_{0000000}.pt` filename convention so
    lexicographic sort matches chronological order. Changing the filename
    format elsewhere without updating the pad width would silently break
    both `keep_last` pruning and resume-latest on startup, since both pick
    the last element of this list.

    Args:
        ckpt_dir -- directory to scan; missing directory returns []

    Returns:
        Sorted list of absolute paths to matching checkpoint files.
    """
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
    """
    Gather full model + optimizer state, write atomically, prune old files.

    Under FSDP both states are collected inside a single FULL_STATE_DICT
    context so the optim-state tensors bind to fully-unsharded parameters;
    mixing contexts between model and optimizer has caused silent divergence
    on resume in past torch versions. The temp-file + os.replace write means
    a kill mid-save leaves the previous checkpoint intact instead of a
    truncated .pt file. Non-master ranks participate in the FSDP gather
    (otherwise the collective would hang) but exit before touching disk.

    Args:
        model       -- FSDP-wrapped (ddp=True) or raw (ddp=False) model
        optimizer   -- the optimizer whose state should round-trip with the model
        step        -- global step number; encoded zero-padded into the filename
        cfg         -- model config object; saved so downstream eval can
                       reconstruct the model without re-importing the variant
        vocab_size  -- tokenizer vocab size at train time; saved for sanity-check
                       on load against a (possibly updated) tokenizer
        ckpt_dir    -- directory to write into; created if missing
        ddp         -- True if FSDP path; False for single-GPU / CPU
        master      -- whether this rank writes to disk (rank 0 only)
        keep_last   -- number of most-recent checkpoints to retain; older ones
                       are unlinked after a successful write

    Returns:
        None. Writes to disk as a side effect on master rank.
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

    for name, tensor in model_state.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(
                f"Non-finite parameter {name!r} at checkpoint save — refusing to write"
            )

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

    logger.success(f"Checkpoint saved → {final_path}")


def load_checkpoint(model, optimizer, path: str, ddp: bool, cfg=None) -> int:
    """
    Restore model + optimizer from disk, returning the step to resume at.

    Every rank reads the file (`rank0_only=False` on load) so FSDP has access
    to the full state on each rank — the complement to the `rank0_only=True`
    save path. Must mirror save's single-context pattern; splitting the model
    and optimizer loads across two `state_dict_type` blocks has historically
    produced optimizer state bound to the wrong shard shapes.

    `weights_only=False` is required because the checkpoint contains the
    pickled `cfg` dataclass — flip to `weights_only=True` only if you
    separate config out.

    Args:
        model     -- same FSDP-wrapped or raw model used during save
        optimizer -- freshly constructed optimizer to be filled in-place
        path      -- absolute path to a `step_{N:07d}.pt` file produced by
                     `save_checkpoint`
        ddp       -- whether the model is FSDP-wrapped; must match the save run
        cfg       -- current run's model config; when provided, the checkpoint's
                     stored `vocab_size` is compared against `cfg.vocab_size`
                     and a mismatch raises ValueError. Defended here so a
                     tokenizer swap between runs fails loud instead of loading
                     weights with the wrong embedding table shape.

    Returns:
        The step number the checkpoint was taken at; the caller advances the
        training loop from this value.

    Raises:
        ValueError if `cfg` is provided and the checkpoint's vocab_size
        disagrees with `cfg.vocab_size`.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if cfg is not None:
        ckpt_vocab = ckpt.get("vocab_size")
        if ckpt_vocab is not None and ckpt_vocab != cfg.vocab_size:
            raise ValueError(
                f"Checkpoint vocab_size ({ckpt_vocab}) does not match "
                f"current cfg.vocab_size ({cfg.vocab_size}); tokenizer may have changed"
            )

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
