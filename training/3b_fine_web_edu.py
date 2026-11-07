#!/usr/bin/env python3
"""Deprecated entry point; demoted to re-export shim at Stage 4.5.

Kept so that
    python -m training.3b_fine_web_edu
still works and so ``importlib.util.spec_from_file_location`` in
tests/test_training.py can still load the symbols by file path.

Use ``training/run_3b.py`` for new code.
"""
from training.constants import USE_GRAD_CHECKPOINT
from training.data import (
    FineWebEduDataset,
    _seed_worker,
    _validate_vocab_compatibility,
    build_dataloader,
)
from training.checkpoint import (
    _list_ckpts,
    load_checkpoint,
    save_checkpoint,
)
from training.schedule import get_lr
from training.trainer import Trainer


def main() -> None:
    """Forward to training.run_3b.main()."""
    from training.run_3b import main as _entry
    _entry()


__all__ = [
    "USE_GRAD_CHECKPOINT",
    "FineWebEduDataset",
    "_seed_worker",
    "_validate_vocab_compatibility",
    "build_dataloader",
    "get_lr",
    "Trainer",
    "_list_ckpts",
    "load_checkpoint",
    "save_checkpoint",
    "main",
]


if __name__ == "__main__":
    main()
