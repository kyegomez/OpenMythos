"""Learning-rate schedule for OpenMythos pretraining.

Extracted from training/3b_fine_web_edu.py at Stage 4.5.
"""
from __future__ import annotations

import math


def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    """
    Linear warmup → half-cosine decay to `min_lr`.

    Standard language-model pretraining schedule. The warmup phase prevents
    Adam's second-moment estimate from collapsing to a huge LR in the first
    few steps when gradients are noisy. The cosine tail lets the model make
    small, increasingly conservative updates near the end of training rather
    than crashing to `min_lr` at a fixed step.

    Behavior by region:
        step < warmup                 → linear ramp 0 → max_lr
        warmup ≤ step < total         → cosine decay max_lr → min_lr
        step ≥ total                  → clamped at min_lr (safety for
                                        off-by-one step counters at the end
                                        of training)

    Args:
        step    -- current global optimizer step (0-indexed)
        warmup  -- number of warmup steps before cosine decay begins
        total   -- step at which the cosine reaches `min_lr`
        max_lr  -- peak learning rate reached at the end of warmup
        min_lr  -- floor learning rate at and after `total` steps

    Returns:
        Scalar learning rate for this step.
    """
    if step < warmup:
        return max_lr * step / warmup
    if step >= total:
        return min_lr
    decay = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay))
