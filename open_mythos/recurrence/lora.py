import warnings

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Depth-wise LoRA adapter (per loop iteration)
# ---------------------------------------------------------------------------


class LoRAAdapter(nn.Module):
    """
    Depth-wise LoRA adaptation for the recurrent block (Bae et al., 2024).

    Pure weight-tying (identical weights every loop) limits expressiveness;
    fully distinct weights per loop eliminate parameter savings. This adapter
    sits in between: a shared low-rank down-projection and up-projection matrix B
    are shared across all loops, while a small per-loop scale vector shifts the
    effective transformation at each depth without adding significant parameters.

    delta(x, t) = (down(x) * scale[t]) @ B
    """

    def __init__(self, dim: int, rank: int, max_loops: int):
        """
        Args:
            dim       -- model hidden dimension (input and output size)
            rank      -- low-rank bottleneck dimension
            max_loops -- maximum number of loop iterations (determines embedding table size)
        """
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)  # shared A: dim → rank
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.02)  # shared B: rank → dim
        self.scale = nn.Embedding(max_loops, rank)  # per-loop element-wise scale
        # One-shot warning flag for depth extrapolation beyond trained max_loops.
        self._warned_extrapolation = False

    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        """
        Args:
            x      -- input tensor of shape (B, T, dim)
            loop_t -- current loop index used to look up the per-loop scale

        Returns:
            Delta tensor of shape (B, T, dim) to be added to the block output
        """
        # Clamp for depth extrapolation: at inference n_loops can exceed the
        # training max_loop_iters. Iterations beyond the trained range reuse
        # the last learned per-loop scale rather than indexing out of range.
        max_t = self.scale.num_embeddings - 1
        if loop_t > max_t:
            if not self._warned_extrapolation:
                warnings.warn(
                    f"LoRAAdapter: loop_t={loop_t} exceeds max_loops={self.scale.num_embeddings}; "
                    f"reusing final scale for all extrapolated iterations. Model quality may degrade.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_extrapolation = True
            t_idx = max_t
        else:
            t_idx = loop_t
        # P3.6: index the embedding weight directly with a Python int to avoid
        # constructing a scalar tensor every call (faster and torch.compile
        # friendly). self.scale.weight has shape (max_loops, rank).
        s = self.scale.weight[t_idx]  # (rank,)
        down = self.down(x) * s  # (B, T, rank)
        return down @ self.B  # (B, T, dim)
