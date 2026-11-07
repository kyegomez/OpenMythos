from typing import Optional, Union

import torch
import torch.nn as nn

from open_mythos.blocks import TransformerBlock
from open_mythos.config import MythosConfig
from open_mythos.kv_cache import StaticKVCache
from open_mythos.primitives.norm import RMSNorm
from open_mythos.recurrence.act import ACTHalting
from open_mythos.recurrence.factory import build_injection
from open_mythos.recurrence.lora import LoRAAdapter


# ---------------------------------------------------------------------------
# Recurrent Block (one set of weights, looped T times)
# ---------------------------------------------------------------------------


class RecurrentBlock(nn.Module):
    """
    The core recurrent block of OpenMythos — a single TransformerBlock looped T times.

    At each loop iteration t, the hidden state h is updated via:
        1. loop_index_embedding: inject sinusoidal loop-index signal into h
        2. TransformerBlock:     compute attention + MoE FFN on normalized (h + e)
        3. LoRAAdapter:          apply depth-wise LoRA delta to transformer output
        4. LTIInjection:         stable update h = A·h + B·e + transformer_out
        5. ACTHalting:           accumulate per-position halting probabilities;
                                  positions that have converged stop contributing

    The encoded input e (output of the Prelude) is injected at every step to keep
    the original input signal alive across arbitrary loop depth, preventing drift.
    The ACT mechanism produces a weighted sum of hidden states across iterations,
    where the weights reflect when each position converged.

    More loop iterations at inference = deeper reasoning chains, following the
    depth-extrapolation property of looped transformers (Saunshi et al., 2025).
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses dim, lora_rank, max_loop_iters, act_threshold
        """
        super().__init__()
        self.cfg = cfg
        self.block = TransformerBlock(cfg, use_moe=True)
        # Stage 4.3 (P4.3): factory replaces hardcoded LTIInjection instantiation.
        self.injection = build_injection(cfg)
        self.act = ACTHalting(cfg.dim)
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.norm = RMSNorm(cfg.dim)
        self.loop_dim = (
            cfg.dim // 8
        )  # fraction of channels receiving loop-index embedding
        # Enforce even loop_dim so the sin/cos split in loop_idx_table below is
        # balanced (see loop_index_embedding for rationale). cfg.dim % 16 == 0
        # is the precise invariant but we localize the check here.
        if self.loop_dim % 2 != 0:
            raise ValueError(
                f"RecurrentBlock: loop_dim = cfg.dim // 8 = {self.loop_dim} must "
                f"be even. Set cfg.dim to a multiple of 16 (got cfg.dim={cfg.dim})."
            )

        # P3.7: precompute loop-index sinusoid table once, indexed per iteration.
        # Mirrors loop_index_embedding's math exactly; buffer shape (max_loops, dim)
        # so the forward path can do a single add instead of allocating fresh
        # freqs/sin/cos tensors on every loop iteration.
        loop_theta = 10000.0
        freqs = 1.0 / (
            loop_theta
            ** (torch.arange(0, self.loop_dim, 2, dtype=torch.float32) / self.loop_dim)
        )
        t_range = torch.arange(cfg.max_loop_iters, dtype=torch.float32).unsqueeze(-1)
        angles = t_range * freqs  # (max_loops, loop_dim//2)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:, : self.loop_dim]
        # Pad with zeros to full dim so a simple `h + table[t]` suffices.
        table = torch.zeros(cfg.max_loop_iters, cfg.dim)
        table[:, : self.loop_dim] = emb
        self.register_buffer("loop_idx_table", table)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[Union[dict, StaticKVCache]] = None,
    ) -> torch.Tensor:
        """
        Run the recurrent loop for up to n_loops iterations with ACT early exit.

        Args:
            h        -- initial hidden state from the Prelude, shape (B, T, dim)
            e        -- encoded input frozen for injection each step, shape (B, T, dim)
            freqs_cis-- precomputed RoPE frequencies
            mask     -- additive causal mask or None
            n_loops  -- number of loop iterations; defaults to cfg.max_loop_iters.
                        Can be increased at inference for deeper reasoning (depth extrapolation).
            kv_cache -- cache dict passed through to the inner TransformerBlock;
                        each loop iteration uses a separate cache key

        Returns:
            ACT-weighted sum of hidden states across iterations, shape (B, T, dim)
        """
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape

        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device)
        h_out = torch.zeros_like(h)

        for t in range(n_loops):
            # P3.7: use precomputed loop-index table. If depth extrapolates past
            # the trained max_loops, clamp to the last row (same spirit as LoRA).
            t_idx = min(t, self.loop_idx_table.shape[0] - 1)
            h_loop = h + self.loop_idx_table[t_idx].to(dtype=h.dtype).view(1, 1, -1)
            combined = self.norm(h_loop + e)
            cache_key = f"recurrent_loop_{t}"
            trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
            trans_out = trans_out + self.lora(trans_out, t)
            h_new = self.injection(h, e, trans_out)
            # Gate the hidden-state update: halted positions keep h unchanged so
            # their state is frozen at the iteration they halted on. Running
            # positions take the fresh LTI update. This also prevents halted
            # positions from writing new entries at later cache keys.
            still_running_mask = (~halted).unsqueeze(-1).float()
            h = h_new * still_running_mask + h * (1.0 - still_running_mask)

            p = self.act(h)  # (B, T)
            still_running = ~halted

            # ACT remainder trick: once cumulative_p + p crosses threshold,
            # assign the remaining probability mass as the final weight.
            # Gate by still_running so halted positions contribute exactly
            # once (on the halting step) and zero thereafter — otherwise
            # threshold<1 leaves a non-zero remainder that leaks every step.
            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= self.cfg.act_threshold,
                remainder,
                p,
            )
            weight = weight * still_running.float()
            h_out = h_out + weight.unsqueeze(-1) * h

            cumulative_p = cumulative_p + p * still_running.float()
            halted = halted | (cumulative_p >= self.cfg.act_threshold)

            # P3.5: early-exit removed. The previous `halted.all()` check forced
            # a CPU<->GPU sync every iteration and blocked torch.compile.
            # Halted positions are already gated out of h and h_out updates
            # above, so extra iterations cost only the transformer call for
            # already-halted tokens — an acceptable trade for sync-free loops.
            # Inference (kv_cache != None) had the short-circuit disabled
            # already, so removing it does not change decode behavior.

        return h_out
