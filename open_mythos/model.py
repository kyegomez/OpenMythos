from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from open_mythos.blocks import TransformerBlock
from open_mythos.config import MythosConfig
from open_mythos.kv_cache import StaticKVCache
from open_mythos.primitives.norm import RMSNorm
from open_mythos.primitives.rope import precompute_rope_freqs
from open_mythos.recurrence.block import RecurrentBlock

# Std deviation for weight init (matches GPT-2 family and DeepSeek-V3 convention).
# Centralized here + exported for consistent use across main model and
# experimental variants — prevents silent drift if one site is tuned.
_INIT_STD: float = 0.02


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------


class OpenMythos(nn.Module):
    """
    OpenMythos — Recurrent-Depth Transformer language model.

    Implements the hypothesized Claude Mythos architecture as a Recurrent-Depth
    Transformer (RDT). The model divides computation into three functional blocks:

        Input tokens
             ↓
        [Prelude]          — prelude_layers standard transformer blocks, run once
             ↓
        [Recurrent Block]  — one transformer block looped T times with input injection
             ↑_______↓      h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
             ↓
        [Coda]             — coda_layers standard transformer blocks, run once
             ↓
        Output logits

    Key properties:
    - Same weights, more loops → deeper reasoning, no parameter growth
    - Depth extrapolation: train on N loops, test on N+k loops (emergent)
    - ACT halting: variable compute per position within a batch
    - MoE FFN in the recurrent block: breadth across domains
    - LTI-stable injection: spectral radius < 1 guaranteed by construction
    - Supports both GQA and MLA attention (set via cfg.attn_type)
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig specifying all architecture hyperparameters
        """
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        # Stage 4.3 (P4.3): single freqs_cis buffer sized to the chosen attention's
        # rope dim. Pre-4.3 registered both freqs_cis (GQA) and freqs_cis_mla (MLA)
        # and selected between them in forward; only one was ever read so the other
        # was dead memory. Numerically identical to the old path since the selected
        # dim is unchanged.
        if cfg.attention.attn_type == "mla":
            rope_dim = cfg.attention.qk_rope_head_dim
        else:  # gqa
            rope_dim = cfg.dim // cfg.attention.n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs(rope_dim, cfg.max_seq_len, cfg.rope_theta),
        )

        self.prelude = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.prelude_layers)]
        )
        self.recurrent = RecurrentBlock(cfg)
        self.coda = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.coda_layers)]
        )

        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize all linear and embedding weights with N(0, _INIT_STD)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=_INIT_STD)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=_INIT_STD)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Build an additive causal mask: 0 on and below the diagonal, -inf above.

        Args:
            seq_len -- sequence length
            device  -- target device

        Returns:
            Tensor of shape (1, 1, seq_len, seq_len) broadcastable over (B, H, T, S)
        """
        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: Optional[int] = None,
        kv_cache: Optional[Union[dict, StaticKVCache]] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass through Prelude → Recurrent Block → Coda.

        Args:
            input_ids -- token indices of shape (B, T)
            n_loops   -- recurrent loop depth; defaults to cfg.max_loop_iters.
                         Increase at inference to extrapolate to harder problems.
            kv_cache  -- dict mutated in-place for autoregressive KV caching;
                         pass an empty dict {} and reuse across decode steps
            start_pos -- index of the first token in input_ids within the full
                         sequence; used to select the correct RoPE frequencies
                         during incremental decoding (0 for prefill, prompt_len
                         for each subsequent decode step)

        Returns:
            Logits of shape (B, T, vocab_size)
        """
        T = input_ids.shape[1]
        device = input_ids.device

        x = self.embed(input_ids)
        # Stage 4.3 (P4.3): single buffer precomputed at init; no attn_type branch.
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]
        mask = self._causal_mask(T, device) if T > 1 else None

        for i, layer in enumerate(self.prelude):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"prelude_{i}")

        e = x  # encoded input frozen for injection every loop
        x = self.recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)

        for i, layer in enumerate(self.coda):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"coda_{i}")

        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        n_loops: int = 8,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Autoregressive token generation with KV caching.

        On step 0 the full prompt is processed. On subsequent steps only the
        last generated token is passed, with all previous keys and values
        retrieved from kv_cache. This keeps decode cost proportional to one
        token per step rather than the full growing sequence.

        n_loops can be set higher than the training value to extrapolate to
        harder problems at inference time (depth extrapolation property).

        Args:
            input_ids      -- prompt token indices of shape (B, T)
            max_new_tokens -- number of tokens to generate
            n_loops        -- recurrent loop depth for each decode step
            temperature    -- softmax temperature; lower = more greedy
            top_k          -- restrict sampling to top-K logits (0 = disabled)

        Returns:
            Token indices of shape (B, T + max_new_tokens)
        """
        # P3.3: StaticKVCache pre-allocates (B, max_seq_len, ...) buffers per
        # layer so decode steps only write the new-token slice — no torch.cat
        # reallocation across the generation loop.
        kv_cache: StaticKVCache = StaticKVCache(max_seq_len=self.cfg.max_seq_len)
        prompt_len = input_ids.shape[1]
        for step in range(max_new_tokens):
            if step == 0:
                cur_ids = input_ids
                start_pos = 0
            else:
                cur_ids = input_ids[:, -1:]
                start_pos = prompt_len + step - 1
            # Stop if the next forward would exceed RoPE precomputation range.
            if start_pos + cur_ids.shape[1] > self.cfg.max_seq_len:
                break
            logits = self.forward(
                cur_ids, n_loops=n_loops, kv_cache=kv_cache, start_pos=start_pos
            )
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids
