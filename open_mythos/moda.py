"""
Mixture-of-Depths Attention (MoDA) + DeepSeek Mixture-of-Experts FFN
======================================================================
Paper (attention):   "Mixture-of-Depths Attention"   arXiv 2603.15619
Paper (MoE):         "DeepSeekMoE: Towards Ultimate Expert Specialization
                      in Mixture-of-Experts Language Models" arXiv 2401.06066
Reference impl (V3): https://github.com/deepseek-ai/DeepSeek-V3

Architecture
------------
This file fuses two independent architectural improvements:

  1. **MoDA** — each attention head jointly attends to current-layer sequence
     KV pairs (causal) *and* depth KV pairs from all preceding layers at the
     same token position, under a single softmax.

  2. **DeepSeek MoE** (replaces the dense SwiGLU FFN in every block):
       * K_s  *shared experts* — always activated, capture common knowledge.
       * N_r  *routed experts* — sparse; top-K activated per token.
       * Fine-grained expert segmentation: each expert has a small hidden dim
         (≈ dense_hidden / m) so that activating more experts keeps FLOPs
         constant while improving specialisation.
       * Expert-level balance loss prevents routing collapse.

Gate routing (faithful to DeepSeek-V3 model.py)
------------------------------------------------
  scores       = softmax(x W^T)          # or sigmoid for V3 style
  original     = scores                  # saved for weight computation
  [optional]   scores += bias            # V3 aux-loss-free routing
  [optional]   group-limited masking     # V3 device-group routing
  indices      = topk(scores, K)
  weights      = original[indices]       # un-biased original scores
  [sigmoid]    weights /= sum(weights)   # re-normalise for sigmoid gating
  weights     *= route_scale

Balance loss (DeepSeekMoE §3.3, used when training without V3 bias routing)
  L_ExpBal = Σ_i  f_i · P_i
  f_i = (N_r / (K · T)) · #{tokens routing to i}   (normalised frequency)
  P_i = (1/T) Σ_t s_{i,t}                          (mean soft gate score)

Memory note
-----------
MoDA's unified attention has O(T·L) combined KV length.  For long sequences
use the Triton kernel from https://github.com/hustvl/MoDA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoDAConfig:
    """Configuration for a MoDA + DeepSeek-MoE decoder-only language model.

    Attention (MoDA)
    ----------------
    vocab_size:        Vocabulary size.
    d_model:           Hidden dimension (must equal n_heads_q * head_dim).
    n_layers:          Number of transformer layers.
    n_heads_q:         Query heads.
    n_heads_kv:        Key/value heads for GQA (must divide n_heads_q).
    head_dim:          Per-head dimension (usually d_model // n_heads_q).
    max_seq_len:       Maximum sequence length for the RoPE cache.
    rope_base:         RoPE frequency base.
    attn_dropout:      Attention dropout (0 for inference).
    norm_eps:          RMSNorm epsilon.

    MoE FFN (DeepSeekMoE / DeepSeek-V3 style)
    ------------------------------------------
    n_shared_experts:     Always-active shared experts (K_s).  Capture common
                          knowledge; excluded from routing and balance loss.
    n_routed_experts:     Total pool of routed experts (N_r).
    n_activated_experts:  Top-K selected from routed experts per token (K').
    expert_hidden_dim:    Per-expert intermediate dimension.
                          Set to  dense_ffn_hidden / m  where m is the
                          fine-grained segmentation factor so that total
                          activated FLOPs match a dense FFN:
                          (n_shared + n_activated) × expert_hidden ≈ dense_hidden
    moe_balance_alpha:    Weight of the expert-level balance loss.  Set to
                          0.0 to disable (e.g. when using V3 bias routing).
    moe_score_func:       "softmax" (DeepSeekMoE / V2) or "sigmoid" (V3).
    moe_n_groups:         Number of expert groups for group-limited routing
                          (V3 uses 8; set 1 to disable, default).
    moe_topk_groups:      Number of groups a token may route to
                          (V3 uses 3; set 1 to disable, default).
    moe_route_scale:      Scalar multiplied onto the selected gate weights
                          after normalisation (V3 uses 2.5446; default 1.0).

    Defaults approximate the DeepSeekMoE 2B configuration scaled to
    d_model = 2048, keeping per-token FLOPs equal to a dense SwiGLU with
    hidden_dim = 5 632  (≈ 8/3 × 2048):
        (n_shared + n_activated) × expert_hidden = (2+6) × 704 = 5 632.
    """

    # ---- Transformer / MoDA ----
    vocab_size: int = 32_000
    d_model: int = 2048
    n_layers: int = 24
    n_heads_q: int = 16
    n_heads_kv: int = 8
    head_dim: int = 128
    max_seq_len: int = 4_096
    rope_base: float = 10_000.0
    attn_dropout: float = 0.0
    norm_eps: float = 1e-6

    # ---- DeepSeek MoE FFN ----
    n_shared_experts: int = 2  # K_s
    n_routed_experts: int = 64  # N_r
    n_activated_experts: int = 6  # K' top-K from routed pool
    expert_hidden_dim: int = 704  # per-expert intermediate dim
    moe_balance_alpha: float = 0.001  # balance-loss weight (0 = disabled)
    moe_score_func: str = "softmax"  # "softmax" | "sigmoid"
    moe_n_groups: int = 1  # expert groups (1 = no grouping)
    moe_topk_groups: int = 1  # groups to route into (1 = no limit)
    moe_route_scale: float = 1.0  # gate-weight scale factor


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias, no mean subtraction)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Create an RMSNorm layer.

        Args:
            dim: Feature dimension to normalise over (the last axis of input).
            eps: Stability constant added before the reciprocal square-root to
                 prevent division by zero when the RMS is near zero.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise *x* by its root-mean-square and apply a learnable scale.

        Args:
            x: Input tensor of arbitrary shape ``[..., dim]``.

        Returns:
            Normalised tensor, same shape as *x*.
        """
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) with lazy cache extension.

    Args:
        dim:         Per-head dimension (head_dim).
        max_seq_len: Initial cache size.
        base:        Frequency base (default 10 000).
    """

    def __init__(
        self, dim: int, max_seq_len: int = 8_192, base: float = 10_000.0
    ) -> None:
        """Initialise RoPE and pre-compute the cos/sin cache.

        Args:
            dim:         Per-head dimension.  Must be even.
            max_seq_len: Number of positions to cache on construction.  The
                         cache doubles automatically when a longer sequence
                         is encountered.
            base:        Frequency base θ.  Higher values slow the rotation
                         rate, extending effective context length.
        """
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Pre-compute and register ``_cos`` / ``_sin`` buffers up to *seq_len*.

        Called once at init and again (doubling capacity) whenever ``forward``
        is asked for a sequence longer than the current cache.

        Args:
            seq_len: Number of positions to pre-compute.
        """
        t = torch.arange(
            seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [T, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [T, dim]
        self.register_buffer("_cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("_sin", emb.sin()[None, None], persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cached (cos, sin) tables for the first *seq_len* positions.

        Args:
            seq_len: Number of positions required.

        Returns:
            Tuple of ``(cos, sin)``, each shaped ``[1, 1, seq_len, dim]``,
            ready to broadcast with ``[B, H, T, dim]`` query / key tensors.
        """
        if seq_len > self._cos.shape[2]:
            self._build_cache(seq_len * 2)
        return self._cos[:, :, :seq_len], self._sin[:, :, :seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Return *x* with its last dimension split and swapped with negation.

    Given ``x = [x₁, x₂]`` (each half of the last dim), returns
    ``[-x₂, x₁]``.  Combined with the cos/sin multiply in
    :func:`apply_rotary_emb` this implements the 2-D rotation matrix
    that defines RoPE.

    Args:
        x: Tensor with an even-sized last dimension.

    Returns:
        Rotated tensor with the same shape as *x*.
    """
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply Rotary Position Embeddings in-place to query or key tensors.

    Implements ``x_rot = x * cos + rotate_half(x) * sin``, which is
    equivalent to multiplying each consecutive pair of dimensions by a
    2-D rotation matrix whose angle depends on the sequence position and
    the dimension's frequency.

    Args:
        x:   Query or key tensor, shape ``[B, H, T, d]``.
        cos: Pre-computed cosines, shape ``[1, 1, T, d]``.
        sin: Pre-computed sines,   shape ``[1, 1, T, d]``.

    Returns:
        Rotated tensor with the same shape and dtype as *x*.
    """
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# DeepSeek MoE FFN
# ---------------------------------------------------------------------------


class DeepSeekExpert(nn.Module):
    """Single fine-grained SwiGLU expert.

    Faithful to DeepSeek-V3 ``Expert``:
        output = w2( SiLU(w1(x)) ⊙ w3(x) )

    where w1 is the gate projection, w3 the up-projection, w2 the
    down-projection — identical to a SwiGLU FFN at smaller hidden dim.

    Args:
        d_model:    Input / output dimension.
        hidden_dim: Expert intermediate dimension (≪ dense FFN hidden_dim).
    """

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        """Create a single fine-grained SwiGLU expert.

        Args:
            d_model:    Token hidden dimension (input and output size).
            hidden_dim: Expert intermediate dimension.  Typically much
                        smaller than the dense FFN hidden dim — set to
                        ``dense_hidden / m`` where *m* is the fine-grained
                        segmentation factor so total activated FLOPs match
                        the dense baseline.
        """
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)  # gate
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)  # up
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``w2( SiLU(w1(x)) ⊙ w3(x) )``.

        Args:
            x: Token features assigned to this expert, shape
               ``[num_assigned_tokens, d_model]``.

        Returns:
            Expert output, shape ``[num_assigned_tokens, d_model]``.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DeepSeekGate(nn.Module):
    """Token-to-expert routing gate.

    Faithful to DeepSeek-V3 ``Gate`` (minus distributed sharding).

    Routing algorithm
    ~~~~~~~~~~~~~~~~~
    1.  ``scores = softmax(x W^T)``  or  ``sigmoid(x W^T)``
    2.  ``original_scores = scores``  (saved — will be used for gate weights)
    3.  [optional] ``scores += bias``  (V3 aux-loss-free bias routing)
    4.  [optional] Group-limited masking:
            - reshape scores → [T, n_groups, experts_per_group]
            - keep only top-``topk_groups`` groups per token
            - mask the rest to −∞
    5.  ``indices = topk(scores, K')``          (routing decision)
    6.  ``weights = original_scores[indices]``  (un-biased weights)
    7.  [sigmoid only] ``weights /= sum(weights)``  (re-normalise)
    8.  ``weights *= route_scale``

    The ``original_scores`` (full distribution, before bias/masking) are also
    returned so the MoE layer can compute the expert-level balance loss.

    Args:
        d_model:           Token hidden dimension.
        n_routed_experts:  Total routed expert pool size (N_r).
        n_activated:       Top-K experts to select (K').
        score_func:        ``"softmax"`` or ``"sigmoid"``.
        n_groups:          Number of expert groups (1 = disabled).
        topk_groups:       Groups a token may route to (1 = disabled).
        route_scale:       Scalar applied to final gate weights.
        use_bias:          If True, add a learnable per-expert bias used only
                           for the routing decision (V3 aux-loss-free scheme).
    """

    def __init__(
        self,
        d_model: int,
        n_routed_experts: int,
        n_activated: int,
        score_func: str = "softmax",
        n_groups: int = 1,
        topk_groups: int = 1,
        route_scale: float = 1.0,
        use_bias: bool = False,
    ) -> None:
        """Create the routing gate.

        Args:
            d_model:          Token hidden dimension.
            n_routed_experts: Total number of routed experts in the pool (N_r).
            n_activated:      How many experts to select per token (K').
            score_func:       Affinity function — ``"softmax"`` (original
                              DeepSeekMoE / V2) or ``"sigmoid"`` (V3).
            n_groups:         Number of expert groups for device-limited
                              routing.  Set to 1 to disable (default).
            topk_groups:      Number of groups each token may route into.
                              Set to 1 to disable (default).
            route_scale:      Scalar multiplied onto gate weights after
                              optional sigmoid normalisation (V3 uses 2.5446;
                              default 1.0 leaves weights unchanged).
            use_bias:         If ``True``, initialise a learnable per-expert
                              float32 bias added to routing scores only (not
                              gate weights).  Enables the V3 aux-loss-free
                              load-balancing scheme where the bias is adjusted
                              outside the optimizer to track expert loads.
        """
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.n_activated = n_activated
        self.score_func = score_func
        self.n_groups = n_groups
        self.topk_groups = topk_groups
        self.route_scale = route_scale

        # Gating projection: [N_r, D]
        self.weight = nn.Parameter(torch.empty(n_routed_experts, d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Optional per-expert routing bias (V3 aux-loss-free load balancing).
        # Updated outside the optimizer by monitoring expert loads — not trained
        # through the balance loss.  Initialised to zero.
        self.bias: Optional[nn.Parameter] = (
            nn.Parameter(torch.zeros(n_routed_experts, dtype=torch.float32))
            if use_bias
            else None
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing weights and expert indices.

        Args:
            x: ``[num_tokens, d_model]`` (flattened B × T).

        Returns:
            weights:        ``[num_tokens, K']``  gate weights (dtype = x.dtype).
            indices:        ``[num_tokens, K']``  selected expert indices (int64).
            original_scores: ``[num_tokens, N_r]``  full soft scores (float32),
                             used by :class:`DeepSeekMoE` for the balance loss.
        """
        # Affinity logits
        logits = F.linear(x, self.weight)  # [T, N_r]

        if self.score_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        else:  # sigmoid (V3)
            scores = logits.sigmoid().to(torch.float32)

        original_scores = scores  # un-biased; used for weights + balance loss

        # Routing scores (may differ from original_scores if bias is active)
        routing = scores
        if self.bias is not None:
            routing = routing + self.bias

        # Group-limited routing (V3 device-group constraint)
        if self.n_groups > 1:
            # [T, n_groups, experts_per_group]
            g = routing.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = g.amax(dim=-1)  # [T, G]
            else:
                # Top-2 sum per group (V3 heuristic)
                group_scores = g.topk(2, dim=-1)[0].sum(dim=-1)
            _, top_groups = group_scores.topk(self.topk_groups, dim=-1)  # [T, topk_g]
            mask = torch.ones(
                x.size(0), self.n_groups, dtype=torch.bool, device=x.device
            ).scatter_(
                1, top_groups, False
            )  # True = masked out
            routing = g.masked_fill(mask.unsqueeze(-1), float("-inf")).flatten(1)

        # Top-K selection (on routing scores which may include bias / group mask)
        _, indices = routing.topk(self.n_activated, dim=-1)  # [T, K']

        # Gate weights from original (un-biased) scores
        weights = original_scores.gather(1, indices)  # [T, K']

        if self.score_func == "sigmoid":
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        weights = (weights * self.route_scale).to(x.dtype)
        return weights, indices, original_scores


class DeepSeekMoE(nn.Module):
    """DeepSeek Mixture-of-Experts layer — drop-in replacement for a dense FFN.

    Combines shared experts (always active) and routed experts (sparse top-K)
    exactly as in DeepSeek-V3 ``MoE``, adapted for single-device training
    (no ColumnParallel/RowParallel, no all_reduce).

    Forward pass
    ~~~~~~~~~~~~
    ::

        x_flat = x.view(-1, D)                         # [B*T, D]

        # Shared path (always executed)
        z = shared_experts(x_flat)                     # [B*T, D]

        # Routed path (sparse)
        weights, indices, scores = gate(x_flat)        # [B*T, K'], [B*T, K'], [B*T, N_r]
        y = zeros_like(x_flat)
        for each expert i:
            toks = tokens that selected expert i
            y[toks] += experts[i](x_flat[toks]) * weights[toks, rank_of_i]

        output = (y + z).view(B, T, D)

        # Training: expert-level balance loss (DeepSeekMoE §3.3)
        L_ExpBal = Σ_i  f_i · P_i
          f_i = (N_r / (K' · T)) · #{tokens → expert i}
          P_i = mean_t(scores_{t,i})

    Args:
        cfg: :class:`MoDAConfig` instance.
    """

    def __init__(self, cfg: MoDAConfig) -> None:
        """Build the MoE layer from a :class:`MoDAConfig`.

        Constructs:
          * ``shared_experts`` — one dense SwiGLU FFN with hidden dimension
            ``n_shared_experts × expert_hidden_dim``.
          * ``gate``           — :class:`DeepSeekGate` for top-K routing.
          * ``experts``        — ``nn.ModuleList`` of ``n_routed_experts``
            :class:`DeepSeekExpert` instances, each with ``expert_hidden_dim``
            intermediate units.

        Args:
            cfg: Model configuration.  The relevant fields are
                 ``n_shared_experts``, ``n_routed_experts``,
                 ``n_activated_experts``, ``expert_hidden_dim``,
                 ``moe_balance_alpha``, ``moe_score_func``,
                 ``moe_n_groups``, ``moe_topk_groups``, and
                 ``moe_route_scale``.
        """
        super().__init__()
        self.d_model = cfg.d_model
        self.n_routed_experts = cfg.n_routed_experts
        self.n_activated_experts = cfg.n_activated_experts
        self.moe_balance_alpha = cfg.moe_balance_alpha

        # Shared experts: single dense SwiGLU with hidden = K_s × expert_hidden
        # (matches DeepSeek-V3's ``MLP(dim, n_shared_experts * moe_inter_dim)``)
        shared_hidden = cfg.n_shared_experts * cfg.expert_hidden_dim
        self.shared_experts = _SharedFFN(cfg.d_model, shared_hidden)

        # Routing gate
        self.gate = DeepSeekGate(
            d_model=cfg.d_model,
            n_routed_experts=cfg.n_routed_experts,
            n_activated=cfg.n_activated_experts,
            score_func=cfg.moe_score_func,
            n_groups=cfg.moe_n_groups,
            topk_groups=cfg.moe_topk_groups,
            route_scale=cfg.moe_route_scale,
            use_bias=False,
        )

        # Routed experts pool
        self.experts = nn.ModuleList(
            [
                DeepSeekExpert(cfg.d_model, cfg.expert_hidden_dim)
                for _ in range(cfg.n_routed_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the MoE layer.

        Args:
            x: ``[B, T, D]`` hidden states.

        Returns:
            output:        ``[B, T, D]``  updated hidden states.
            balance_loss:  Scalar expert-level balance loss (during training),
                           or ``None`` during inference.
        """
        shape = x.shape
        x_flat = x.view(-1, self.d_model)  # [T_tot, D]
        n_tokens = x_flat.size(0)

        # ---- Shared experts (all tokens) ---------------------------------
        z = self.shared_experts(x_flat)  # [T_tot, D]

        # ---- Routed experts (sparse) -------------------------------------
        weights, indices, scores = self.gate(x_flat)
        # weights: [T_tot, K'], indices: [T_tot, K'], scores: [T_tot, N_r]

        y = torch.zeros_like(x_flat)

        # Dispatch: for each expert compute on its assigned tokens
        # (token-major loop matches DeepSeek-V3's reference implementation)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)
        for i, expert in enumerate(self.experts):
            if counts[i].item() == 0:
                continue
            tok_idx, rank_in_k = torch.where(
                indices == i
            )  # which tokens & which K slot
            y[tok_idx] += expert(x_flat[tok_idx]) * weights[tok_idx, rank_in_k, None]

        output = (y + z).view(shape)

        # ---- Expert-level balance loss (DeepSeekMoE §3.3) ----------------
        balance_loss: Optional[torch.Tensor] = None
        if self.training and self.moe_balance_alpha > 0.0:
            balance_loss = self._balance_loss(indices, scores, n_tokens)

        return output, balance_loss

    def _balance_loss(
        self,
        indices: torch.Tensor,  # [T, K']  int64
        scores: torch.Tensor,  # [T, N_r] float32 (full distribution)
        n_tokens: int,
    ) -> torch.Tensor:
        """Compute the expert-level balance loss (DeepSeekMoE §3.3).

        Penalises routing imbalance by encouraging the model to spread tokens
        evenly across experts.  Only the soft-score term ``P_i`` receives a
        gradient; the hard-count term ``f_i`` is non-differentiable and acts
        as a fixed weighting coefficient.

        ::

            f_i = (N_r / (K' × T)) × #{tokens routed to expert i}
            P_i = (1/T) Σ_t scores[t, i]
            L   = Σ_i  f_i · P_i

        For perfect balance ``f_i = 1`` for all *i* and ``L = Σ P_i = 1``
        (softmax) or some constant (sigmoid).  Overloaded experts produce
        large ``f_i``, pushing their mean score ``P_i`` up via the gradient
        and thereby attracting more tokens — stabilising load over training.

        Args:
            indices:  ``[T, K']`` int64 — expert indices selected per token.
            scores:   ``[T, N_r]`` float32 — full soft distribution from the
                      gate (before top-K selection), used for ``P_i``.
            n_tokens: Total number of tokens in the batch (``B × T``).

        Returns:
            Scalar balance loss tensor.
        """
        Nr, K = self.n_routed_experts, self.n_activated_experts

        # Routing counts per expert (non-differentiable)
        counts = torch.zeros(Nr, dtype=torch.float32, device=indices.device)
        counts.scatter_add_(
            0,
            indices.flatten(),
            torch.ones(indices.numel(), dtype=torch.float32, device=indices.device),
        )
        f = counts * (Nr / (K * n_tokens))  # normalised frequency [N_r]

        # Mean soft gate score per expert (differentiable through softmax/sigmoid)
        P = scores.mean(dim=0)  # [N_r]

        # f is derived from hard top-K → no gradient; gradient flows through P only
        return (f * P).sum()


class _SharedFFN(nn.Module):
    """Dense SwiGLU FFN used for the always-active shared experts.

    Mirrors :class:`DeepSeekExpert` but is a single larger MLP whose
    ``hidden_dim`` equals ``n_shared_experts × expert_hidden_dim``.  This
    matches DeepSeek-V3's ``MLP(dim, n_shared_experts * moe_inter_dim)``.

    Not part of the public API — instantiated only by :class:`DeepSeekMoE`.
    """

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        """Create the shared-expert FFN.

        Args:
            d_model:    Token hidden dimension (input and output).
            hidden_dim: Combined intermediate size for all shared experts
                        (``n_shared_experts × expert_hidden_dim``).
        """
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared SwiGLU FFN to every token.

        Args:
            x: Flattened token features, shape ``[B*T, d_model]``.

        Returns:
            Transformed features, shape ``[B*T, d_model]``.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# MoDA Attention (unchanged from base file)
# ---------------------------------------------------------------------------


class MoDAAttention(nn.Module):
    """Mixture-of-Depths Attention — read side.

    Each query jointly attends (single softmax) to:
      * Sequence KVs at the current layer (causal GQA).
      * Depth KVs from all preceding layers at the *same* token position.

    Depth cache entries are written externally by :class:`MoDABlock` from
    the full block output X_l^out (after the MoE FFN).

    Args:
        cfg: :class:`MoDAConfig` instance.
    """

    def __init__(self, cfg: MoDAConfig) -> None:
        """Build the MoDA attention module.

        Creates four projection matrices (Q, K, V, O) sized for GQA and
        stores the attention scale and dropout rate.

        Args:
            cfg: Model configuration.  Must satisfy
                 ``n_heads_q % n_heads_kv == 0`` (GQA requirement).

        Raises:
            ValueError: If ``n_heads_q`` is not divisible by ``n_heads_kv``.
        """
        super().__init__()
        if cfg.n_heads_q % cfg.n_heads_kv != 0:
            raise ValueError(
                f"n_heads_q ({cfg.n_heads_q}) must be divisible by "
                f"n_heads_kv ({cfg.n_heads_kv}) for GQA."
            )

        self.n_heads_q = cfg.n_heads_q
        self.n_heads_kv = cfg.n_heads_kv
        self.head_dim = cfg.head_dim
        self.gqa_group = cfg.n_heads_q // cfg.n_heads_kv
        self.scale = cfg.head_dim**-0.5
        self.dropout = cfg.attn_dropout

        inner_q = cfg.n_heads_q * cfg.head_dim
        inner_kv = cfg.n_heads_kv * cfg.head_dim

        self.q_proj = nn.Linear(cfg.d_model, inner_q, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, inner_kv, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, inner_kv, bias=False)
        self.o_proj = nn.Linear(inner_q, cfg.d_model, bias=False)

    def _expand_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads along dim 1 to match the number of query heads.

        With GQA group size G, each KV head is shared by G query heads.
        ``repeat_interleave(G, dim=1)`` produces the correct interleaved
        expansion so that query head ``h`` is paired with KV head ``h // G``.

        Args:
            kv: Key or value tensor whose dim 1 is the KV-head axis.
                Supported shapes: ``[B, Hk, T, d]`` (sequence) and
                ``[B, Hk, T, L, d]`` (depth stack).

        Returns:
            Tensor with dim 1 expanded from ``Hk`` to ``Hq = Hk × G``.
            Returns *kv* unchanged when ``gqa_group == 1``.
        """
        if self.gqa_group == 1:
            return kv
        return kv.repeat_interleave(self.gqa_group, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        depth_k_cache: List[torch.Tensor],
        depth_v_cache: List[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MoDA attention output.

        Args:
            x:             ``[B, T, D]`` input hidden states.
            depth_k_cache: ``L`` tensors each ``[B, Hk, T, d]`` — depth keys.
            depth_v_cache: Matching depth values.
            cos/sin:       RoPE tables ``[1, 1, T, d]``.

        Returns:
            ``[B, T, D]`` output hidden states.
        """
        B, T, D = x.shape
        Hq, Hk, d = self.n_heads_q, self.n_heads_kv, self.head_dim

        Q = self.q_proj(x).view(B, T, Hq, d).transpose(1, 2)
        K = self.k_proj(x).view(B, T, Hk, d).transpose(1, 2)
        V = self.v_proj(x).view(B, T, Hk, d).transpose(1, 2)

        Q = apply_rotary_emb(Q, cos, sin)
        K = apply_rotary_emb(K, cos, sin)

        K_e = self._expand_kv(K)
        V_e = self._expand_kv(V)

        L = len(depth_k_cache)

        if L == 0:
            out = F.scaled_dot_product_attention(
                Q,
                K_e,
                V_e,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale,
            )
        else:
            # Sequence logits [B, Hq, T, T] with causal mask
            seq_logits = torch.matmul(Q, K_e.transpose(-2, -1)) * self.scale
            causal_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device, dtype=Q.dtype),
                diagonal=1,
            )
            seq_logits = seq_logits + causal_mask

            # Depth KVs: [B, Hk, L, T, d] → [B, Hk, T, L, d]
            K_depth = torch.stack(depth_k_cache, dim=2).permute(0, 1, 3, 2, 4)
            V_depth = torch.stack(depth_v_cache, dim=2).permute(0, 1, 3, 2, 4)
            K_depth_e = self._expand_kv(K_depth)
            V_depth_e = self._expand_kv(V_depth)

            # Depth logits [B, Hq, T, L]
            depth_logits = torch.einsum("bhid,bhild->bhil", Q, K_depth_e) * self.scale

            # Unified softmax over T + L positions
            combined = torch.cat([seq_logits, depth_logits], dim=-1)
            weights = F.softmax(combined, dim=-1)
            if self.training and self.dropout > 0.0:
                weights = F.dropout(weights, p=self.dropout)

            seq_contrib = torch.matmul(weights[:, :, :, :T], V_e)
            depth_contrib = torch.einsum(
                "bhil,bhild->bhid", weights[:, :, :, T:], V_depth_e
            )
            out = seq_contrib + depth_contrib

        out = out.transpose(1, 2).reshape(B, T, Hq * d)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# MoDA Transformer Block
# ---------------------------------------------------------------------------


class MoDABlock(nn.Module):
    """Single MoDA + DeepSeek-MoE transformer block.

    Structure (post-norm, per MoDA paper recommendation):

    .. code-block::

        x  ──► Attention ──► + ──► RMSNorm ──► x_mid
        x                    ↑ (residual)
        x_mid ──► MoE    ──► + ──► RMSNorm ──► x_out
        x_mid               ↑ (residual)
        x_out ──► W_K^W  ──► k_write  }  appended to MoDA depth KV cache
              └─► W_V^W  ──► v_write  }  by MoDAModel for the next layer

    The MoE layer also returns an optional expert-level balance loss scalar
    which is propagated up to :class:`MoDAModel` for inclusion in the total
    training loss.

    Args:
        cfg: :class:`MoDAConfig` instance.
    """

    def __init__(self, cfg: MoDAConfig) -> None:
        """Build one MoDA + MoE transformer block.

        Constructs and wires together:
          * ``attn``      — :class:`MoDAAttention` (depth-aware GQA).
          * ``moe``       — :class:`DeepSeekMoE` (shared + routed experts).
          * ``norm_attn`` / ``norm_ffn`` — post-sublayer :class:`RMSNorm`.
          * ``k_write`` / ``v_write`` — depth-cache write projections
            ``D → n_heads_kv × head_dim``.

        Args:
            cfg: Model configuration.
        """
        super().__init__()
        inner_kv = cfg.n_heads_kv * cfg.head_dim

        self.attn = MoDAAttention(cfg)
        self.moe = DeepSeekMoE(cfg)
        self.norm_attn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.norm_ffn = RMSNorm(cfg.d_model, cfg.norm_eps)

        # MoDA depth-cache write projections: K_l = X_l^out W_K^W, V_l = X_l^out W_V^W
        self.k_write = nn.Linear(cfg.d_model, inner_kv, bias=False)
        self.v_write = nn.Linear(cfg.d_model, inner_kv, bias=False)

        self._n_heads_kv = cfg.n_heads_kv
        self._head_dim = cfg.head_dim

    def forward(
        self,
        x: torch.Tensor,
        depth_k_cache: List[torch.Tensor],
        depth_v_cache: List[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Run one MoDA + MoE transformer block.

        Args:
            x:             ``[B, T, D]`` input hidden states.
            depth_k_cache: Depth keys from all preceding layers, each ``[B, Hk, T, d]``.
            depth_v_cache: Matching depth values.
            cos/sin:       RoPE tables ``[1, 1, T, d]``.

        Returns:
            x_out:        ``[B, T, D]`` updated hidden states.
            k_write:      ``[B, Hk, T, d]`` depth cache key for this layer.
            v_write:      ``[B, Hk, T, d]`` depth cache value for this layer.
            balance_loss: Scalar expert-level balance loss, or ``None`` at inference.
        """
        B, T, _ = x.shape

        # Post-norm attention sub-layer
        x = self.norm_attn(x + self.attn(x, depth_k_cache, depth_v_cache, cos, sin))

        # Post-norm MoE sub-layer
        moe_out, balance_loss = self.moe(x)
        x = self.norm_ffn(x + moe_out)

        # Depth write projections from X_l^out (full block output, after MoE)
        k_write = (
            self.k_write(x).view(B, T, self._n_heads_kv, self._head_dim).transpose(1, 2)
        )
        v_write = (
            self.v_write(x).view(B, T, self._n_heads_kv, self._head_dim).transpose(1, 2)
        )

        # RoPE on k_write for positional consistency during future depth reads
        k_write = apply_rotary_emb(k_write, cos, sin)

        return x, k_write, v_write, balance_loss


# ---------------------------------------------------------------------------
# Full MoDA + MoE Language Model
# ---------------------------------------------------------------------------


class MoDAModel(nn.Module):
    """Decoder-only LM with Mixture-of-Depths Attention and DeepSeek MoE FFN.

    Loss = LM cross-entropy  +  moe_balance_alpha × mean(per-layer balance losses)

    The depth KV cache is a local list inside :meth:`forward` — never stored
    on ``self``, so autograd is clean across independent forward calls.

    Args:
        cfg: :class:`MoDAConfig` specifying the full model.
    """

    def __init__(self, cfg: MoDAConfig) -> None:
        """Build the full MoDA + MoE language model.

        Constructs the token embedding, RoPE, all transformer blocks, a final
        RMSNorm, and the language-model head.  The embedding and LM-head
        weights are tied so they share the same parameter.

        Args:
            cfg: :class:`MoDAConfig` that fully specifies the model.
        """
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.blocks = nn.ModuleList([MoDABlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_out = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.lm_head.weight = self.embed.weight  # weight tying

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply GPT-style weight initialisation to every sub-module.

        * :class:`nn.Linear` and :class:`nn.Embedding` weights are drawn from
          ``Normal(0, 0.02)`` — the standard initialisation used by GPT-2 and
          most subsequent transformer implementations.
        * :class:`DeepSeekGate` weight matrices are re-initialised with
          ``kaiming_uniform`` (fan-in) to match the default ``nn.Linear``
          init and avoid the Normal distribution being too narrow for a matrix
          used without a subsequent non-linearity.

        Called automatically at the end of :meth:`__init__`.
        """
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, DeepSeekGate):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the full MoDA + MoE language model.

        Args:
            input_ids: ``[B, T]`` token indices.
            labels:    ``[B, T]`` targets for LM loss; -100 positions ignored.

        Returns:
            logits:    ``[B, T, vocab_size]``.
            loss:      ``lm_loss + balance_loss`` if labels provided, else ``None``.
        """
        B, T = input_ids.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len={self.cfg.max_seq_len}."
            )

        x = self.embed(input_ids)
        cos, sin = self.rope(T)

        depth_k_cache: List[torch.Tensor] = []
        depth_v_cache: List[torch.Tensor] = []
        balance_losses: List[torch.Tensor] = []

        for block in self.blocks:
            x, k_write, v_write, bal = block(x, depth_k_cache, depth_v_cache, cos, sin)
            depth_k_cache.append(k_write)
            depth_v_cache.append(v_write)
            if bal is not None:
                balance_losses.append(bal)

        x = self.norm_out(x)
        logits = self.lm_head(x)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            if balance_losses and self.cfg.moe_balance_alpha > 0.0:
                avg_balance = torch.stack(balance_losses).mean()
                loss = lm_loss + self.cfg.moe_balance_alpha * avg_balance
            else:
                loss = lm_loss

        return logits, loss

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count the total number of scalar parameters in the model.

        Args:
            trainable_only: If ``True``, count only parameters with
                            ``requires_grad=True``, excluding frozen layers.

        Returns:
            Integer parameter count.
        """
        params = (
            self.parameters()
            if not trainable_only
            else (p for p in self.parameters() if p.requires_grad)
        )
        return sum(p.numel() for p in params)

    def extra_repr(self) -> str:
        """Return a one-line summary string shown inside ``repr(model)``.

        Displayed by PyTorch's ``__repr__`` directly after the class name,
        before the sub-module tree.

        Returns:
            Human-readable string listing key model dimensions and the total
            parameter count.
        """
        c = self.cfg
        return (
            f"vocab={c.vocab_size}, d_model={c.d_model}, layers={c.n_layers}, "
            f"heads={c.n_heads_q}/{c.n_heads_kv} (GQA), "
            f"experts=shared×{c.n_shared_experts}+routed×{c.n_routed_experts}"
            f"(top-{c.n_activated_experts}), "
            f"params={self.num_parameters():,}"
        )
