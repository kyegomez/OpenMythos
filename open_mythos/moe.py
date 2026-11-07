from typing import Protocol, Tuple, Union, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

from open_mythos.config import MythosConfig
from open_mythos.primitives.expert import Expert


# ---------------------------------------------------------------------------
# Router Protocol + default implementation (Stage 4.3 completion)
# ---------------------------------------------------------------------------


@runtime_checkable
class RouterProtocol(Protocol):
    """Structural contract for MoE token-to-expert routers.

    A router consumes a flat token batch and returns three tensors that
    together describe the dispatch plan. Any module that exposes the
    three listed attributes + ``__call__`` signature satisfies the
    Protocol — no explicit subclassing needed.

    Attributes:
        n_experts: total number of routed experts the router can pick from
        topk:      number of experts to select per token

    The call contract:
        (topk_idx, topk_scores, scores) = router(x)

        x          -- (N, D) flattened input where N = B*T
        topk_idx   -- (N, K) int64, expert ids per token slot
        topk_scores-- (N, K) float, renormalized gating weights (sum=1 per row)
        scores     -- (N, n_experts) float, the full pre-top-K softmax
                       distribution (needed for aux-loss balance penalties)
    """

    n_experts: int
    topk: int

    def __call__(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


class BiasRouter(nn.Module):
    """DeepSeek-V3 aux-loss-free bias router.

    The gate weights that flow into the dispatch come from an unbiased
    softmax of a linear projection, so the gradient path is clean. A
    separate per-expert bias buffer is added only at top-K argmax time
    to steer under-used experts — never appearing in the weight gradient.
    The bias is adjusted externally during training (e.g., +δ on unused
    experts each step) to balance load without distorting the loss.

    Attributes:
        linear:      nn.Linear(dim, n_experts, bias=False) — the gate
        router_bias: (n_experts,) buffer, non-grad, hand-tuned for balance
        n_experts, topk: forwarded from cfg; expose for Protocol
    """

    def __init__(self, dim: int, n_experts: int, topk: int):
        super().__init__()
        self.n_experts = n_experts
        self.topk = topk
        self.linear = nn.Linear(dim, n_experts, bias=False)
        # Non-grad load-balance bias; training-time adjustable.
        self.register_buffer("router_bias", torch.zeros(n_experts))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route flat input tokens to expert slots.

        Args:
            x -- (N, D) flattened token batch.

        Returns:
            topk_idx    -- (N, K) expert assignments
            topk_scores -- (N, K) renormalized weights, sum=1 per row
            scores      -- (N, n_experts) full softmax (for balance loss)
        """
        logits = self.linear(x)
        scores = F.softmax(logits, dim=-1)
        _, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)
        topk_scores = scores.gather(-1, topk_idx)
        # .clamp(min=1e-9) guards against fp16/bf16 softmax underflow (P0.3).
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp(
            min=1e-9
        )
        return topk_idx, topk_scores, scores


def build_router(cfg: MythosConfig) -> RouterProtocol:
    """Construct the router module matching ``cfg``.

    Stage 4.3 (P4.3): currently only ``BiasRouter`` (DeepSeek-V3 style)
    is implemented. Structured as a factory so future routers (grouped,
    hash-based, learned gating variants) plug in without touching
    ``MoEFFN`` itself.
    """
    return BiasRouter(
        dim=cfg.dim,
        n_experts=cfg.moe.n_experts,
        topk=cfg.moe.n_experts_per_tok,
    )


# ---------------------------------------------------------------------------
# DeepSeek-style MoE FFN
# ---------------------------------------------------------------------------


class MoEFFN(nn.Module):
    """
    Fine-grained Mixture-of-Experts FFN (DeepSeekMoE, Dai et al., 2024).

    Two classes of experts:
    - Routed experts: n_experts small FFNs; each token activates top-K of them
      via a learned router. A per-expert bias on router logits is updated during
      training to keep load balanced across experts without distorting the loss.
    - Shared experts: n_shared_experts larger FFNs always activated for every token,
      absorbing common cross-domain patterns (syntax, basic reasoning) that would
      otherwise be redundantly learned by many routed experts.

    Total activated parameters per token ≈ topk/n_experts of routed + all shared,
    keeping compute sparse while the total parameter count stays large.
    """

    def __init__(self, cfg: MythosConfig, moe_balance_alpha: float = 0.0):
        """
        Args:
            cfg                -- MythosConfig; uses n_experts, n_shared_experts,
                                  n_experts_per_tok, dim, expert_dim.
            moe_balance_alpha  -- Coefficient for the expert-level balance loss
                                  (DeepSeekMoE §3.3).  Only applied when the
                                  caller opts in via ``return_balance_loss=True``
                                  in ``forward``.
        """
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_shared = cfg.n_shared_experts
        self.topk = cfg.n_experts_per_tok
        self.moe_balance_alpha = moe_balance_alpha

        # Stage 4.3: router is a swappable Protocol-compliant submodule.
        # Previously an inline nn.Linear + register_buffer pair; now built
        # via factory so alternative routers plug in without touching MoEFFN.
        # State-dict layout changed from router.weight / router_bias to
        # router.linear.weight / router.router_bias (documented break).
        self.router = build_router(cfg)

        self.routed_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.expert_dim, cfg=cfg) for _ in range(cfg.n_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [
                Expert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok, cfg=cfg)
                for _ in range(self.n_shared)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        return_balance_loss: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x                   -- input of shape (B, T, dim)
            return_balance_loss -- when True, also return the DeepSeekMoE §3.3
                                   expert-level balance loss scaled by
                                   ``self.moe_balance_alpha``.  Default ``False``
                                   preserves the original single-tensor return
                                   for all existing callers.

        Returns:
            If ``return_balance_loss`` is False (default): the (B, T, dim) output
            tensor only. Otherwise: a ``(output, balance_loss)`` tuple where
            ``balance_loss`` is a zero-dim scalar tensor.
        """
        B, T, D = x.shape
        flat = x.view(B * T, D)

        # Stage 4.3: delegate routing to the Router Protocol instance.
        topk_idx, topk_scores, scores = self.router(flat)

        # P3.2: vectorized routed-expert dispatch.
        # Flatten the (token, k_slot) pairs so every top-K assignment is one
        # entry in three parallel tensors. One expert at a time, we gather all
        # of its tokens in a single matmul instead of iterating per k-slot.
        N = flat.shape[0]
        K = self.topk
        flat_idx = topk_idx.reshape(-1)          # (N*K,)  expert id per assignment
        flat_weights = topk_scores.reshape(-1)   # (N*K,)  router weight per assignment
        flat_tokens = (
            torch.arange(N, device=flat.device)
            .unsqueeze(1)
            .expand(-1, K)
            .reshape(-1)
        )  # (N*K,)  token index per assignment

        out = torch.zeros_like(flat)
        # One sync to read bincount into Python — amortized across all experts,
        # far cheaper than per-expert .any() checks inside the inner loop.
        counts = torch.bincount(flat_idx, minlength=self.n_experts).tolist()
        for eid in range(self.n_experts):
            if counts[eid] == 0:
                continue
            mask = flat_idx == eid
            token_selector = flat_tokens[mask]          # (M,)
            weights = flat_weights[mask].unsqueeze(-1)  # (M, 1)
            expert_out = self.routed_experts[eid](flat[token_selector])  # (M, D)
            out.index_add_(0, token_selector, expert_out * weights)

        # shared experts always fire for every token
        for shared in self.shared_experts:
            out = out + shared(flat)

        out = out.view(B, T, D)

        if not return_balance_loss:
            return out

        balance_loss = self._balance_loss(topk_idx, scores, N)
        return out, balance_loss

    def _balance_loss(
        self,
        topk_idx: torch.Tensor,
        scores: torch.Tensor,
        n_tokens: int,
    ) -> torch.Tensor:
        """Expert-level balance loss from DeepSeekMoE §3.3.

        Penalises routing imbalance so the model spreads tokens evenly across
        experts.  Only the soft-score term ``P_i`` carries a gradient; the hard
        count term ``f_i`` is non-differentiable and weights ``P_i``.

        ::

            f_i = (N_r / (K × T)) × #{tokens routed to expert i}
            P_i = (1/T) Σ_t scores[t, i]
            L   = alpha × Σ_i  f_i · P_i

        Args:
            topk_idx: ``(T, K)`` int64 — expert ids selected per token (where
                      ``T = B*T`` is the flattened token count).
            scores:   ``(T, N_r)`` float — full softmax distribution over
                      experts (pre top-K), used for ``P_i``.
            n_tokens: Total flattened token count ``B*T``.

        Returns:
            Zero-dim scalar tensor on the same device/dtype as ``scores``.
        """
        Nr = self.n_experts
        K = self.topk

        counts = torch.bincount(topk_idx.reshape(-1), minlength=Nr).to(scores.dtype)
        f = counts * (Nr / (K * n_tokens))  # [Nr] — non-differentiable
        P = scores.mean(dim=0)              # [Nr] — gradient flows here

        return self.moe_balance_alpha * (f * P).sum()
