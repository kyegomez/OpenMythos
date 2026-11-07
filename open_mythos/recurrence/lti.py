import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# LTI-stable injection parameters  (spectral radius < 1 by construction)
# ---------------------------------------------------------------------------


class LTIInjection(nn.Module):
    """
    Stable input injection for the recurrent update rule (Parcae, Prairie et al., 2026).

    The recurrent hidden state evolves as:
        h_{t+1} = A · h_t  +  B · e  +  Transformer(h_t, e)

    where e is the encoded input injected at every loop step to prevent drift.
    Without constraints, A can develop spectral radius ≥ 1, causing the hidden
    state to explode across loop iterations and destabilize training.

    This class guarantees ρ(A) < 1 by construction via a ZOH discretization:
        A_continuous = Diag(-exp(log_A))       always negative diagonal
        A_discrete   = exp(Δt · A_continuous)  element-wise, values in (0, 1)

    where log_A and log_dt are learned parameters and exp ensures positivity.
    This makes looped model training robust to hyperparameter choices and stable
    even at high learning rates.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim -- hidden state dimension; one scalar per channel for A and B
        """
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))  # log of A_continuous magnitude
        self.log_dt = nn.Parameter(torch.zeros(1))  # log of discretization step Δt
        self.B = nn.Parameter(torch.ones(dim) * 0.1)

    def get_A(self) -> torch.Tensor:
        """
        Compute the discretized diagonal state matrix A_discrete.

        Returns:
            1-D tensor of shape (dim,) with all values strictly in (0, 1),
            guaranteeing ρ(A) < 1 regardless of learned parameter values.
        """
        # Compute in log space to avoid 0 * inf = NaN when log_dt → -∞, log_A → +∞.
        # dt * A_c = -exp(log_dt) * exp(log_A) = -exp(log_dt + log_A)
        # Clamp keeps the product finite in float32 for any gradient step size.
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

    def boundary_saturation(self, margin: float = 1.0) -> dict:
        """Report the fraction of channels saturating the ±20 clamp boundary.

        The clamp in :meth:`get_A` silently pins ``log_dt + log_A`` to the
        range ``[-20, 20]`` to keep ``exp(exp(·))`` finite. Channels that hit
        a boundary collapse to a degenerate A:

        - ``log_dt + log_A → +20``: ``A ≈ exp(-exp(20)) ≈ 0`` — the channel's
          state contribution decays to zero every step (effectively "dead").
        - ``log_dt + log_A → -20``: ``A ≈ exp(-exp(-20)) ≈ 1`` — the channel's
          state never decays (effectively static; no LTI dynamics).

        Either is a silent failure mode during training. This method exposes
        the count so a caller (e.g. training-loop periodic logger) can flag
        saturation above a threshold.

        Args:
            margin -- how close to ±20 counts as "near-boundary" (default 1.0
                      → channels within 1 unit of the clamp).

        Returns:
            Dict with three keys:
              - ``low_fraction``:  fraction of channels where A ≈ 1 (static)
              - ``high_fraction``: fraction of channels where A ≈ 0 (dead)
              - ``total``:         sum of the two
        """
        with torch.no_grad():
            combined = self.log_dt + self.log_A  # broadcast scalar + (dim,)
            low = float((combined < -20.0 + margin).float().mean().item())
            high = float((combined > 20.0 - margin).float().mean().item())
        return {
            "low_fraction": low,
            "high_fraction": high,
            "total": low + high,
        }

    def forward(
        self, h: torch.Tensor, e: torch.Tensor, transformer_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute h_{t+1} = A·h_t + B·e + transformer_out.

        Args:
            h               -- current hidden state (B, T, dim)
            e               -- encoded input from Prelude, frozen across loops (B, T, dim)
            transformer_out -- output of the recurrent TransformerBlock at this step (B, T, dim)

        Returns:
            Updated hidden state of shape (B, T, dim)
        """
        A = self.get_A()
        return A * h + self.B * e + transformer_out
