import torch


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def precompute_rope_freqs(
    dim: int, max_len: int, theta: float = 500000.0
) -> torch.Tensor:
    """
    Precompute complex-valued RoPE rotation matrices for positions 0..max_len-1.

    Each position gets a complex phasor e^{i·m·θ_k} for each frequency pair k.
    Stored as a complex tensor so that rotation is a single pointwise multiply.

    Args:
        dim     -- head dimension (must be even); frequencies are computed for dim//2 pairs
        max_len -- maximum sequence length to precompute
        theta   -- RoPE base (higher = slower frequency decay; 500k is the LLaMA-3 default)

    Returns:
        complex64 tensor of shape (max_len, dim//2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to query or key tensors.

    Interprets each pair of adjacent features as a 2D complex number and
    multiplies by the precomputed phasor for that position, rotating the
    representation in the complex plane without changing its norm.

    Args:
        x         -- tensor of shape (B, T, H, head_dim); head_dim must be even
        freqs_cis -- precomputed complex frequencies of shape (T, head_dim//2),
                     already sliced to exactly the positions being processed
                     (caller is responsible for correct start_pos offset)

    Returns:
        Rotated tensor of the same shape and dtype as x
    """
    # P3.4: preserve input dtype. torch.view_as_complex requires float32 or
    # float64, so only upcast when the input is a lower-precision dtype (bf16/
    # fp16). Float inputs skip the cast entirely.
    orig_dtype = x.dtype
    if x.dtype in (torch.float32, torch.float64):
        xc = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
    else:
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(xc * freqs_cis.unsqueeze(0).unsqueeze(2)).flatten(-2)
    return out.to(orig_dtype)


# ---------------------------------------------------------------------------
# Loop-index RoPE (differentiates recurrent block across iterations)
# ---------------------------------------------------------------------------


def loop_index_embedding(
    h: torch.Tensor, loop_t: int, loop_dim: int, theta: float = 10000.0
) -> torch.Tensor:
    """
    Inject a sinusoidal loop-index signal into the first loop_dim channels of h.

    Analogous to RoPE for sequence position, but applied over recurrence depth
    instead of token position. Without this, the shared recurrent block weights
    must handle both early-stage pattern-matching and late-stage refinement with
    no signal distinguishing which loop they are on. Adding the loop index lets
    the same parameters implement functionally distinct operations per iteration.

    Args:
        h        -- hidden state tensor of shape (B, T, dim)
        loop_t   -- current loop iteration index (0-based)
        loop_dim -- number of leading channels to receive the embedding (must be even)
        theta    -- sinusoidal base frequency

    Returns:
        h with a sinusoidal bias added to its first loop_dim channels; same shape
    """
    # loop_dim must be even: the embedding concatenates sin + cos halves, and an
    # odd loop_dim would truncate mid-cosine producing an asymmetric (sin-heavy)
    # encoding that is neither standard sinusoidal nor RoPE.
    if loop_dim % 2 != 0:
        raise ValueError(
            f"loop_dim must be even, got {loop_dim} "
            "(sin/cos halves would split asymmetrically)"
        )
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, loop_dim, 2, device=h.device, dtype=h.dtype) / loop_dim)
    )
    angles = loop_t * freqs  # (loop_dim//2,)
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
    emb_full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    emb_full[:loop_dim] = emb
    return h + emb_full.unsqueeze(0).unsqueeze(0)
