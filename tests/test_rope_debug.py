"""RoPE invariant checks for precompute_rope_freqs and apply_rope."""

import torch

from open_mythos.main import apply_rope, precompute_rope_freqs

DIM = 8
MAX_LEN = 6
THETA = 500000.0


def test_rope_freqs_angles_match_position_times_base() -> None:
    freqs = precompute_rope_freqs(dim=DIM, max_len=MAX_LEN, theta=THETA)
    assert freqs.shape == (MAX_LEN, DIM // 2)

    torch.testing.assert_close(
        freqs.abs(),
        torch.ones(MAX_LEN, DIM // 2),
        atol=1e-5,
        rtol=1e-5,
    )

    # torch.angle() returns values in [-π, π], while t * base may wrap past 2π
    # for low-index dimensions at higher positions. Compare modulo 2π so the
    # test asserts the underlying phase equivalence, not a raw numeric match.
    def wrap(a: torch.Tensor) -> torch.Tensor:
        return (a + torch.pi) % (2 * torch.pi) - torch.pi

    base = 1.0 / (THETA ** (torch.arange(0, DIM, 2, dtype=torch.float32) / DIM))
    for t in range(MAX_LEN):
        expected_angles = t * base
        actual_angles = freqs[t].angle()
        torch.testing.assert_close(
            wrap(actual_angles), wrap(expected_angles), atol=1e-5, rtol=1e-5
        )


def test_rope_freqs_position_zero_is_identity_phasor() -> None:
    freqs = precompute_rope_freqs(dim=DIM, max_len=MAX_LEN, theta=THETA)
    torch.testing.assert_close(
        freqs[0].abs(), torch.ones(DIM // 2), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        freqs[0].angle(), torch.zeros(DIM // 2), atol=1e-5, rtol=1e-5
    )


def test_apply_rope_preserves_shape_and_dtype() -> None:
    torch.manual_seed(42)
    freqs = precompute_rope_freqs(dim=DIM, max_len=MAX_LEN, theta=THETA)

    B, T, H = 2, MAX_LEN, 3
    x = torch.randn(B, T, H, DIM)
    out = apply_rope(x, freqs)
    assert out.shape == x.shape

    x_half = x.half()
    out_half = apply_rope(x_half, freqs)
    assert out_half.dtype == torch.float16


def test_apply_rope_preserves_vector_norms() -> None:
    torch.manual_seed(42)
    freqs = precompute_rope_freqs(dim=DIM, max_len=MAX_LEN, theta=THETA)

    B, T, H = 2, MAX_LEN, 3
    x = torch.randn(B, T, H, DIM)
    out = apply_rope(x, freqs)

    norms_in = x.norm(dim=-1)
    norms_out = out.norm(dim=-1)
    torch.testing.assert_close(norms_in, norms_out, atol=1e-5, rtol=1e-5)


def test_apply_rope_at_position_zero_is_identity() -> None:
    torch.manual_seed(42)
    freqs = precompute_rope_freqs(dim=DIM, max_len=MAX_LEN, theta=THETA)

    x1 = torch.randn(1, 1, 2, DIM)
    out1 = apply_rope(x1, freqs[:1])
    torch.testing.assert_close(x1, out1, atol=1e-6, rtol=1e-6)


def test_apply_rope_different_positions_produce_different_rotations() -> None:
    freqs = precompute_rope_freqs(dim=DIM, max_len=MAX_LEN, theta=THETA)

    x2 = torch.ones(1, MAX_LEN, 1, DIM)
    out2 = apply_rope(x2, freqs)

    # Every non-zero position must differ from the identity (pos 0) output.
    for t in range(1, MAX_LEN):
        assert not torch.allclose(out2[0, t, 0], out2[0, 0, 0], atol=1e-5)


def test_apply_rope_is_invertible_via_conjugate_freqs() -> None:
    torch.manual_seed(42)
    freqs = precompute_rope_freqs(dim=DIM, max_len=MAX_LEN, theta=THETA)

    B, T, H = 2, MAX_LEN, 3
    x3 = torch.randn(B, T, H, DIM)
    rotated = apply_rope(x3, freqs)

    xc = torch.view_as_complex(rotated.float().reshape(*rotated.shape[:-1], -1, 2))
    inv_freqs = freqs.conj()
    recovered = (
        torch.view_as_real(xc * inv_freqs.unsqueeze(0).unsqueeze(2))
        .flatten(-2)
        .to(x3.dtype)
    )
    torch.testing.assert_close(x3, recovered, atol=1e-5, rtol=1e-5)


def test_apply_rope_start_pos_changes_encoding() -> None:
    """Decode-step regression: freqs[start_pos:start_pos+1] must differ from freqs[:1]."""
    torch.manual_seed(42)
    freqs = precompute_rope_freqs(dim=DIM, max_len=MAX_LEN, theta=THETA)

    prompt_len = 4
    decode_token = torch.randn(1, 1, 1, DIM)

    out_buggy = apply_rope(decode_token, freqs[:1])
    out_fixed = apply_rope(decode_token, freqs[prompt_len : prompt_len + 1])

    assert not torch.allclose(out_buggy, out_fixed)


def test_apply_rope_inner_product_depends_only_on_relative_offset() -> None:
    torch.manual_seed(42)
    dim = 16
    max_len = 32
    freqs_big = precompute_rope_freqs(dim=dim, max_len=max_len, theta=THETA)
    q = torch.randn(1, 1, 1, dim)
    k = torch.randn(1, 1, 1, dim)

    def rope_at(tensor: torch.Tensor, pos: int) -> torch.Tensor:
        seq = torch.zeros(1, pos + 1, 1, dim)
        seq[0, pos] = tensor[0, 0]
        return apply_rope(seq, freqs_big[: pos + 1])[:, pos : pos + 1]

    dot_3_9 = (rope_at(q, 3) * rope_at(k, 9)).sum()
    dot_1_7 = (rope_at(q, 1) * rope_at(k, 7)).sum()
    dot_0_6 = (rope_at(q, 0) * rope_at(k, 6)).sum()

    torch.testing.assert_close(dot_3_9, dot_1_7, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dot_1_7, dot_0_6, atol=1e-5, rtol=1e-5)
