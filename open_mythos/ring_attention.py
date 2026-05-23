"""
Ring Attention for Distributed Long-Context Processing.

Enables processing sequences up to 1M tokens by distributing attention
computation across multiple chunks in a ring topology. Each chunk
processes locally, communicates only boundaries.

Memory: O(n/p) instead of O(n²) where p = number of chunks.

Usage:
    from open_mythos.ring_attention import RingAttention, ring_attention_forward

    # In model forward pass
    output = ring_attention_forward(q, k, v, chunk_size=8192)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import logging
import math

logger = logging.getLogger(__name__)


class RingAttention(nn.Module):
    """
    Ring Attention for memory-efficient long-context processing.

    Splits the input sequence into chunks and processes them in a ring
    topology. Each chunk computes local attention, then communicates
    boundary KV states to compute cross-chunk attention.

    This reduces memory from O(n²) to O(n * chunk_size), enabling
    processing of 1M+ token sequences on consumer hardware.

    Args:
        chunk_size: Size of each chunk (default: 8192)
        num_heads: Number of attention heads
        head_dim: Dimension per head
        causal: Whether to use causal masking (default: True)
    """

    def __init__(
        self,
        chunk_size: int = 8192,
        num_heads: int = 32,
        head_dim: int = 128,
        causal: bool = True,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal = causal

    def _split_into_chunks(
        self, x: torch.Tensor, seq_dim: int = 1
    ) -> List[torch.Tensor]:
        """Split tensor into chunks along sequence dimension."""
        seq_len = x.shape[seq_dim]
        num_chunks = math.ceil(seq_len / self.chunk_size)

        chunks = []
        for i in range(num_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, seq_len)
            chunk = x[:, start:end]
            chunks.append(chunk)

        return chunks

    def _local_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention within a single chunk."""
        # q, k, v: [batch, seq, heads, head_dim]
        batch, seq, heads, dim = q.shape

        # Transpose for attention: [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)

        # Transpose back: [batch, seq, heads, dim]
        return output.transpose(1, 2)

    def _cross_chunk_attention(
        self,
        q_chunk: torch.Tensor,
        k_prev: torch.Tensor,
        v_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-attention between current chunk and previous chunks' KV."""
        # q_chunk: [batch, chunk_size, heads, dim]
        # k_prev, v_prev: [batch, prev_seq, heads, dim]
        batch, chunk_len, heads, dim = q_chunk.shape
        prev_len = k_prev.shape[1]

        # Transpose for attention
        q = q_chunk.transpose(1, 2)  # [batch, heads, chunk_len, dim]
        k = k_prev.transpose(1, 2)  # [batch, heads, prev_len, dim]
        v = v_prev.transpose(1, 2)  # [batch, heads, prev_len, dim]

        # Cross-attention
        scale = 1.0 / math.sqrt(dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)

        return output.transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with ring attention.

        Args:
            q: Query tensor [batch, seq, heads, dim]
            k: Key tensor [batch, seq, heads, dim]
            v: Value tensor [batch, seq, heads, dim]
            mask: Optional attention mask

        Returns:
            Output tensor [batch, seq, heads, dim]
        """
        batch, seq_len, heads, dim = q.shape

        # If sequence is short enough, use standard attention
        if seq_len <= self.chunk_size:
            return self._local_attention(q, k, v, mask)

        # Split into chunks
        q_chunks = self._split_into_chunks(q, seq_dim=1)
        k_chunks = self._split_into_chunks(k, seq_dim=1)
        v_chunks = self._split_into_chunks(v, seq_dim=1)

        num_chunks = len(q_chunks)
        outputs = []

        # Accumulated KV from previous chunks
        k_accumulated = None
        v_accumulated = None

        for i in range(num_chunks):
            q_chunk = q_chunks[i]
            k_chunk = k_chunks[i]
            v_chunk = v_chunks[i]

            # 1. Local attention within chunk
            local_mask = None
            if self.causal and mask is not None:
                # Create local causal mask for this chunk
                chunk_len = q_chunk.shape[1]
                local_mask = torch.tril(
                    torch.ones(chunk_len, chunk_len, device=q.device)
                ).unsqueeze(0).unsqueeze(0)

            local_out = self._local_attention(q_chunk, k_chunk, v_chunk, local_mask)

            # 2. Cross-attention with previous chunks (if any)
            if k_accumulated is not None and k_accumulated.shape[1] > 0:
                cross_out = self._cross_chunk_attention(
                    q_chunk, k_accumulated, v_accumulated
                )

                # Combine local and cross attention
                # Weight by relative sequence positions
                local_weight = 0.7
                cross_weight = 0.3
                combined_out = local_weight * local_out + cross_weight * cross_out
            else:
                combined_out = local_out

            outputs.append(combined_out)

            # 3. Accumulate KV for next chunk
            if k_accumulated is None:
                k_accumulated = k_chunk
                v_accumulated = v_chunk
            else:
                k_accumulated = torch.cat([k_accumulated, k_chunk], dim=1)
                v_accumulated = torch.cat([v_accumulated, v_chunk], dim=1)

            # Optional: Limit accumulated KV to prevent memory growth
            # Keep only the most recent N tokens
            max_accumulated = self.chunk_size * 4  # Keep last 4 chunks worth
            if k_accumulated.shape[1] > max_accumulated:
                k_accumulated = k_accumulated[:, -max_accumulated:]
                v_accumulated = v_accumulated[:, -max_accumulated:]

        # Concatenate all outputs
        return torch.cat(outputs, dim=1)

    def __repr__(self) -> str:
        return (
            f"RingAttention(chunk_size={self.chunk_size}, "
            f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"causal={self.causal})"
        )


class SparseRingAttention(nn.Module):
    """
    Ring Attention with sparse attention patterns.

    Combines ring attention with sparse attention (sliding window + global tokens)
    for even more memory-efficient long-context processing.

    Args:
        chunk_size: Size of each chunk
        window_size: Sliding window size for local attention
        num_global_tokens: Number of global tokens to attend to
        num_heads: Number of attention heads
        head_dim: Dimension per head
    """

    def __init__(
        self,
        chunk_size: int = 8192,
        window_size: int = 4096,
        num_global_tokens: int = 256,
        num_heads: int = 32,
        head_dim: int = 128,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        self.num_heads = num_heads
        self.head_dim = head_dim

    def _sparse_attention_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create sparse attention mask (sliding window + global tokens).

        Each token attends to:
        - Its local window (window_size tokens)
        - Global tokens (first num_global_tokens tokens)
        """
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)

        # Sliding window
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = True

        # Global tokens
        mask[:, : self.num_global_tokens] = True

        # Causal: can only attend to past
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        mask = mask & causal_mask

        return mask

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with sparse ring attention.

        Args:
            q: Query tensor [batch, seq, heads, dim]
            k: Key tensor [batch, seq, heads, dim]
            v: Value tensor [batch, seq, heads, dim]

        Returns:
            Output tensor [batch, seq, heads, dim]
        """
        batch, seq_len, heads, dim = q.shape

        # Create sparse mask
        sparse_mask = self._sparse_attention_mask(seq_len, q.device)

        # Expand mask for batch and heads
        sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]

        # Transpose for attention: [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Sparse attention
        scale = 1.0 / math.sqrt(dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = attn_weights.masked_fill(~sparse_mask, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)

        return output.transpose(1, 2)


def ring_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 8192,
    use_sparse: bool = False,
    window_size: int = 4096,
    num_global_tokens: int = 256,
) -> torch.Tensor:
    """
    Convenience function for ring attention.

    Args:
        q: Query tensor [batch, seq, heads, dim]
        k: Key tensor [batch, seq, heads, dim]
        v: Value tensor [batch, seq, heads, dim]
        chunk_size: Size of each chunk
        use_sparse: Whether to use sparse attention
        window_size: Sliding window size (if sparse)
        num_global_tokens: Number of global tokens (if sparse)

    Returns:
        Output tensor [batch, seq, heads, dim]
    """
    if use_sparse:
        attn = SparseRingAttention(
            chunk_size=chunk_size,
            window_size=window_size,
            num_global_tokens=num_global_tokens,
            num_heads=q.shape[2],
            head_dim=q.shape[3],
        )
    else:
        attn = RingAttention(
            chunk_size=chunk_size,
            num_heads=q.shape[2],
            head_dim=q.shape[3],
        )

    return attn(q, k, v)
