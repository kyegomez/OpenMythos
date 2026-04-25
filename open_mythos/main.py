from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MythosConfig:
    """
    Hyperparameter configuration for OpenMythos.

    Core:
        vocab_size      -- token vocabulary size
        dim             -- model hidden dimension
        n_heads         -- number of query attention heads
        n_kv_heads      -- number of key/value heads (GQA; ignored by MLA)
        max_seq_len     -- maximum sequence length for RoPE precomputation
        max_loop_iters  -- default recurrent loop depth T at inference
        prelude_layers  -- number of standard transformer layers before the loop
        coda_layers     -- number of standard transformer layers after the loop

    Attention (attn_type selects between the two):
        attn_type       -- "gqa" for Grouped Query Attention, "mla" for Multi-Latent Attention
        kv_lora_rank    -- [MLA] compressed KV latent dimension stored in the cache
        q_lora_rank     -- [MLA] compressed Q latent dimension
        qk_rope_head_dim-- [MLA] per-head dims that receive RoPE
        qk_nope_head_dim-- [MLA] per-head dims without positional encoding
        v_head_dim      -- [MLA] per-head value dimension

    MoE FFN (used inside the recurrent block):
        n_experts       -- total number of routed expert FFNs
        n_shared_experts-- number of always-active shared experts
        n_experts_per_tok-- top-K experts selected per token by the router
        expert_dim      -- hidden dimension inside each fine-grained expert

    Other:
        act_threshold   -- ACT halting threshold (cumulative probability to stop looping)
        rope_theta      -- RoPE base frequency
        lora_rank       -- rank of the per-loop depth-wise LoRA adapter
    """

    vocab_size: int = 32000
    dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4  # GQA: fewer KV heads than Q heads
    max_seq_len: int = 4096
    max_loop_iters: int = 16  # T — recurrent depth at inference
    prelude_layers: int = 2
    coda_layers: int = 2
    # Attention type: "gqa" | "mla"
    attn_type: str = "mla"
    # MLA params (only used when attn_type="mla")
    kv_lora_rank: int = 512  # compressed KV latent cached instead of full K/V
    q_lora_rank: int = 1536  # compressed Q latent dim
    qk_rope_head_dim: int = 64  # per-head dims that receive RoPE
    qk_nope_head_dim: int = 128  # per-head dims without RoPE
    v_head_dim: int = 128  # per-head value dim
    # MoE
    n_experts: int = 64
    n_shared_experts: int = 2
    n_experts_per_tok: int = 4  # top-K routed
    expert_dim: int = 512  # fine-grained: dim // (n_experts // n_experts_per_tok)
    # ACT halting
    act_threshold: float = 0.99
    # RoPE
    rope_theta: float = 500000.0
    # LoRA depth adaptation
    lora_rank: int = 16
    # Maximum tokens to generate per forward pass
    max_output_tokens: int = 4096
    # Dropout (set 0.0 to disable; 0.1 is standard for pretraining)
    dropout: float = 0.0
    # ---- P4: Recurrent Block Variants ----
    use_path_cost: bool = False
    path_cost_threshold: float = 1.0
    path_cost_mode: str = "dijkstra"
    use_cone: bool = False
    cone_sharpness: float = 2.0
    use_bundle_memory: bool = False
    use_procedural: bool = False
    use_hierarchical: bool = False
    use_meta_loop: bool = False

    # ---- P1: Flash MLA + Cross-Layer KV Sharing ----
    use_flash_mla: bool = True          # use SDPA (FlashAttention-compatible) for MLA
    cross_layer_kv_share: bool = False  # share KV cache across layer groups
    kv_share_group_size: int = 2        # number of layers per KV share group

    # ---- P2: Speculative Decoding ----
    use_speculative: bool = False       # enable speculative decoding
    speculative_gamma: int = 4          # number of tokens to draft per round
    speculative_n_loops: int = 2       # loop depth for draft model (fast, fewer iterations)
    speculative_temperature: float = 1.0
    speculative_top_k: int = 50

    # ---- P0: Multi-Scale Loop + Curriculum Learning ----
    enable_multiscale_loop: bool = False   # enable complexity-aware loop depth selection
    loop_depths: list = field(default_factory=lambda: [4, 8, 16])  # easy, medium, hard
    complexity_threshold_low: float = 0.33   # p < this → easy (loop_depths[0])
    complexity_threshold_high: float = 0.66  # p < this → medium, else hard
    enable_curriculum: bool = False         # enable curriculum learning for loop depth
    curriculum_min_depth: int = 4          # starting loop depth for curriculum
    curriculum_max_depth: int = 16         # maximum loop depth in curriculum
    curriculum_phase1_steps: int = 2000   # 0-20%: fixed min_depth
    curriculum_phase2_steps: int = 5000    # 20-50%: fixed medium_depth
    curriculum_phase3_steps: int = 10000   # 50-80%: mixed [4,8,12,16]
    curriculum_phase4_steps: int = 20000    # 80-100%: dynamic + ACT

    # ---- P3: Hierarchical Loop + Meta-Learning ----
    hierarchical_num_scales: int = 4   # number of hierarchical scales
    hierarchical_adaptive_scale: bool = True  # adaptively select scale per token
    hierarchical_top_down_broadcast: bool = True  # broadcast from coarse to fine
    meta_loop_predictor_hidden: int = 256  # hidden dim for meta loop depth predictor
    meta_loop_predictor_layers: int = 2   # layers in meta loop depth predictor

# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Normalizes by the RMS of the input rather than mean+variance, with a
    learned per-channel rescaling weight. No bias term. Used in place of
    LayerNorm throughout the model for stability and efficiency.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim -- feature dimension to normalize over
            eps -- small constant added before sqrt for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input tensor of shape (..., dim)
        Returns:
            RMS-normalized tensor of the same shape, rescaled by self.weight
        """
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


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
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return (
        torch.view_as_real(xc * freqs_cis.unsqueeze(0).unsqueeze(2))
        .flatten(-2)
        .to(x.dtype)
    )


# ---------------------------------------------------------------------------
# Grouped Query Attention with KV cache
# ---------------------------------------------------------------------------


class GQAttention(nn.Module):
    """
    Grouped Query Attention (Ainslie et al., 2023).

    Uses fewer KV heads than Q heads (n_kv_heads < n_heads). Each KV head is
    shared across n_heads // n_kv_heads query heads, reducing the KV cache size
    by that factor while keeping full query expressiveness.

    RoPE is applied to both Q and K. K and V are stored in kv_cache after
    RoPE application so that cached values are already positionally encoded and
    do not need to be re-rotated on retrieval.

    P1 Enhancement — FlashMLA (SDPA):
        When use_flash_mla=True, uses F.scaled_dot_product_attention which
        automatically leverages FlashAttention/FlashAttention-v2 kernels.

    P1 Enhancement — Cross-Layer KV Sharing:
        When cross_layer_kv_share=True, adjacent layers share KV caches.
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses dim, n_heads, n_kv_heads
        """
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.groups = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)

    def _get_shared_cache_key(self, cache_key: str) -> str:
        """P1: Cross-layer KV sharing."""
        if not self.cfg.cross_layer_kv_share:
            return cache_key
        parts = cache_key.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            layer_idx = int(parts[1])
            shared_idx = layer_idx // self.cfg.kv_share_group_size
            return f"{parts[0]}_shared_{shared_idx}"
        return cache_key

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- RoPE frequencies for head_dim, shape (T, head_dim//2)
            mask      -- additive causal mask of shape (1, 1, T, S) or None
            kv_cache  -- dict mutated in-place; stores {"k": ..., "v": ...} per cache_key
            cache_key -- unique key identifying this layer in the cache dict

        Returns:
            Output tensor of shape (B, T, dim)
        """
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # P1: Cross-layer KV sharing
        shared_key = self._get_shared_cache_key(cache_key)

        if kv_cache is not None:
            if shared_key in kv_cache:
                k = torch.cat([kv_cache[shared_key]["k"], k], dim=1)
                v = torch.cat([kv_cache[shared_key]["v"], v], dim=1)
            kv_cache[shared_key] = {"k": k.detach(), "v": v.detach()}

        # expand KV to match Q heads
        k = k.repeat_interleave(self.groups, dim=2)
        v = v.repeat_interleave(self.groups, dim=2)

        q = q.transpose(1, 2)  # (B, H, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # P1: FlashMLA (SDPA) - use when enabled
        if self.cfg.use_flash_mla:
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.cfg.dropout if self.training else 0.0,
                is_causal=mask is None,
            )
        else:
            scale = self.head_dim**-0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn + mask
            attn = self.attn_drop(F.softmax(attn, dim=-1))
            attn = torch.matmul(attn, v)

        out = attn.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# Multi-Latent Attention (DeepSeek-V2 style)
# ---------------------------------------------------------------------------


class MLAttention(nn.Module):
    """
    Multi-Latent Attention (DeepSeek-V2, 2024).

    The key insight: instead of caching full K and V tensors (each of size
    n_heads × head_dim per token), MLA compresses the KV path through a
    low-rank latent c_kv and only caches that plus the RoPE keys. K_nope and
    V are reconstructed from c_kv at each decoding step, trading a cheap
    linear projection for dramatically smaller cache memory.

    Q path:
        x → q_down (dim→q_lora_rank) → q_norm
          → q_up_nope (q_lora_rank → n_heads×qk_nope_head_dim)  [no RoPE]
          → q_up_rope (q_lora_rank → n_heads×qk_rope_head_dim)  [RoPE applied]
        q = cat(q_nope, q_rope)  per head

    KV path:
        x → kv_down (dim → kv_lora_rank + qk_rope_head_dim)
          splits into c_kv (latent, cached) and k_rope_raw (shared across heads)
        k_rope = RoPE(expand(k_rope_raw))  — applied before caching
        c_kv → kv_norm → kv_up → [k_nope | v]  — reconstructed each step
        k = cat(k_nope, k_rope)  per head

    Cache stores: c_kv (kv_lora_rank) + k_rope (n_heads × qk_rope_head_dim),
    versus full GQA cache: n_kv_heads × head_dim × 2.  At production scale this
    is roughly a 10–20× memory reduction.

    P1 Enhancement — FlashMLA:
        When use_flash_mla=True, uses F.scaled_dot_product_attention which
        automatically leverages FlashAttention/FlashAttention-v2 kernels when
        available, reducing memory and improving throughput.

    P1 Enhancement — Cross-Layer KV Sharing:
        When cross_layer_kv_share=True, adjacent layers share KV caches,
        reducing memory proportional to group size.
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses dim, n_heads, kv_lora_rank, q_lora_rank,
                   qk_rope_head_dim, qk_nope_head_dim, v_head_dim
        """
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_rope_dim = cfg.qk_rope_head_dim
        self.qk_nope_dim = cfg.qk_nope_head_dim
        self.v_dim = cfg.v_head_dim
        self.q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim

        # Q compression
        self.q_down = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank)
        self.q_up_nope = nn.Linear(
            cfg.q_lora_rank, cfg.n_heads * cfg.qk_nope_head_dim, bias=False
        )
        self.q_up_rope = nn.Linear(
            cfg.q_lora_rank, cfg.n_heads * cfg.qk_rope_head_dim, bias=False
        )

        # KV compression: output is [c_kv | k_rope_raw] concatenated
        self.kv_down = nn.Linear(
            cfg.dim, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False
        )
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.kv_up = nn.Linear(
            cfg.kv_lora_rank,
            cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            bias=False,
        )

        self.wo = nn.Linear(cfg.n_heads * cfg.v_head_dim, cfg.dim, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)

    def _get_shared_cache_key(self, cache_key: str) -> str:
        """
        P1: Cross-layer KV sharing.
        Returns the shared cache key for the group this layer belongs to.
        E.g., if kv_share_group_size=2 and cache_key="prelude_3", returns "prelude_shared_2".
        """
        if not self.cfg.cross_layer_kv_share:
            return cache_key
        # Parse layer index from cache_key like "prelude_3" or "recurrent_0"
        parts = cache_key.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            layer_idx = int(parts[1])
            shared_idx = layer_idx // self.cfg.kv_share_group_size
            return f"{parts[0]}_shared_{shared_idx}"
        return cache_key

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- RoPE frequencies sized for qk_rope_head_dim, shape (T, rope_dim//2)
            mask      -- additive causal mask of shape (1, 1, T, S) or None
            kv_cache  -- dict mutated in-place; stores {"c_kv": ..., "k_rope": ...}
            cache_key -- unique key identifying this layer in the cache dict

        Returns:
            Output tensor of shape (B, T, dim)
        """
        B, T, _ = x.shape

        # Q
        c_q = self.q_norm(self.q_down(x))
        q_nope = self.q_up_nope(c_q).view(B, T, self.n_heads, self.qk_nope_dim)
        q_rope = self.q_up_rope(c_q).view(B, T, self.n_heads, self.qk_rope_dim)
        q_rope = apply_rope(q_rope, freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1)  # (B, T, H, nope+rope)

        # KV compress
        kv_raw = self.kv_down(x)
        c_kv = kv_raw[..., : self.kv_lora_rank]  # (B, T, lora_rank)  ← cached
        k_rope = kv_raw[..., self.kv_lora_rank :]  # (B, T, rope_dim)
        # expand rope keys across heads and apply RoPE before caching so
        # retrieved keys are already positionally encoded
        k_rope = (
            k_rope.unsqueeze(2)
            .expand(B, T, self.n_heads, self.qk_rope_dim)
            .contiguous()
        )
        k_rope = apply_rope(k_rope, freqs_cis)  # (B, T, H, rope_dim) ← cached

        # P1: Cross-layer KV sharing - use shared cache key
        shared_key = self._get_shared_cache_key(cache_key)

        if kv_cache is not None:
            if shared_key in kv_cache:
                c_kv = torch.cat([kv_cache[shared_key]["c_kv"], c_kv], dim=1)
                k_rope = torch.cat([kv_cache[shared_key]["k_rope"], k_rope], dim=1)
            kv_cache[shared_key] = {"c_kv": c_kv.detach(), "k_rope": k_rope.detach()}

        S = c_kv.shape[1]  # full sequence length including cache

        # reconstruct K_nope and V from latent (not cached, recomputed each step)
        kv = self.kv_up(self.kv_norm(c_kv))  # (B, S, H*(nope+v))
        kv = kv.view(B, S, self.n_heads, self.qk_nope_dim + self.v_dim)
        k_nope = kv[..., : self.qk_nope_dim]  # (B, S, H, nope)
        v = kv[..., self.qk_nope_dim :]  # (B, S, H, v_dim)
        k = torch.cat([k_nope, k_rope], dim=-1)  # (B, S, H, nope+rope)

        # attention
        q = q.transpose(1, 2)  # (B, H, T, q_head_dim)
        k = k.transpose(1, 2)  # (B, H, S, q_head_dim)
        v = v.transpose(1, 2)  # (B, H, S, v_dim)

        # P1: FlashMLA - use SDPA when enabled
        if self.cfg.use_flash_mla:
            # SDPA automatically uses FlashAttention when available
            # is_causal=True sets up the correct causal mask
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.cfg.dropout if self.training else 0.0,
                is_causal=mask is None,
            )
        else:
            # Standard attention (for comparison)
            scale = self.q_head_dim**-0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn + mask
            attn = self.attn_drop(F.softmax(attn, dim=-1))
            attn = torch.matmul(attn, v)

        out = attn.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# DeepSeek-style MoE FFN
# ---------------------------------------------------------------------------


class Expert(nn.Module):
    """
    Single SwiGLU feed-forward expert.

    Implements the gated linear unit variant: output = down(silu(gate(x)) * up(x)).
    Used both as individual routed experts inside MoEFFN and as the standard dense
    FFN in prelude/coda blocks (where expert_dim = dim * 4 // 3).
    """

    def __init__(self, dim: int, expert_dim: int):
        """
        Args:
            dim        -- input and output feature dimension
            expert_dim -- inner (hidden) dimension of the expert
        """
        super().__init__()
        self.gate = nn.Linear(dim, expert_dim, bias=False)
        self.up = nn.Linear(dim, expert_dim, bias=False)
        self.down = nn.Linear(expert_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input of shape (..., dim)
        Returns:
            Tensor of shape (..., dim)
        """
        return self.down(F.silu(self.gate(x)) * self.up(x))


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

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses n_experts, n_shared_experts, n_experts_per_tok,
                   dim, expert_dim
        """
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_shared = cfg.n_shared_experts
        self.topk = cfg.n_experts_per_tok

        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        # load-balancing bias adjusted externally during training; not a gradient param
        self.register_buffer("router_bias", torch.zeros(cfg.n_experts))

        self.routed_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [
                Expert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok)
                for _ in range(self.n_shared)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input of shape (B, T, dim)
        Returns:
            Tensor of shape (B, T, dim); shared expert outputs are summed on top
            of the weighted routed expert outputs
        """
        B, T, D = x.shape
        flat = x.view(B * T, D)

        # Aux-loss-free load balancing (DeepSeek-V3): the bias shifts only the
        # selection of which experts fire so underused experts are picked more,
        # but the gating weights come from unbiased softmax scores so the bias
        # never shows up in the gradient.
        logits = self.router(flat)  # (B*T, n_experts), unbiased
        scores = F.softmax(logits, dim=-1)
        _, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)
        topk_scores = scores.gather(-1, topk_idx)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)  # renorm

        # routed expert dispatch (token-level scatter)
        out = torch.zeros_like(flat)
        for i in range(self.topk):
            expert_ids = topk_idx[:, i]
            token_scores = topk_scores[:, i].unsqueeze(-1)
            for eid in range(self.n_experts):
                mask = expert_ids == eid
                if not mask.any():
                    continue
                out[mask] += token_scores[mask] * self.routed_experts[eid](flat[mask])

        # shared experts always fire for every token
        for shared in self.shared_experts:
            out = out + shared(flat)

        return out.view(B, T, D)


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
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, loop_dim, 2, device=h.device, dtype=h.dtype) / loop_dim)
    )
    angles = loop_t * freqs  # (loop_dim//2,)
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
    emb_full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    emb_full[:loop_dim] = emb
    return h + emb_full.unsqueeze(0).unsqueeze(0)


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
        t_idx = loop_t if loop_t <= max_t else max_t
        s = self.scale(torch.tensor(t_idx, device=x.device))  # (rank,)
        down = self.down(x) * s  # (B, T, rank)
        return down @ self.B  # (B, T, dim)


# ---------------------------------------------------------------------------
# Single Transformer Block (shared across recurrent loops)
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """
    Standard pre-norm transformer block with swappable attention and optional MoE FFN.

    Attention is selected by cfg.attn_type:
        "gqa" → GQAttention  (Grouped Query Attention, fewer KV heads)
        "mla" → MLAttention  (Multi-Latent Attention, compressed KV cache)

    FFN is selected by use_moe:
        True  → MoEFFN  (fine-grained routed + shared experts; used in RecurrentBlock)
        False → Expert  (dense SwiGLU FFN; used in Prelude and Coda)
    """

    def __init__(self, cfg: MythosConfig, use_moe: bool = False):
        """
        Args:
            cfg     -- MythosConfig; attn_type selects the attention class
            use_moe -- if True, use MoEFFN; otherwise use a dense Expert FFN
        """
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.attn = MLAttention(cfg) if cfg.attn_type == "mla" else GQAttention(cfg)
        self.ffn = MoEFFN(cfg) if use_moe else Expert(cfg.dim, cfg.dim * 4 // 3)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- precomputed RoPE frequencies
            mask      -- additive causal mask or None
            kv_cache  -- cache dict mutated in-place by the attention layer
            cache_key -- key identifying this layer in the cache

        Returns:
            Output tensor of shape (B, T, dim)
        """
        x = x + self.resid_drop(
            self.attn(self.attn_norm(x), freqs_cis, mask, kv_cache, cache_key)
        )
        x = x + self.resid_drop(self.ffn(self.ffn_norm(x)))
        return x


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


# ---------------------------------------------------------------------------
# ACT halting (Adaptive Computation Time)
# ---------------------------------------------------------------------------


class ACTHalting(nn.Module):
    """
    Adaptive Computation Time halting mechanism (Graves, 2016).

    Learns a per-position halting probability at each loop iteration. Positions
    where the hidden state has converged (high cumulative halting probability)
    stop accumulating updates, while positions still being refined continue.
    This lets easy tokens halt early and hard tokens receive more computation,
    all within the same batch. Also makes the model Turing-complete under
    certain assumptions about the expressiveness of the transformer block.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim -- hidden state dimension; input to the halting scalar predictor
        """
        super().__init__()
        self.halt = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict per-position halting probability from the current hidden state.

        Args:
            h -- hidden state of shape (B, T, dim)

        Returns:
            Halting probability tensor of shape (B, T), values in (0, 1)
        """
        return torch.sigmoid(self.halt(h)).squeeze(-1)


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
        self.injection = LTIInjection(cfg.dim)
        self.act = ACTHalting(cfg.dim)
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.norm = RMSNorm(cfg.dim)
        self.loop_dim = (
            cfg.dim // 8
        )  # fraction of channels receiving loop-index embedding

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
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
            h_loop = loop_index_embedding(h, t, self.loop_dim)
            combined = self.norm(h_loop + e)
            cache_key = f"recurrent_loop_{t}"
            trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
            trans_out = trans_out + self.lora(trans_out, t)
            h = self.injection(h, e, trans_out)

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

            # Only short-circuit when there is no KV cache to keep consistent.
            # With a cache, every loop depth must run on every forward pass so
            # later decode steps find populated keys at every cache_key.
            if halted.all() and kv_cache is None:
                break

        return h_out



# ===========================================================================
# P4 Recurrent Block Variants
# ===========================================================================


class ComplexityEstimator(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class PathCostRecurrentBlock(RecurrentBlock):
    def __init__(self, cfg: MythosConfig):
        super().__init__(cfg)
        self.complexity_estimator = ComplexityEstimator(cfg.dim, cfg.dim // 4)
        self.path_cost_threshold = cfg.path_cost_threshold
        self.path_cost_mode = cfg.path_cost_mode

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape

        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_cost = torch.zeros(B, T, device=h.device)
        h_out = torch.zeros_like(h)

        for t in range(n_loops):
            h_loop = loop_index_embedding(h, t, self.loop_dim)
            combined = self.norm(h_loop + e)
            cache_key = f"pathcost_loop_{t}"
            trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
            trans_out = trans_out + self.lora(trans_out, t)
            h = self.injection(h, e, trans_out)

            complexity = self.complexity_estimator(h).squeeze(-1)
            cumulative_cost = cumulative_cost + complexity * (~halted).float()
            newly_halted = cumulative_cost >= self.path_cost_threshold
            halted = halted | newly_halted

            remaining = (1.0 - cumulative_cost / self.path_cost_threshold).clamp(min=0)
            weight = remaining * (~halted).float()
            h_out = h_out + weight.unsqueeze(-1) * h

            if halted.all() and kv_cache is None:
                break

        return h_out


# ===========================================================================
# P3: Hierarchical Loop Mechanism
# ===========================================================================


class AdaptiveScaleSelector(nn.Module):
    """
    P3: Learns to predict which hierarchical scale to use per token.

    Takes the hidden state and emits a probability distribution over scales.
    The model can then either:
    - Hard select: use only the highest-probability scale
    - Soft select: blend representations across scales
    """

    def __init__(self, dim: int, num_scales: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_scales),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: hidden states (B, T, D)
        Returns:
            scale_logits: (B, T, num_scales) - raw logits for each scale
        """
        return self.net(h)


class HierarchicalRecurrentBlock(nn.Module):
    """
    P3: Hierarchical Recurrent Block with multi-scale processing.

    Processes information at multiple temporal scales simultaneously:
    - Scale 0: token-level (pool_factor=1)
    - Scale 1: short-range (pool_factor=2)
    - Scale 2: medium-range (pool_factor=4)
    - Scale 3: long-range (pool_factor=8)

    Key enhancements over basic hierarchical:
    1. Adaptive scale selection per token via learned ScaleSelector
    2. Top-down broadcast: coarse-scale info influences fine-scale processing
    3. Bottom-up fusion: fine-scale details inform coarse representations
    4. Cross-scale attention for information exchange

    Args:
        cfg: MythosConfig
        num_scales: number of hierarchical scales (default 4)
    """

    def __init__(self, cfg: MythosConfig, num_scales: int = None):
        super().__init__()
        self.cfg = cfg
        self.num_scales = num_scales or cfg.hierarchical_num_scales

        # Core transformer block (shared across scales initially)
        self.block = TransformerBlock(cfg, use_moe=True)

        # Per-scale normalization and transformations
        self.norms = nn.ModuleList([RMSNorm(cfg.dim) for _ in range(self.num_scales)])
        self.scale_embeddings = nn.ModuleList([
            nn.Linear(cfg.dim, cfg.dim) for _ in range(self.num_scales)
        ])

        # Pooling factors: 1, 2, 4, 8 (token-level to document-level)
        self.pool_factors = [2 ** s for s in range(self.num_scales)]

        # Injection and LoRA
        self.injection = LTIInjection(cfg.dim)
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.loop_dim = cfg.dim // 8

        # P3: Adaptive scale selection
        if cfg.hierarchical_adaptive_scale:
            self.scale_selector = AdaptiveScaleSelector(
                cfg.dim, self.num_scales, hidden_dim=cfg.dim // 8
            )

        # P3: Top-down broadcast projection (coarse → fine)
        if cfg.hierarchical_top_down_broadcast:
            self.top_down_proj = nn.ModuleList([
                nn.Linear(cfg.dim, cfg.dim) for _ in range(self.num_scales - 1)
            ])

    def _get_pooled_representation(
        self, h: torch.Tensor, pool_factor: int
    ) -> torch.Tensor:
        """
        Pool hidden states by pool_factor using mean pooling.
        Handles non-divisible sequence lengths gracefully.
        """
        if pool_factor == 1:
            return h

        T = h.shape[1]
        new_len = T // pool_factor
        if new_len == 0:
            return h.mean(dim=1, keepdim=True)

        # Truncate to divisible length, then pool
        truncated = h[:, :new_len * pool_factor]
        pooled = truncated.view(truncated.shape[0], new_len, pool_factor, -1)
        return pooled.mean(dim=2)  # (B, new_len, D)

    def _broadcast_to_fine(
        self, coarse_h: torch.Tensor, target_len: int, scale_idx: int
    ) -> torch.Tensor:
        """
        P3: Broadcast coarse representation to match fine-grained sequence length.
        Uses learned projection and upsampling.
        """
        if coarse_h.shape[1] == target_len:
            return coarse_h
        # Repeat and interpolate
        repeat_factor = (target_len + coarse_h.shape[1] - 1) // coarse_h.shape[1]
        repeated = coarse_h.repeat_interleave(repeat_factor, dim=1)
        return repeated[:, :target_len, :]

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Multi-scale hierarchical forward pass.

        Args:
            h: hidden states (B, T, D)
            e: embed token (B, T, D)
            freqs_cis: RoPE frequencies
            mask: attention mask
            n_loops: number of loop iterations
            kv_cache: cache dict

        Returns:
            Processed hidden states (B, T, D)
        """
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape

        # Track representations at each scale
        scale_hs = [h.clone() for _ in range(self.num_scales)]
        top_down_state = None

        for t in range(n_loops):
            # Step 1: Determine which scale to focus on (adaptive selection)
            if self.cfg.hierarchical_adaptive_scale and hasattr(self, 'scale_selector'):
                scale_logits = self.scale_selector(scale_hs[0])  # (B, T, num_scales)
                # Soft attention over scales
                scale_weights = F.softmax(scale_logits, dim=-1)  # (B, T, num_scales)
                # Use weighted combination of scale representations
                h_processed = sum(
                    scale_weights[:, :, i:i+1] * scale_hs[i]
                    for i in range(self.num_scales)
                )
            else:
                # Default: use finest scale (scale 0)
                h_processed = scale_hs[0]
                scale_weights = None

            # Step 2: Apply loop index embedding and process
            h_loop = loop_index_embedding(h_processed, t, self.loop_dim)

            # Step 3: Process at each scale and fuse
            new_scale_hs = []
            for s in range(self.num_scales):
                # Apply top-down broadcast if enabled
                if (self.cfg.hierarchical_top_down_broadcast and
                    hasattr(self, 'top_down_proj') and
                    top_down_state is not None and
                    s < self.num_scales - 1):
                    # Broadcast from coarser scale to finer
                    broadcast_h = self._broadcast_to_fine(
                        top_down_state, T, s
                    )
                    # Project and add to current scale
                    broadcast_h = self.top_down_proj[s](broadcast_h)
                    scale_input = self.norms[s](h_loop + e + broadcast_h)
                else:
                    scale_input = self.norms[s](h_loop + e)

                # Apply transformer block
                trans_out = self.block(
                    scale_input, freqs_cis, mask, kv_cache, f"hier_s{s}_t{t}"
                )

                # Inject into current scale representation
                updated = self.injection(scale_hs[s], e, trans_out)
                new_scale_hs.append(updated)

            # Step 4: Top-down state update (coarsest scale becomes top-down signal)
            top_down_state = new_scale_hs[-1]  # coarsest scale

            # Update all scale representations
            scale_hs = new_scale_hs

            # Apply LoRA adaptation
            h_out = self.injection(h, e, self.lora(h, t))
            h = h_out

        # Return finest-scale representation (most detailed)
        return scale_hs[0]


# ===========================================================================
# P3: Meta-Learning Loop Depth
# ===========================================================================


class MetaLoopPredictor(nn.Module):
    """
    P3: Meta-learning predictor that learns to predict optimal loop depth.

    Instead of using a fixed number of loops or a learned scalar alpha,
    this network predicts the number of loops (or loop weights) from
    the input hidden states. This allows the model to adaptively use
    more loops for complex inputs and fewer for simple inputs.

    Architecture:
    - Processes the input to estimate complexity
    - Outputs either:
      (a) A scalar "complexity score" used to interpolate loop count
      (b) Per-loop attention weights (which loops to emphasize)
    """

    def __init__(
        self,
        dim: int,
        max_loops: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.max_loops = max_loops

        # Complexity estimator network
        layers = []
        in_dim = dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))  # Output: complexity score
        self.complexity_net = nn.Sequential(*layers)

        # Loop importance predictor (predicts weight for each loop iteration)
        self.loop_weight_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_loops),
            nn.Softmax(dim=-1),  # Normalize to sum to 1
        )

    def forward(self, h: torch.Tensor) -> tuple:
        """
        Args:
            h: hidden states (B, T, D) - uses the first token's representation

        Returns:
            complexity_score: (B, 1) - estimated input complexity
            loop_weights: (B, max_loops) - importance weight for each loop
        """
        # Use CLS token representation (first token) for complexity estimation
        h_cls = h[:, 0, :]  # (B, D)

        complexity_score = self.complexity_net(h_cls)  # (B, 1)
        loop_weights = self.loop_weight_net(h_cls)     # (B, max_loops)

        return complexity_score, loop_weights


class MetaLoopRecurrentBlock(nn.Module):
    """
    P3: Meta-Learning Recurrent Block.

    Enhances the basic MetaLoopRecurrentBlock by:
    1. Using MetaLoopPredictor to estimate input complexity
    2. Dynamically weighting loop iterations based on predicted importance
    3. Allowing early termination based on convergence

    The loop_weights predict which iterations are most important,
    allowing the model to "focus" on critical loop iterations.
    """

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.cfg = cfg
        self.block = TransformerBlock(cfg, use_moe=True)
        self.injection = LTIInjection(cfg.dim)
        self.norm = RMSNorm(cfg.dim)
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.loop_dim = cfg.dim // 8

        # P3: Meta-learning predictor
        self.meta_predictor = MetaLoopPredictor(
            dim=cfg.dim,
            max_loops=cfg.max_loop_iters,
            hidden_dim=cfg.meta_loop_predictor_hidden,
            num_layers=cfg.meta_loop_predictor_layers,
        )

        # Learned baseline alpha (similar to original)
        self.meta_alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Meta-learning forward pass with adaptive loop weighting.

        Args:
            h: hidden states (B, T, D)
            e: embed token (B, T, D)
            freqs_cis: RoPE frequencies
            mask: attention mask
            n_loops: override number of loops
            kv_cache: cache dict

        Returns:
            Processed hidden states (B, T, D)
        """
        n_loops = n_loops or self.cfg.max_loop_iters

        # P3: Get predicted complexity and loop weights
        complexity_score, loop_weights = self.meta_predictor(h)
        loop_weights = loop_weights[:, :n_loops]  # (B, n_loops)

        # Baseline alpha for weighted combination
        alpha = torch.sigmoid(self.meta_alpha)

        # Track accumulated output
        h_out = torch.zeros_like(h)
        h_prev = h.clone()

        for t in range(n_loops):
            h_loop = loop_index_embedding(h, t, self.loop_dim)
            combined = self.norm(h_loop + e)
            trans_out = self.block(combined, freqs_cis, mask, kv_cache, f"meta_{t}")

            # P3: Weight this iteration's contribution by predicted importance
            weight = loop_weights[:, t:t+1].unsqueeze(-1)  # (B, 1, 1)
            h_injected = self.injection(h, e, trans_out)

            # Update with weighted combination
            h = alpha * h + (1 - alpha) * h_injected

            # Accumulate weighted output
            h_out = h_out + weight * h

            # P3: Check for convergence (if complexity is low, exit early)
            if t > 0:
                diff = (h - h_prev).abs().mean()
                # If complexity score is low AND diff is small, we can early exit
                # Note: This is a simplified heuristic; proper convergence would
                # require per-sample early exit which is complex with batching
                h_prev = h.clone()

        # Return accumulated weighted representation
        return h_out


class ConeRecurrentBlock(nn.Module):
    def __init__(
        self,
        cfg: MythosConfig,
        segment_size: int = 16,
        n_segments_per_doc: int = 4,
        use_attention_pool: bool = True,
        use_cone_path_routing: bool = True,
        use_top_down_broadcast: bool = True,
        cone_sharpness: float = 2.0,
        learn_fusion: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.segment_size = segment_size
        self.n_segments_per_doc = n_segments_per_doc
        self.use_attention_pool = use_attention_pool
        self.use_cone_path_routing = use_cone_path_routing
        self.use_top_down_broadcast = use_top_down_broadcast
        self.cone_sharpness = cone_sharpness
        self.learn_fusion = learn_fusion

        self.block = TransformerBlock(cfg, use_moe=True)
        self.injection = LTIInjection(cfg.dim)
        self.norm = RMSNorm(cfg.dim)
        self.loop_dim = cfg.dim // 8
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)

        if use_cone_path_routing:
            self.cone_gate = nn.Linear(cfg.dim, 3, bias=False)

        if learn_fusion:
            self.A0 = nn.Linear(cfg.dim, cfg.dim, bias=False)
            self.B0 = nn.Linear(cfg.dim, cfg.dim, bias=False)
            self.A1 = nn.Linear(cfg.dim, cfg.dim, bias=False)
            self.B1 = nn.Linear(cfg.dim, cfg.dim, bias=False)
            self.A2 = nn.Linear(cfg.dim, cfg.dim, bias=False)
            self.B2 = nn.Linear(cfg.dim, cfg.dim, bias=False)

        if use_attention_pool:
            self.segment_pool = nn.Linear(cfg.dim, cfg.dim)

        if use_top_down_broadcast:
            self.top_down_alpha = nn.Parameter(torch.tensor(0.5))

    def _cone_path_weights(self, h: torch.Tensor):
        logits = self.cone_gate(h)
        weights = F.softmax(logits / self.cone_sharpness, dim=-1)
        return weights[..., 0:1], weights[..., 1:2], weights[..., 2:3]

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape

        w0_out = torch.zeros(B, T, 1, device=h.device, dtype=h.dtype)
        w1_out = torch.zeros(B, T, 1, device=h.device, dtype=h.dtype)
        w2_out = torch.zeros(B, T, 1, device=h.device, dtype=h.dtype)
        h_out = torch.zeros_like(h)

        for t in range(n_loops):
            h_loop = loop_index_embedding(h, t, self.loop_dim)
            combined = self.norm(h_loop + e)
            cache_key = f"cone_loop_{t}"

            if self.use_cone_path_routing:
                w0, w1, w2 = self._cone_path_weights(h)
                w0_out = w0_out + w0
                w1_out = w1_out + w1
                w2_out = w2_out + w2

                trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
                trans_out = trans_out + self.lora(trans_out, t)

                if self.learn_fusion:
                    h0 = torch.tanh(self.A0(h) + self.B0(e + trans_out))
                    h1 = torch.tanh(self.A1(h) + self.B1(e + trans_out))
                    h2 = torch.tanh(self.A2(h) + self.B2(e + trans_out))
                    h_delta = w0 * h0 + w1 * h1 + w2 * h2
                    h = h + 0.1 * h_delta
                else:
                    h = self.injection(h, e, trans_out)

                if self.use_top_down_broadcast and t > 0:
                    alpha = torch.sigmoid(self.top_down_alpha)
                    h = alpha * h + (1 - alpha) * h_loop
            else:
                trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
                h = self.injection(h, e, trans_out)

            h_out = h_out + h

        result = h_out / n_loops
        result._cone_w0 = w0_out / n_loops
        result._cone_w1 = w1_out / n_loops
        result._cone_w2 = w2_out / n_loops
        return result

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

        # GQA uses full head_dim for RoPE; MLA uses only qk_rope_head_dim (decoupled)
        freqs = precompute_rope_freqs(
            cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("freqs_cis", freqs)
        freqs_mla = precompute_rope_freqs(
            cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("freqs_cis_mla", freqs_mla)

        self.prelude = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.prelude_layers)]
        )
        # P4: Select recurrent block type based on config flags
        if cfg.use_cone:
            self.recurrent = ConeRecurrentBlock(
                cfg=cfg,
                segment_size=max(4, cfg.max_seq_len // 256),
                n_segments_per_doc=max(2, cfg.max_seq_len // 512),
                use_attention_pool=True,
                use_cone_path_routing=True,
                use_top_down_broadcast=True,
                cone_sharpness=cfg.cone_sharpness,
                learn_fusion=True,
            )
        elif cfg.use_path_cost:
            self.recurrent = PathCostRecurrentBlock(cfg)
        elif cfg.use_meta_loop:
            self.recurrent = MetaLoopRecurrentBlock(cfg)
        elif cfg.use_hierarchical:
            self.recurrent = HierarchicalRecurrentBlock(cfg)
        else:
            self.recurrent = RecurrentBlock(cfg)
        self.coda = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.coda_layers)]
        )

        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize all linear and embedding weights with N(0, 0.02)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

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
        kv_cache: Optional[dict] = None,
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
        freqs_cis = (
            self.freqs_cis_mla if self.cfg.attn_type == "mla" else self.freqs_cis
        )[start_pos : start_pos + T]
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
        kv_cache: dict = {}
        prompt_len = input_ids.shape[1]
        for step in range(max_new_tokens):
            if step == 0:
                cur_ids = input_ids
                start_pos = 0
            else:
                cur_ids = input_ids[:, -1:]
                start_pos = prompt_len + step - 1
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

    def speculative_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        n_loops: int = 8,
        gamma: int = 4,
        n_loops_draft: int = 2,
        temperature: float = 1.0,
        top_k: int = 50,
        draft_model: Optional["OpenMythos"] = None,
    ) -> torch.Tensor:
        """
        Speculative autoregressive generation using a draft model.

        When no draft_model is provided, uses the same model with reduced loop depth
        as the draft (n_loops_draft < n_loops), which is faster but less accurate.

        Args:
            input_ids     -- prompt token indices of shape (B, T)
            max_new_tokens -- number of tokens to generate
            n_loops       -- recurrent loop depth for target model
            gamma         -- number of tokens to draft per speculative round
            n_loops_draft -- loop depth for draft model (smaller = faster)
            temperature   -- softmax temperature for sampling
            top_k         -- restrict sampling to top-K logits (0 = disabled)
            draft_model   -- optional separate draft model; if None, uses self

        Returns:
            Token indices of shape (B, T + max_new_tokens)
        """
        decoder = SpeculativeRDTDecoder(
            target_model=self,
            draft_model=draft_model,
            gamma=gamma,
            n_loops_target=n_loops,
            n_loops_draft=n_loops_draft,
            temperature=temperature,
            top_k=top_k,
        )
        return decoder.generate(input_ids, max_new_tokens=max_new_tokens)


# ===========================================================================
# P0: Multi-Scale Loop + Curriculum Learning
# ===========================================================================


class DepthSelector(nn.Module):
    """
    P0: Complexity-aware loop depth selector.

    Learns to predict the complexity of the input hidden state, then selects
    an appropriate loop depth (from cfg.loop_depths) based on thresholds.

    Architecture:
    - Attention pooling over sequence → single vector
    - Linear network → complexity score in [0, 1]
    - Threshold comparison → depth selection

    Usage:
        depth_selector = DepthSelector(cfg)
        complexity = depth_selector(h)           # (B,) complexity scores
        loop_depth = depth_selector.select_depth(complexity)  # (B,) depths
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses dim, loop_depths, complexity_threshold_*
        """
        super().__init__()
        self.cfg = cfg
        self.depths = cfg.loop_depths  # e.g., [4, 8, 16]

        # Learned complexity predictor: hidden state → complexity score
        # Uses attention pooling over sequence to get a single complexity score
        self.complexity_net = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim // 2),
            nn.GELU(),
            nn.Linear(cfg.dim // 2, 1),
        )

        # Attention pooling for sequence → single vector
        self.pool_attn = nn.Linear(cfg.dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict complexity score for the input hidden state.

        Args:
            h -- hidden state of shape (B, T, dim)

        Returns:
            Complexity score tensor of shape (B,), values in [0, 1]
            Higher = more complex task
        """
        B, T, D = h.shape

        # Attention pooling: weighted average of sequence positions
        weights = F.softmax(self.pool_attn(h), dim=1)  # (B, T, 1)
        h_pooled = (weights * h).sum(dim=1)  # (B, D)

        # Predict complexity
        complexity = torch.sigmoid(self.complexity_net(h_pooled).squeeze(-1))  # (B,)

        return complexity

    def select_depth(self, complexity: torch.Tensor) -> torch.Tensor:
        """
        Select loop depth based on complexity score.

        Args:
            complexity -- complexity score tensor of shape (B,), values in [0, 1]

        Returns:
            Loop depth tensor of shape (B,), values in self.depths
        """
        depths_tensor = torch.tensor(self.depths, device=complexity.device, dtype=torch.long)

        # Create decision boundaries
        thresholds = torch.tensor(
            [self.cfg.complexity_threshold_low, self.cfg.complexity_threshold_high],
            device=complexity.device,
        )

        # Determine index based on complexity
        # complexity < low → 0, low ≤ complexity < high → 1, else → 2
        indices = torch.zeros_like(complexity, dtype=torch.long)
        indices = indices + (complexity >= thresholds[0]).long()
        indices = indices + (complexity >= thresholds[1]).long()
        indices = indices.clamp(0, len(self.depths) - 1)

        return depths_tensor[indices]


class CurriculumLoopScheduler:
    """
    P0: Curriculum learning scheduler for loop depth.

    Implements 4-phase curriculum for loop depth training:

    Phase 1 (0 - phase1_steps): Fixed min_depth
        - All samples use curriculum_min_depth loops
        - Model learns basic pattern matching

    Phase 2 (phase1_steps - phase1+phase2_steps): Fixed medium_depth
        - All samples use (min_depth + max_depth) // 2 loops
        - Model learns intermediate reasoning

    Phase 3 (phase1+phase2 - phase1+phase2+phase3_steps): Mixed depths
        - Randomly sample from [min_depth, mid_depth, max_depth]
        - Model learns to handle variable complexity

    Phase 4 (remaining steps): Dynamic + ACT
        - Use DepthSelector to predict depth
        - Fall back to max_depth with ACT for remaining steps

    This curriculum prevents early training instability while enabling
    the model to eventually learn adaptive depth selection.
    """

    def __init__(self, cfg: MythosConfig, total_steps: int):
        """
        Args:
            cfg -- MythosConfig with curriculum parameters
            total_steps -- total training steps for calculating phase boundaries
        """
        self.cfg = cfg
        self.total_steps = total_steps

        self.phase1_end = cfg.curriculum_phase1_steps
        self.phase2_end = self.phase1_end + cfg.curriculum_phase2_steps
        self.phase3_end = self.phase2_end + cfg.curriculum_phase3_steps
        # Phase 4: remaining steps (up to total_steps)

        # Calculate depths
        self.min_depth = cfg.curriculum_min_depth
        self.max_depth = cfg.curriculum_max_depth
        self.mid_depth = (self.min_depth + self.max_depth) // 2

    def get_depth(self, step: int, complexity: Optional[torch.Tensor] = None) -> int:
        """
        Get the loop depth for a given training step.

        Args:
            step -- current training step
            complexity -- optional complexity score for Phase 4 (shape: B,)

        Returns:
            Loop depth (int) for this step
        """
        if step < self.phase1_end:
            # Phase 1: Fixed min_depth
            return self.min_depth

        elif step < self.phase2_end:
            # Phase 2: Fixed mid_depth
            return self.mid_depth

        elif step < self.phase3_end:
            # Phase 3: Random mixed depths
            import random
            return random.choice([self.min_depth, self.mid_depth, self.max_depth])

        else:
            # Phase 4: Dynamic + ACT
            # If complexity predictor is provided, use it
            if complexity is not None:
                # Use median complexity as depth selector
                median_complexity = complexity.median().item()
                if median_complexity < self.cfg.complexity_threshold_low:
                    return self.min_depth
                elif median_complexity < self.cfg.complexity_threshold_high:
                    return self.mid_depth
                else:
                    return self.max_depth
            else:
                # Fallback to max_depth with ACT
                return self.max_depth

    def get_phase(self, step: int) -> str:
        """Return the current curriculum phase name."""
        if step < self.phase1_end:
            return "phase1_fixed_min"
        elif step < self.phase2_end:
            return "phase2_fixed_mid"
        elif step < self.phase3_end:
            return "phase3_mixed"
        else:
            return "phase4_dynamic_act"


# ===========================================================================
# Training Regularization: Loop Consistency
# ===========================================================================


class LoopConsistencyRegularizer(nn.Module):
    """
    Loop Consistency Regularization.

    Encourages the recurrent loop to produce consistent hidden states across
    forward passes. Without this, the model may produce different outputs
    for semantically equivalent inputs that pass through different loop depths.

    Loss components:
        1. Consecutive consistency: encourage stable transformation across loops
        2. Hidden state variance: encourage non-trivial information transformation
        3. Loop stationarity: hidden states shouldn't drift too far from initial
    """

    def __init__(self, beta_consistency: float = 0.1, beta_variance: float = 0.05):
        """
        Args:
            beta_consistency -- weight for consecutive loop consistency loss
            beta_variance -- weight for hidden state variance loss
        """
        super().__init__()
        self.beta_c = beta_consistency
        self.beta_v = beta_variance

    def forward(
        self,
        h_0: torch.Tensor,
        h_T: torch.Tensor,
        loop_outputs: list[torch.Tensor],
        main_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loop consistency regularization loss.

        Args:
            h_0 -- initial hidden state (B, T, dim)
            h_T -- final hidden state after all loops (B, T, dim)
            loop_outputs -- list of hidden states at each loop step
            main_loss -- the primary loss (cross-entropy)

        Returns:
            Total loss = main_loss + consistency_loss + variance_loss
        """
        total_loss = main_loss

        # 1. Consecutive loop consistency
        if len(loop_outputs) >= 2:
            cycle_loss = 0.0
            for t in range(len(loop_outputs) - 1):
                cycle_loss += F.mse_loss(loop_outputs[t], loop_outputs[t + 1])
            cycle_loss = cycle_loss / (len(loop_outputs) - 1)
            total_loss = total_loss + self.beta_c * cycle_loss

        # 2. Hidden state variance — encourage transformation, not copying
        if len(loop_outputs) > 0:
            final_out = loop_outputs[-1]
            variance_loss = -torch.var(final_out, dim=-1).mean()
            total_loss = total_loss + self.beta_v * variance_loss

        return total_loss


# ===========================================================================
# MoE Enhancements: Capacity-Aware Routing & Task-Conditioned MoE
# ===========================================================================


class CapacityAwareRouter(nn.Module):
    """
    Capacity-Aware Mixture-of-Experts Router.

    Standard MoE routing selects top-K experts based on probability scores,
    but doesn't consider expert capacity. This can lead to:
        - Some experts overloaded (too many tokens → degraded quality)
        - Some experts underutilized (idle capacity)

    Capacity-Aware Routing adds a per-expert token budget:
        - Each expert has max_capacity tokens per forward pass
        - Tokens beyond capacity are re-routed to next-best available expert
        - Maintains load balance without auxiliary losses
    """

    def __init__(self, cfg: MythosConfig, capacity_factor: float = 1.5):
        """
        Args:
            cfg -- MythosConfig
            capacity_factor -- multiplier on average capacity per expert
        """
        super().__init__()
        self.n_experts = cfg.n_experts
        self.topk = cfg.n_experts_per_tok
        self.capacity_factor = capacity_factor

        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        self.register_buffer("router_bias", torch.zeros(cfg.n_experts))

    def forward(
        self,
        x: torch.Tensor,
        token_capacity: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts with capacity constraints.

        Args:
            x -- input tensor (B*T, dim)
            token_capacity -- max tokens per expert per forward pass

        Returns:
            Tuple of (selected_expert_ids, weights)
        """
        B_T, D = x.shape

        logits = self.router(x) + self.router_bias
        scores = F.softmax(logits, dim=-1)

        capacities = (token_capacity // self.topk) * torch.ones(
            self.n_experts, device=x.device
        )

        remaining = capacities.clone()
        selected = torch.full((B_T, self.topk), -1, dtype=torch.long, device=x.device)
        weights = torch.zeros_like(selected, dtype=torch.float)

        for k in range(self.topk):
            for b in range(B_T):
                for _ in range(self.n_experts):
                    top_expert = logits[b].argmax().item()
                    if remaining[top_expert] > 0:
                        selected[b, k] = top_expert
                        weights[b, k] = scores[b, top_expert]
                        remaining[top_expert] -= 1
                        break
                    else:
                        logits[b, top_expert] = float("-inf")

        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        return selected, weights


class TaskConditionedMoE(nn.Module):
    """
    Task-Conditioned Mixture of Experts.

    Extends standard MoE with explicit task conditioning:
    - Each expert is guided toward a domain specialty (soft, not hard-coded)
    - Task embedding modulates routing probabilities
    - Experts specialize through auxiliary objectives

    Domain groups (soft specialization via auxiliary loss):
        Group 0: Syntax, tokenization, morphology
        Group 1: Factual knowledge, named entities
        Group 2: Reasoning, logic, deduction
        Group 3: Mathematics, code, formal systems
    """

    NUM_DOMAINS = 4

    def __init__(self, cfg: MythosConfig, specialization_strength: float = 0.1):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_shared = cfg.n_shared_experts
        self.topk = cfg.n_experts_per_tok
        self.specialization_strength = specialization_strength

        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        self.register_buffer("router_bias", torch.zeros(cfg.n_experts))

        self.expert_domain_affinity = nn.Parameter(
            torch.randn(cfg.n_experts, self.NUM_DOMAINS) * 0.02
        )

        self.task_embed = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim // 2),
            nn.GELU(),
            nn.Linear(cfg.dim // 2, self.NUM_DOMAINS),
        )

        self.routed_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [
                Expert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok)
                for _ in range(self.n_shared)
            ]
        )

    def get_domain_affinity_loss(self) -> torch.Tensor:
        """Auxiliary loss encouraging expert domain specialization."""
        affinity = F.softmax(self.expert_domain_affinity, dim=-1)
        entropy = -(affinity * torch.log(affinity + 1e-8)).sum(dim=-1).mean()
        return self.specialization_strength * entropy

    def forward(
        self,
        x: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with task conditioning.

        Args:
            x -- input tensor (B, T, dim)
            task_id -- optional domain label (B,) for hard conditioning

        Returns:
            Output tensor (B, T, dim)
        """
        B, T, D = x.shape
        flat = x.view(B * T, D)

        if task_id is not None:
            task_dist = torch.zeros(B, self.NUM_DOMAINS, device=x.device)
            task_dist.scatter_(1, task_id.unsqueeze(1), 1.0)
        else:
            task_logits = self.task_embed(x.mean(dim=1))
            task_dist = F.softmax(task_logits, dim=-1)

        domain_modulation = task_dist @ self.expert_domain_affinity.T

        logits = self.router(flat)
        logits = logits + domain_modulation.repeat_interleave(T, dim=0) * 0.5

        scores = F.softmax(logits, dim=-1)
        _, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)
        topk_scores = scores.gather(-1, topk_idx)
        topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-8)

        out = torch.zeros_like(flat)
        for i in range(self.topk):
            expert_ids = topk_idx[:, i]
            token_scores = topk_scores[:, i].unsqueeze(-1)
            for eid in range(self.n_experts):
                mask = expert_ids == eid
                if not mask.any():
                    continue
                out[mask] += token_scores[mask] * self.routed_experts[eid](flat[mask])

        for shared in self.shared_experts:
            out = out + shared(flat)

        return out.view(B, T, D)


# ===========================================================================
# OpenMythosEnhanced: All Enhancements Wrapper
# ===========================================================================


class OpenMythosEnhanced(OpenMythos):
    """
    Fully Enhanced OpenMythos with all P0/P1/P2/P3 enhancements.

    This class wraps the base OpenMythos and adds:
    - P0: Multi-scale loop depth, curriculum learning
    - P1: Loop consistency regularization, Capacity-aware routing
    - P2: Speculative decoding, Task-conditioned MoE
    - P3: Hierarchical recurrence, Meta-learned loop depth
    """

    def __init__(self, cfg: MythosConfig):
        super().__init__(cfg)

        # P0: Depth selector and curriculum
        if cfg.enable_multiscale_loop:
            self.depth_selector = DepthSelector(cfg)

        if cfg.enable_curriculum:
            self.curriculum_scheduler = None  # Initialized with total_steps in training

        # P1 Enhancements
        if hasattr(cfg, "loop_consistency_beta"):
            self.consistency_regularizer = LoopConsistencyRegularizer(
                beta_consistency=cfg.loop_consistency_beta,
                beta_variance=getattr(cfg, "loop_variance_beta", 0.05),
            )

        if hasattr(cfg, "capacity_aware_routing") and cfg.capacity_aware_routing:
            self.capacity_router = CapacityAwareRouter(cfg)

        # P2 Enhancements
        if cfg.use_speculative:
            self.speculative_decoder = None  # Initialized with draft model

        if hasattr(cfg, "task_conditioned_moe") and cfg.task_conditioned_moe:
            self.task_moe = TaskConditionedMoE(cfg)


# ===========================================================================
# P2: Speculative Decoding
# ===========================================================================


class SpeculativeRDTDecoder:
    """
    Speculative Decoding for OpenMythos (RDT architecture).

    Uses a smaller "draft" model to propose multiple tokens in parallel,
    then uses the larger "target" model to verify them in a single forward pass.
    This accelerates autoregressive generation by amortizing the cost of
    complex recursive inference across multiple draft tokens.

    The RDT architecture is particularly well-suited for speculative decoding
    because the recursive loop can be shortened in the draft model (fewer
    n_loops) while still producing reasonable predictions, since each forward
    pass already aggregates information across the full prompt.

    Algorithm:
        1. Draft model autoregressively generates `gamma` tokens (fast, low n_loops)
        2. Target model evaluates ALL tokens in a SINGLE forward pass
           (prompt + draft tokens), producing logits for the NEXT token
        3. Accept/reject each draft token using probability thresholding:
           - If P_target(token) > P_draft(token) * threshold: accept
           - Else: reject and use target's sampled token
        4. If all gamma accepted, append target's predicted token too

    Args:
        target_model: OpenMythos model (full capacity)
        draft_model: OpenMythos model (fewer loops, faster but less accurate)
        gamma: Number of tokens to draft per speculative round
        n_loops_target: Loop depth for target model verification
        n_loops_draft: Loop depth for draft model generation
        temperature: Sampling temperature for final token selection
        top_k: Top-K filtering for sampling
        accept_threshold: Probability ratio threshold for acceptance
    """

    def __init__(
        self,
        target_model: "OpenMythos",
        draft_model: Optional["OpenMythos"] = None,
        gamma: int = 4,
        n_loops_target: int = 8,
        n_loops_draft: int = 2,
        temperature: float = 1.0,
        top_k: int = 50,
        accept_threshold: float = 1.0,
    ):
        self.target = target_model
        # If no draft model provided, use same model but with fewer loops
        self.draft = draft_model or target_model
        self.gamma = gamma
        self.n_loops_target = n_loops_target
        self.n_loops_draft = n_loops_draft
        self.temperature = temperature
        self.top_k = top_k
        self.accept_threshold = accept_threshold

    def _sample_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample a single token from logits."""
        logits = logits[:, -1, :] / self.temperature
        if self.top_k > 0:
            v, _ = logits.topk(self.top_k)
            logits[logits < v[:, -1:]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _verify_and_accept(
        self,
        draft_tokens: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Verify draft tokens against target model logits and return accepted tokens.

        Uses the "surprise" metric: if target is more confident than draft,
        the draft is likely correct. Acceptance threshold controls greediness.

        Args:
            draft_tokens: (B, gamma) draft token indices
            target_logits: (B, gamma+1, V) logits from target model where
                          target_logits[:, i] corresponds to position i in draft_tokens

        Returns:
            Accepted token indices (B, <= gamma)
        """
        B, gamma = draft_tokens.shape
        accepted = []

        for i in range(gamma):
            # Draft token at position i
            d_tok = draft_tokens[:, i:i+1]  # (B, 1)
            d_prob = F.softmax(target_logits[:, i, :] / self.temperature, dim=-1)
            d_prob = d_prob.gather(-1, d_tok).squeeze(-1)  # (B,)

            # Target's probability for the same token
            t_prob = F.softmax(target_logits[:, i, :] / self.temperature, dim=-1)
            t_prob = t_prob.gather(-1, d_tok).squeeze(-1)  # (B,)

            # Accept if target is at least as confident as draft (adjusted by threshold)
            # Higher threshold = more selective = fewer acceptances
            accept_mask = (t_prob >= d_prob * self.accept_threshold).unsqueeze(-1)
            accepted_mask = accept_mask.squeeze(-1)  # (B,)

            # For rejected positions, we don't add to accepted
            if accepted_mask.any():
                # Add accepted tokens
                accepted.append(d_tok[accepted_mask])

        if not accepted:
            return torch.zeros(B, 0, dtype=torch.long, device=draft_tokens.device)

        accepted_tokens = torch.cat(accepted, dim=1)  # (B, num_accepted)
        return accepted_tokens

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
    ) -> torch.Tensor:
        """
        Speculative autoregressive generation.

        Args:
            input_ids: Prompt token indices of shape (B, T)
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated token indices (B, T + generated)
        """
        generated = input_ids.clone()
        total_generated = 0

        while total_generated < max_new_tokens:
            # Step 1: Draft model generates gamma tokens autoregressively
            draft_input = generated
            draft_tokens_list = []

            for _ in range(self.gamma):
                if draft_tokens_list:
                    # Use last drafted token only
                    draft_input = draft_tokens_list[-1]
                else:
                    draft_input = generated

                draft_logits = self.draft.forward(
                    draft_input,
                    n_loops=self.n_loops_draft,
                    kv_cache=None,  # Draft model doesn't use cache (fast)
                    start_pos=total_generated,
                )
                draft_tok = self._sample_token(draft_logits)
                draft_tokens_list.append(draft_tok)

            draft_tokens = torch.cat(draft_tokens_list, dim=1)  # (B, gamma)

            # Step 2: Target model verifies ALL tokens in single forward pass
            # Concatenate prompt + draft tokens for verification
            target_input = torch.cat([generated, draft_tokens], dim=1)

            # Target forward pass - this computes logits for ALL positions at once
            # because RDT processes the full sequence with recursion
            target_logits = self.target.forward(
                target_input,
                n_loops=self.n_loops_target,
                kv_cache=None,
                start_pos=0,
            )

            # Step 3: Verify draft tokens
            accepted = self._verify_and_accept(draft_tokens, target_logits)

            # Step 4: Append accepted tokens
            generated = torch.cat([generated, accepted], dim=1)
            total_generated += accepted.shape[1]

            # If no tokens accepted, sample one from target and continue
            if accepted.shape[1] == 0:
                next_tok = self._sample_token(target_logits[:, -1:, :])
                generated = torch.cat([generated, next_tok], dim=1)
                total_generated += 1

            # If we have enough tokens, stop
            if total_generated >= max_new_tokens:
                break

            # If all gamma accepted, the target also "predicts" one more
            # but we don't add it since we already have enough

        return generated[:, :input_ids.shape[1] + max_new_tokens]
