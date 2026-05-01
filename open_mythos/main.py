"""
OpenMythos v1 — DeepSeek-V3 Native Architecture (MLX)
Robust Loader: Filter weights to match model's internal parameter structure exactly.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import glob
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class MythosConfig:
    vocab_size: int = 102400
    dim: int = 2048
    n_heads: int = 16
    max_seq_len: int = 4096
    max_loop_iters: int = 16
    prelude_layers: int = 2
    coda_layers: int = 2
    attn_type: str = "mla"
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128
    n_experts: int = 64
    n_shared_experts: int = 2
    n_experts_per_tok: int = 6
    expert_dim: int = 1408
    rope_theta: float = 10000.0
    # ── upstream config-compatibility fields (MLX arch does not use these yet) ──
    # GQA head count (MLA path ignores this; kept for variants.py compatibility)
    n_kv_heads: int = 0
    # ACT halting threshold (reserved for future implementation)
    act_threshold: float = 0.99
    # Per-loop depth-wise LoRA rank (0 = disabled)
    lora_rank: int = 0
    # Maximum tokens to generate per forward pass
    max_output_tokens: int = 4096
    # Dropout probability (0.0 = disabled)
    dropout: float = 0.0

# ---------------------------------------------------------------------------
# Utils & Core Layers
# ---------------------------------------------------------------------------

def precompute_rope_freqs(dim: int, max_len: int, theta: float = 10000.0) -> mx.array:
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    t = mx.arange(max_len).astype(mx.float32)
    return mx.outer(t, freqs)

def apply_rope(x: mx.array, freqs: mx.array) -> mx.array:
    B, L, H, D = x.shape
    x1, x2 = x[..., :D//2], x[..., D//2:]
    cos, sin = mx.cos(freqs[:L])[None, :, None, :], mx.sin(freqs[:L])[None, :, None, :]
    return mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))
    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)

class MLAttention(nn.Module):
    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qk_rope_head_dim, self.qk_nope_head_dim = cfg.qk_rope_head_dim, cfg.qk_nope_head_dim
        self.kv_lora_rank = cfg.kv_lora_rank
        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim), bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(cfg.dim, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(cfg.kv_lora_rank)
        self.kv_b_proj = nn.Linear(cfg.kv_lora_rank, cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.v_head_dim, cfg.dim, bias=False)
        self.scale = (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) ** -0.5

    def __call__(self, x, freqs, mask=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1)
        q_nope, q_rope = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        q_rope = apply_rope(q_rope, freqs)
        q = mx.concatenate([q_nope, q_rope], axis=-1)
        kv_comp = self.kv_a_proj_with_mqa(x)
        kv_lat, k_rope = mx.split(kv_comp, [self.kv_lora_rank], axis=-1)
        kv_lat = self.kv_a_layernorm(kv_lat)
        kv = self.kv_b_proj(kv_lat).reshape(B, L, self.n_heads, -1)
        k_nope, v = mx.split(kv, [self.qk_nope_head_dim], axis=-1)
        k_rope = apply_rope(mx.repeat(k_rope.reshape(B, L, 1, -1), self.n_heads, axis=2), freqs)
        k = mx.concatenate([k_nope, k_rope], axis=-1)
        scores = (q.transpose(0, 2, 1, 3) @ k.transpose(0, 2, 3, 1)) * self.scale
        if mask is not None: scores += mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
        out = (scores @ v.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)

class Expert(nn.Module):
    def __init__(self, dim, h_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, h_dim, bias=False)
        self.up_proj = nn.Linear(dim, h_dim, bias=False)
        self.down_proj = nn.Linear(h_dim, dim, bias=False)
    def __call__(self, x):
        return self.down_proj(nn.SiLU()(self.gate_proj(x)) * self.up_proj(x))

class MLP(nn.Module):
    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.dim, cfg.expert_dim, bias=False)
        self.up_proj = nn.Linear(cfg.dim, cfg.expert_dim, bias=False)
        self.down_proj = nn.Linear(cfg.expert_dim, cfg.dim, bias=False)
    def __call__(self, x):
        return self.down_proj(nn.SiLU()(self.gate_proj(x)) * self.up_proj(x))

class DeepSeekMoE(nn.Module):
    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        self.shared_experts = Expert(cfg.dim, cfg.expert_dim * cfg.n_shared_experts)
        self.switch_mlp = [Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_experts)]

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape
        x_f = x.reshape(-1, D)
        w = mx.softmax(self.gate(x_f).astype(mx.float32), axis=-1)
        idx = mx.argpartition(-w, 4, axis=-1)[:, :4]
        out = mx.zeros_like(x_f)
        for i, expert in enumerate(self.switch_mlp):
            m = mx.any(idx == i, axis=-1, keepdims=True)
            if mx.any(m): out = mx.where(m, out + expert(x_f), out)
        return (out + self.shared_experts(x_f)).reshape(B, L, D)

# ---------------------------------------------------------------------------
# Blocks & Model
# ---------------------------------------------------------------------------

class MythosBlock(nn.Module):
    def __init__(self, cfg: MythosConfig, is_moe: bool = True):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.dim)
        self.self_attn = MLAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.dim)
        self.mlp = DeepSeekMoE(cfg) if is_moe else MLP(cfg)
    def __call__(self, x, freqs, mask=None):
        x = x + self.self_attn(self.input_layernorm(x), freqs, mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class OpenMythos(nn.Module):
    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.prelude = [MythosBlock(cfg, is_moe=(i > 0)) for i in range(cfg.prelude_layers)]
        self.recurrent_block = MythosBlock(cfg, is_moe=True)
        self.coda = [MythosBlock(cfg, is_moe=True) for _ in range(cfg.coda_layers)]
        self.norm = RMSNorm(cfg.dim)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.freqs = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta)

    def __call__(self, tokens, n_loops=None):
        x = self.tok_embeddings(tokens)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        for layer in self.prelude: x = layer(x, self.freqs, mask)
        for _ in range(n_loops or self.cfg.max_loop_iters):
            x = self.recurrent_block(x, self.freqs, mask)
        for layer in self.coda: x = layer(x, self.freqs, mask)
        return self.output(self.norm(x))

    def generate(self, tokens, max_new_tokens=8, n_loops=4):
        for _ in range(max_new_tokens):
            logits = self(tokens, n_loops=n_loops)
            next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            tokens = mx.concatenate([tokens, next_token], axis=1)
            mx.eval(tokens)
        return tokens

# ---------------------------------------------------------------------------
# Weight Loader
# ---------------------------------------------------------------------------

_BATCHED_PROJ_SUFFIXES = {"switch_mlp.gate_proj", "switch_mlp.up_proj", "switch_mlp.down_proj"}


def _resolve_target_key(cfg: MythosConfig, layer_idx: int, suffix: str) -> Optional[str]:
    if layer_idx < cfg.prelude_layers:
        return f"prelude.{layer_idx}.{suffix}"
    elif layer_idx == cfg.prelude_layers:
        return f"recurrent_block.{suffix}"
    elif layer_idx < cfg.prelude_layers + 1 + cfg.coda_layers:
        coda_idx = layer_idx - cfg.prelude_layers - 1
        return f"coda.{coda_idx}.{suffix}"
    return None


def _build_nested_dict(flat_params: dict) -> dict:
    result: dict = {}
    for k, v in flat_params.items():
        if v is None:
            continue
        parts = k.split(".")
        d = result
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return _dicts_to_lists(result)


def _dicts_to_lists(d):
    """Recursively convert dicts whose keys are consecutive ints (0,1,2,...) to lists."""
    if not isinstance(d, dict):
        return d
    converted = {k: _dicts_to_lists(v) for k, v in d.items()}
    if all(k.isdigit() for k in converted):
        max_idx = max(int(k) for k in converted)
        if set(converted.keys()) == {str(i) for i in range(max_idx + 1)}:
            return [converted[str(i)] for i in range(max_idx + 1)]
    return converted


def load_deepseek_v3_subset(model: OpenMythos, mlx_path: str):
    weight_files = sorted(glob.glob(os.path.join(mlx_path, "*.safetensors")))
    print(f"Loading weights from {len(weight_files)} files...")
    weights: dict = {}
    for f in weight_files:
        weights.update(mx.load(f))

    valid_keys = set(k for k, _ in mlx.utils.tree_flatten(model.parameters()))

    new_params: dict = {
        "tok_embeddings.weight": weights.get("model.embed_tokens.weight"),
        "norm.weight": weights.get("model.norm.weight"),
        "output.weight": weights.get("lm_head.weight"),
    }

    for k, v in weights.items():
        if not k.startswith("model.layers."):
            continue
        parts = k.split(".")
        layer_idx = int(parts[2])
        suffix = ".".join(parts[3:])

        # Batched MoE weights: shape (n_experts, ...) → split per expert
        # e.g. "mlp.switch_mlp.gate_proj.weight" → "mlp.switch_mlp.{i}.gate_proj.weight"
        proj_middle = ".".join(suffix.split(".")[:3])  # "mlp.switch_mlp.gate_proj"
        if proj_middle in {f"mlp.{p}" for p in _BATCHED_PROJ_SUFFIXES}:
            for i in range(v.shape[0]):
                per_expert_suffix = suffix.replace("switch_mlp.", f"switch_mlp.{i}.", 1)
                target_key = _resolve_target_key(model.cfg, layer_idx, per_expert_suffix)
                if target_key and target_key in valid_keys:
                    new_params[target_key] = v[i]
            continue

        target_key = _resolve_target_key(model.cfg, layer_idx, suffix)
        if target_key and target_key in valid_keys:
            new_params[target_key] = v

    final_dict = _build_nested_dict(new_params)
    model.update(final_dict)
    mx.eval(model.parameters())
    loaded = sum(1 for v in new_params.values() if v is not None)
    print(f"Loaded {loaded} parameters successfully.")
