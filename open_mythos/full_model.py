"""
DeepSeekV2Lite — Full 27-layer sequential inference model for OpenMythos MCP server.
Shares building blocks with main.py but runs all layers in order (no recurrent loop).
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import glob
import os
from typing import Optional

from open_mythos.main import (
    MythosConfig, RMSNorm, MLAttention, MLP, DeepSeekMoE,
    precompute_rope_freqs, _BATCHED_PROJ_SUFFIXES, _build_nested_dict,
)


class DeepSeekBlock(nn.Module):
    def __init__(self, cfg: MythosConfig, is_moe: bool):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.dim)
        self.self_attn = MLAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.dim)
        self.mlp = DeepSeekMoE(cfg) if is_moe else MLP(cfg)

    def __call__(self, x, freqs, mask=None):
        x = x + self.self_attn(self.input_layernorm(x), freqs, mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class DeepSeekV2Lite(nn.Module):
    """Full 27-layer DeepSeek-V2-Lite inference. Layer 0 is dense, 1-26 are MoE."""

    def __init__(self, cfg: MythosConfig, n_layers: int = 27):
        super().__init__()
        self.cfg = cfg
        self.n_layers = n_layers
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        # first_k_dense_replace=1: layer 0 is dense MLP, rest are MoE
        self.layers = [DeepSeekBlock(cfg, is_moe=(i > 0)) for i in range(n_layers)]
        self.norm = RMSNorm(cfg.dim)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.freqs = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta)

    def __call__(self, tokens: mx.array) -> mx.array:
        x = self.tok_embeddings(tokens)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        for layer in self.layers:
            x = layer(x, self.freqs, mask)
        return self.output(self.norm(x))

    def generate(self, tokens: mx.array, max_new_tokens: int = 128, temperature: float = 0.7) -> mx.array:
        for _ in range(max_new_tokens):
            logits = self(tokens)
            next_logits = logits[:, -1, :].astype(mx.float32)
            if temperature > 0:
                next_logits = next_logits / temperature
            next_token = mx.argmax(next_logits, axis=-1, keepdims=True)
            tokens = mx.concatenate([tokens, next_token], axis=1)
            mx.eval(tokens)
            # Stop on EOS (token 100001 for DeepSeek-V2)
            if int(next_token.item()) == 100001:
                break
        return tokens


def load_deepseek_v2_lite(mlx_path: str, cfg: Optional[MythosConfig] = None) -> DeepSeekV2Lite:
    """Load all 27 DeepSeek-V2-Lite layers into DeepSeekV2Lite model."""
    if cfg is None:
        cfg = MythosConfig()

    weight_files = sorted(glob.glob(os.path.join(mlx_path, "*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"No .safetensors files found in {mlx_path}")

    print(f"Loading {len(weight_files)} weight shards...")
    weights: dict = {}
    for f in weight_files:
        weights.update(mx.load(f))

    model = DeepSeekV2Lite(cfg)
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

        # Batched MoE: split (n_experts, ...) per expert
        proj_middle = ".".join(suffix.split(".")[:3])
        if proj_middle in {f"mlp.{p}" for p in _BATCHED_PROJ_SUFFIXES}:
            for i in range(v.shape[0]):
                per_expert_suffix = suffix.replace("switch_mlp.", f"switch_mlp.{i}.", 1)
                target_key = f"layers.{layer_idx}.{per_expert_suffix}"
                if target_key in valid_keys:
                    new_params[target_key] = v[i]
            continue

        target_key = f"layers.{layer_idx}.{suffix}"
        if target_key in valid_keys:
            new_params[target_key] = v

    final_dict = _build_nested_dict(new_params)
    model.update(final_dict)
    mx.eval(model.parameters())

    loaded = sum(1 for v in new_params.values() if v is not None)
    print(f"Loaded {loaded} parameters ({loaded / len(valid_keys) * 100:.1f}% of model).")
    return model
