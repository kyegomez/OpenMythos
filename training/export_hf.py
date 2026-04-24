"""
HuggingFace Export for OpenMythos
==================================

Export OpenMythos model + tokenizer to HuggingFace format.

Usage:
    # From trained weights (after training/train_p0_p3.py):
    python training/export_hf.py \
        --checkpoint training/checkpoints/openmythos-3b-step100.pt \
        --output huggingface/openmythos-3b \
        --config-name openmythos-3b

    # Without real weights — creates a random-initialized model for testing:
    python training/export_hf.py \
        --output huggingface/openmythos-3b-random \
        --config-name openmythos-3b \
        --random-init

HuggingFace model card will be written to output_dir/README.md.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    from transformers import PreTrainedModel, PretrainedConfig
    from transformers.modeling_outputs import CausalLMOutputWithPast

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    print("ERROR: torch and transformers are required for HuggingFace export")
    sys.exit(1)


# ---------------------------------------------------------------------------
# HuggingFace Config
# ---------------------------------------------------------------------------

class OpenMythosHFConfig(PretrainedConfig):
    """
    HuggingFace PretrainedConfig for OpenMythos.

    Maps to/from MythosConfig (open_mythos/main.py) so that
    `OpenMythosForCausalLM.from_pretrained(path)` works.
    """

    model_type = "openmythos"

    def __init__(
        self,
        # Core
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        intermediate_size: int = 5504,  # ≈ 4 * dim / n_experts_per_tok ratio
        num_hidden_layers: int = 28,    # prelude + recurrent + coda ≈ 2 + T + 2
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        max_position_embeddings: int = 4096,
        # OpenMythos-specific
        max_loop_iters: int = 16,
        prelude_layers: int = 2,
        coda_layers: int = 2,
        attn_type: str = "mla",
        kv_lora_rank: int = 512,
        q_lora_rank: int = 1536,
        qk_rope_head_dim: int = 64,
        qk_nope_head_dim: int = 128,
        v_head_dim: int = 128,
        n_experts: int = 64,
        n_shared_experts: int = 2,
        n_experts_per_tok: int = 4,
        expert_dim: int = 512,
        rope_theta: float = 500000.0,
        loop_depths: tuple = (4, 8, 16),
        # Optional features
        use_hierarchical: bool = False,
        use_meta_loop: bool = False,
        use_path_cost: bool = False,
        use_cone: bool = False,
        use_bundle_memory: bool = False,
        use_procedural: bool = False,
        # HF standard
        rms_norm_eps: float = 1e-6,
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        # OpenMythos
        self.max_loop_iters = max_loop_iters
        self.prelude_layers = prelude_layers
        self.coda_layers = coda_layers
        self.attn_type = attn_type
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.n_experts = n_experts
        self.n_shared_experts = n_shared_experts
        self.n_experts_per_tok = n_experts_per_tok
        self.expert_dim = expert_dim
        self.rope_theta = rope_theta
        self.loop_depths = list(loop_depths)
        self.use_hierarchical = use_hierarchical
        self.use_meta_loop = use_meta_loop
        self.use_path_cost = use_path_cost
        self.use_cone = use_cone
        self.use_bundle_memory = use_bundle_memory
        self.use_procedural = use_procedural
        self.rms_norm_eps = rms_norm_eps
        self.torch_dtype = torch_dtype
        super().__init__(**kwargs)

    @classmethod
    def from_mythos_config(cls, cfg) -> "OpenMythosHFConfig":
        """Convert a MythosConfig (main.py) to OpenMythosHFConfig."""
        return cls(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.dim,
            intermediate_size=cfg.expert_dim,
            num_hidden_layers=cfg.prelude_layers + cfg.coda_layers + 1,  # +1 recurrent
            num_attention_heads=cfg.n_heads,
            num_key_value_heads=cfg.n_kv_heads,
            max_position_embeddings=cfg.max_seq_len,
            max_loop_iters=cfg.max_loop_iters,
            prelude_layers=cfg.prelude_layers,
            coda_layers=cfg.coda_layers,
            attn_type=cfg.attn_type,
            kv_lora_rank=cfg.kv_lora_rank,
            q_lora_rank=cfg.q_lora_rank,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
            v_head_dim=cfg.v_head_dim,
            n_experts=cfg.n_experts,
            n_shared_experts=cfg.n_shared_experts,
            n_experts_per_tok=cfg.n_experts_per_tok,
            expert_dim=cfg.expert_dim,
            rope_theta=cfg.rope_theta,
            loop_depths=cfg.loop_depths,
            use_hierarchical=False,
            use_meta_loop=False,
            use_path_cost=False,
            use_cone=False,
        )

    def to_mythos_config(self):
        """Convert back to MythosConfig dict for reconstruction."""
        from open_mythos.main import MythosConfig
        cfg = MythosConfig()
        cfg.vocab_size = self.vocab_size
        cfg.dim = self.hidden_size
        cfg.n_heads = self.num_attention_heads
        cfg.n_kv_heads = self.num_key_value_heads
        cfg.max_seq_len = self.max_position_embeddings
        cfg.max_loop_iters = self.max_loop_iters
        cfg.prelude_layers = self.prelude_layers
        cfg.coda_layers = self.coda_layers
        cfg.attn_type = self.attn_type
        cfg.kv_lora_rank = self.kv_lora_rank
        cfg.q_lora_rank = self.q_lora_rank
        cfg.qk_rope_head_dim = self.qk_rope_head_dim
        cfg.qk_nope_head_dim = self.qk_nope_head_dim
        cfg.v_head_dim = self.v_head_dim
        cfg.n_experts = self.n_experts
        cfg.n_shared_experts = self.n_shared_experts
        cfg.n_experts_per_tok = self.n_experts_per_tok
        cfg.expert_dim = self.expert_dim
        cfg.rope_theta = self.rope_theta
        cfg.loop_depths = tuple(self.loop_depths)
        return cfg


# ---------------------------------------------------------------------------
# HuggingFace Model Wrapper
# ---------------------------------------------------------------------------

class OpenMythosForCausalLM(PreTrainedModel):
    """
    HuggingFace PreTrainedModel wrapper for OpenMythosLM.

    Exposes the OpenMythos recurrent-depth transformer as a standard
    causal language model compatible with:
        - AutoModelForCausalLM.from_pretrained(...)
        - pipeline("text-generation", model=...)
        - generation_config.json

    Forward:
        input_ids: (B, T) token IDs
        attention_mask: (B, T) — optional, all-ones for OpenMythos
        Returns: CausalLMOutputWithPast with logits (B, T, vocab_size)
    """

    config_class = OpenMythosHFConfig
    base_model_prefix = "openmythos"
    supports_gradient_checkpointing = True

    def __init__(self, config: OpenMythosHFConfig):
        super().__init__(config)
        cfg = config.to_mythos_config()

        # Import here to avoid hard torch dependency at module load
        from open_mythos.main import OpenMythos
        from training.train_p0_p3 import OpenMythosLM

        self.openmythos = OpenMythos(cfg)
        self.lm = OpenMythosLM(self.openmythos, tie_weights=True)

        # Tie weights (already done in OpenMythosLM, but ensure)
        self.lm.head = None  # will use embed.weight

        # Generation config (HF standard)
        self.generation_config = self._make_generation_config(config)

        self.post_init()

    def _make_generation_config(self, config: OpenMythosHFConfig):
        from transformers import GenerationConfig
        return GenerationConfig(
            max_length=config.max_position_embeddings,
            max_new_tokens=config.max_position_embeddings // 4,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=2,  # </s> typically
            pad_token_id=0,   # <pad>
        )

    def get_input_embeddings(self):
        return self.openmythos.embed

    def set_input_embeddings(self, value):
        self.openmythos.embed = value

    def get_output_embeddings(self):
        # Weight-tied: return embedding table
        return self.openmythos.embed

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Args:
            input_ids: (B, T) token indices
            attention_mask: (B, T) — optional; defaults to causal mask
        Returns:
            CausalLMOutputWithPast with logits (B, T, vocab_size)
        """
        result = self.lm(input_ids, targets=None)
        logits = result["logits"]

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=None,  # OpenMythos doesn't use HF KV cache format
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Simple greedy/softmax generation for OpenMythos.

        For full beam search / sampling use:
            from transformers import AutoModelForCausalLM
            model.generate(input_ids, max_new_tokens=128, ...)
        """
        self.eval()
        B, T = input_ids.shape
        device = input_ids.device
        max_len = T + max_new_tokens

        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(generated[:, -self.config.max_position_embeddings:])["logits"]
            next_logits = logits[:, -1, :]  # (B, V)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for b in range(B):
                    for idx in set(generated[b].tolist()):
                        next_logits[b, idx] /= repetition_penalty

            # Greedy
            next_token = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            if next_token.item() == 2:  # </s>
                break

            generated = torch.cat([generated, next_token], dim=1)

            if generated.shape[1] >= max_len:
                break

        return generated


# ---------------------------------------------------------------------------
# Tokenizer Export
# ---------------------------------------------------------------------------

def export_tokenizer(output_dir: str, tokenizer_model_id: str = "openai/gpt-oss-20b"):
    """
    Export tokenizer to output_dir/tokenizer.json + tokenizer_config.json.

    Uses the same tokenizer as MythosTokenizer — a HuggingFace AutoTokenizer.
    """
    from open_mythos.tokenizer import MythosTokenizer

    print(f"Exporting tokenizer from '{tokenizer_model_id}' ...")
    mt = MythosTokenizer(model_id=tokenizer_model_id)
    tok = mt.tokenizer

    tok.save_pretrained(output_dir)
    print(f"  Tokenizer saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Model Card
# ---------------------------------------------------------------------------

MODEL_CARD = """
---
language:
  - en
  - zh
library_name: transformers
pipeline_tag: causal-lm
license: apache-2.0
tags:
  - openmythos
  - recurrent-transformer
  - looped-transformer
  - multi-scale-recurrent
  - moe
  - mixture-of-experts
  - deepseek-r1-style
---

# OpenMythos

OpenMythos is a **Recurrent-Depth Transformer** (RDT) language model inspired by
DeepSeek-R1. Instead of stacking more transformer layers, it inserts a single
**recurrent transformer block** that is unrolled for T steps — achieving deep
reasoning with constant parameter count.

## Key Innovations

- **Recurrent Depth**: Same weights, T loops → emergent reasoning depth
- **Multi-Scale Loop**: Dynamic depth selection (4 / 8 / 16 tokens)
- **MoE FFN**: 64 experts, 4 active per token
- **MLA Attention**: Multi-Latent Attention for 40%+ KV cache reduction
- **LTI Injection**: Stable recurrent dynamics (spectral radius < 1)
- **M-flow Memory** (P4): Cone-lithic episodic memory with min-cost bundle search

## Architecture

```
Input → [Prelude: 2× TransformerBlock] → [RecurrentBlock: T× unrolled]
                                           ↑__________________↓ (LTI)
                                       → [Coda: 2× TransformerBlock] → logits
```

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("openmythos/openmythos-3b")
tokenizer = AutoTokenizer.from_pretrained("openmythos/openmythos-3b")

inputs = tokenizer("The answer is", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=64, temperature=0.7)
print(tokenizer.decode(out[0]))
```

## Training

See `training/train_p0_p3.py` for the full training pipeline including:
- Curriculum learning (multi-scale depth scheduling)
- Loop consistency loss (P1-3)
- Speculative decoding (P2)
- Hierarchical + MetaLoop recurrent blocks (P3)

## Citation

```bibtex
@article{openmythos2025,
  title={OpenMythos: Recurrent-Depth Transformer with M-flow Memory},
  author={OpenMythos Team},
  year={2025}
}
```
"""


# ---------------------------------------------------------------------------
# Main Export
# ---------------------------------------------------------------------------

def export(
    output_dir: str,
    checkpoint_path: Optional[str] = None,
    config_name: str = "openmythos-3b",
    random_init: bool = False,
    tokenizer_model_id: str = "openai/gpt-oss-20b",
):
    """Export OpenMythos to HuggingFace format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Build config
    config = OpenMythosHFConfig(
        hidden_size=2048,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        intermediate_size=512,
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=4,
        max_loop_iters=16,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=1536,
        vocab_size=32000,
    )

    # 2. Build model
    print("Building OpenMythosForCausalLM ...")
    model = OpenMythosForCausalLM(config)

    # 3. Load checkpoint if provided
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path} ...")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        # Support both full-state-dict and {"model_state_dict": ...}
        if "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        model.load_state_dict(ckpt, strict=False)
        print("  Checkpoint loaded.")

    if random_init:
        print("WARNING: Exporting random-initialized model (no real weights).")

    # 4. Save HuggingFace format
    print(f"Saving to {output_dir} ...")
    model.save_pretrained(output_dir)

    # 5. Save tokenizer
    export_tokenizer(output_dir, tokenizer_model_id)

    # 6. Save model card
    card_path = output_path / "README.md"
    card_path.write_text(MODEL_CARD.lstrip())
    print(f"  Model card: {card_path}")

    # 7. Save OpenMythos-specific config as OpenMythosHFConfig.json
    # (already saved by save_pretrained, but also save a copy with full info)
    cfg_json = config.to_dict()
    cfg_json["_name_or_path"] = config_name
    cfg_json["architectures"] = ["OpenMythosForCausalLM"]
    cfg_json["model_type"] = "openmythos"
    with open(output_path / "openmythos_config.json", "w") as f:
        json.dump(cfg_json, f, indent=2, default=list)

    print(f"\n✓ Export complete: {output_dir}")
    print(f"  - model.safetensors (or pytorch_model.bin)")
    print(f"  - tokenizer.json + tokenizer_config.json")
    print(f"  - config.json")
    print(f"  - generation_config.json")
    print(f"  - README.md (model card)")
    print(f"\nTo push to HuggingFace Hub:")
    print(f"  from huggingface_hub import HfApi")
    print(f"  api = HfApi()")
    print(f"  api.upload_folder(folder_path='{output_dir}', repo_id='YOUR_ID/{config_name}')")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export OpenMythos to HuggingFace format")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--checkpoint", "-c", help="Path to .pt checkpoint (optional)")
    parser.add_argument("--config-name", default="openmythos-3b", help="Model name for config")
    parser.add_argument("--random-init", action="true", help="Export with random weights (no checkpoint)")
    parser.add_argument("--tokenizer-model-id", default="openai/gpt-oss-20b",
                        help="HF model ID for tokenizer")

    args = parser.parse_args()

    if not _HAS_TORCH:
        print("ERROR: torch + transformers required")
        sys.exit(1)

    export(
        output_dir=args.output,
        checkpoint_path=args.checkpoint,
        config_name=args.config_name,
        random_init=getattr(args, "random_init", False),
        tokenizer_model_id=args.tokenizer_model_id,
    )
