# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run a single test file
pytest test_main.py -v
pytest tests/test_tokenizer.py -v

# Run a single test class or function
pytest test_main.py::TestOpenMythosGQA -v
pytest test_main.py::TestLTIInjection::test_spectral_radius_lt_1 -v

# Lint
ruff check .
black --check .

# Format
black .

# Training (single GPU)
python training/3b_fine_web_edu.py

# Training (multi-GPU via DDP)
torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") training/3b_fine_web_edu.py
```

## Architecture

OpenMythos is a **Recurrent-Depth Transformer (RDT)** — a theoretical reconstruction of the Claude Mythos architecture. The core idea: instead of stacking unique layers, a single recurrent block is reused across multiple loop iterations per forward pass. More loops at inference = deeper reasoning.

### Three-stage forward pass

```
Input tokens → Embedding
  ↓
[Prelude]          — prelude_layers standard TransformerBlocks (dense FFN), run once
  ↓
[RecurrentBlock]   — one TransformerBlock (MoE FFN) looped T times
  ↑_______↓         h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
  ↓
[Coda]             — coda_layers standard TransformerBlocks (dense FFN), run once
  ↓
Output logits (with weight-tied embedding/head)
```

The Prelude output is frozen as `e` (encoded input) and injected at every loop step to prevent hidden-state drift.

### Key modules (`open_mythos/main.py`)

- **`MythosConfig`** — all hyperparameters. `attn_type="mla"|"gqa"` switches the attention class globally.
- **`GQAttention`** — Grouped Query Attention with `n_kv_heads < n_heads`. KV cache stores full K/V tensors.
- **`MLAttention`** — Multi-Latent Attention (DeepSeek-V2 style). Cache stores only the compressed latent `c_kv` (shape `kv_lora_rank`) and `k_rope`, achieving ~10–20× smaller KV cache than GQA at the same sequence length.
- **`MoEFFN`** — Fine-grained MoE: `n_experts` small routed experts + `n_shared_experts` always-active shared experts. Load balancing is aux-loss-free: a per-expert `router_bias` buffer (not a gradient parameter) shifts routing decisions without distorting the loss.
- **`LTIInjection`** — Guarantees spectral radius ρ(A) < 1 by construction via ZOH discretization: `A = exp(-exp(log_dt + log_A))`. This makes training stable regardless of learning rate. Always verify `model.recurrent.injection.get_A().max() < 1`.
- **`ACTHalting`** — Adaptive Computation Time: per-position halting probability accumulated across loop steps. Positions that converge early stop contributing (weighted sum of hidden states). With a KV cache, all loop iterations must run even if all positions halt, to keep cached keys populated.
- **`LoRAAdapter`** — Depth-wise LoRA: shared down/B matrices, per-loop scale embedding. Clamps loop index at `max_loop_iters - 1` during inference-time depth extrapolation.
- **`loop_index_embedding`** — Sinusoidal injection into the first `dim//8` channels of `h` to differentiate recurrent block behavior across iterations (analogous to RoPE for sequence position).
- **`RecurrentBlock`** — Wires the above together. At each loop: inject loop-index embedding → TransformerBlock → LoRA delta → LTI update → ACT halt check.

### Secondary model (`open_mythos/moda.py`)

A separate standalone decoder-only LM implementing **Mixture-of-Depths Attention (MoDA)** + DeepSeek-V3-style MoE FFN. Not integrated with the main `OpenMythos` class. Each query attends jointly to current-layer sequence KVs (causal) and depth KVs from all preceding layers under a single softmax. Configured via `MoDAConfig`, built as `MoDAModel`.

### Attention type differences

| | GQA | MLA |
|---|---|---|
| Cache contents | full K, V tensors | compressed `c_kv` + `k_rope` |
| RoPE applied to | full `head_dim` | only `qk_rope_head_dim` (decoupled) |
| `freqs_cis` used | `freqs_cis` | `freqs_cis_mla` |
| Extra config fields | `n_kv_heads` | `kv_lora_rank`, `q_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`, `v_head_dim` |

When constructing `MythosConfig` with `attn_type="mla"`, all MLA fields must be provided even when testing with GQA (see test helper `gqa_cfg()` in `test_main.py`).

### Pre-configured variants (`open_mythos/variants.py`)

Seven scale presets from `mythos_1b()` to `mythos_1t()`, all returning `MythosConfig`. Larger variants use `rope_theta=1_000_000` or `2_000_000` and longer `max_seq_len` (up to 1M tokens).

### Tokenizer (`open_mythos/tokenizer.py`)

`MythosTokenizer` wraps HuggingFace `AutoTokenizer` for `openai/gpt-oss-20b`. Used in the training script. Fetching this tokenizer requires network access.

### Public API (`open_mythos/__init__.py`)

All major classes and variant factory functions are exported from the package root. The training script imports directly from `open_mythos.main` and `open_mythos.tokenizer`.
