# OpenMythos

<p align="left">
  <a href="https://pypi.org/project/open-mythos/" target="_blank">
    <picture>
      <source srcset="https://img.shields.io/pypi/v/open-mythos?style=for-the-badge&color=3670A0" media="(prefers-color-scheme: dark)">
      <img alt="Version" src="https://img.shields.io/pypi/v/open-mythos?style=for-the-badge&color=3670A0">
    </picture>
  </a>
  <a href="https://twitter.com/kyegomezb/">
    <picture>
      <source srcset="https://img.shields.io/badge/Twitter-Follow-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" media="(prefers-color-scheme: dark)">
      <img src="https://img.shields.io/badge/Twitter-Follow-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter">
    </picture>
  </a>
  <a href="https://discord.gg/3keGBK9Pvr" target="_blank">
    <picture>
      <source srcset="https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord&logoColor=white" media="(prefers-color-scheme: dark)">
      <img alt="Discord" src="https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord&logoColor=white">
    </picture>
  </a>
  <a href="https://pytorch.org" target="_blank">
    <picture>
      <source srcset="https://img.shields.io/badge/PyTorch-Implemented-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" media="(prefers-color-scheme: dark)">
      <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Implemented-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
    </picture>
  </a>
</p>

> **Disclaimer:** OpenMythos is an independent, community-driven theoretical reconstruction based solely on publicly available research and speculation. It is not affiliated with, endorsed by, or connected to Anthropic or any of their proprietary systems.

OpenMythos is an open-source, theoretical implementation of the Claude Mythos model. It implements a Recurrent-Depth Transformer (RDT) with three stages: **Prelude** (transformer blocks), a looped **Recurrent Block** (up to `max_loop_iters`), and a final **Coda**. Attention is switchable between MLA and GQA, and the feed-forward uses a sparse MoE with routed and shared experts ideal for exploring compute-adaptive, depth-variable reasoning.


## Installation

```bash
pip install open-mythos

#uv pip install open-mythos
```

To enable Flash Attention 2 in `GQAttention` (requires CUDA and build tools):

```bash
pip install open-mythos[flash]
```

## Usage

```python

import torch
from open_mythos.main import OpenMythos, MythosConfig


attn_type = "mla"  # or "gqa"

base = {
    "vocab_size": 1000,
    "dim": 256,
    "n_heads": 8,
    "max_seq_len": 128,
    "max_loop_iters": 4,
    "prelude_layers": 1,
    "coda_layers": 1,
    "n_experts": 8,
    "n_shared_experts": 1,
    "n_experts_per_tok": 2,
    "expert_dim": 64,
    "lora_rank": 8,
    "attn_type": attn_type,
}

if attn_type == "gqa":
    cfg = MythosConfig(**base, n_kv_heads=2)
else:
    cfg = MythosConfig(
        **base,
        n_kv_heads=8,
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
    )

model = OpenMythos(cfg)
total = sum(p.numel() for p in model.parameters())
print(f"\n[{attn_type.upper()}] Parameters: {total:,}")

ids = torch.randint(0, cfg.vocab_size, (2, 16))
logits = model(ids, n_loops=4)
print(f"[{attn_type.upper()}] Logits shape: {logits.shape}")

out = model.generate(ids, max_new_tokens=8, n_loops=8)
print(f"[{attn_type.upper()}] Generated shape: {out.shape}")

A = model.recurrent.injection.get_A()
rho = torch.linalg.eigvals(A).abs().max().item()
print(
    f"[{attn_type.upper()}] Spectral radius ρ(A) = {rho:.4f} (must be < 1)"
)
```



## Model Variants

Pre-configured scales from 1B to 1T parameters:

```python
from open_mythos import (
    mythos_1b,
    mythos_3b,
    mythos_10b,
    mythos_50b,
    mythos_100b,
    mythos_500b,
    mythos_1t,
    OpenMythos,
)

cfg = mythos_7b()  # returns a MythosConfig
model = OpenMythos(cfg)

total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total:,}")
```

| Variant | `dim` | Experts | `expert_dim` | Loop iters | Context | Max output |
|---|---|---|---|---|---|---|
| `mythos_1b` | 2048 | 64 | 2048 | 16 | 4k | 4k |
| `mythos_3b` | 3072 | 64 | 4096 | 16 | 4k | 4k |
| `mythos_10b` | 4096 | 128 | 5632 | 24 | 8k | 4k |
| `mythos_50b` | 6144 | 256 | 9728 | 32 | 8k | 4k |
| `mythos_100b` | 8192 | 256 | 13568 | 32 | 1M | 128k |
| `mythos_500b` | 12288 | 512 | 23040 | 48 | 1M | 128k |
| `mythos_1t` | 16384 | 512 | 34560 | 64 | 1M | 128k |

---

## Training

The training script for the 3B model on FineWeb-Edu is at [`training/3b_fine_web_edu.py`](training/3b_fine_web_edu.py).

**Single GPU:**
```bash
python training/3b_fine_web_edu.py
```

**Multi-GPU (auto-detects GPU count):**
```bash
torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") training/3b_fine_web_edu.py
```

Key design choices:

| Feature | Detail |
|---|---|
| Optimizer | AdamW |
| Dataset | `HuggingFaceFW/fineweb-edu` (`sample-10BT` by default, swap to `sample-100BT` or `default` for full run) |
| Tokenizer | `openai/gpt-oss-20b` via `MythosTokenizer` |
| Parallelism | PyTorch DDP via `torchrun`, sharded streaming dataset |
| Precision | bfloat16 on H100/A100, float16 + GradScaler on older GPUs |
| Schedule | Linear warmup (2000 steps) → cosine decay |
| Target | 30B tokens (~Chinchilla-adjusted for looped architecture) |

---

## Documentation

| Page | Description |
|---|---|
| [`docs/open_mythos.md`](docs/open_mythos.md) | Full API reference for the `OpenMythos` class — constructor, `forward`, `generate`, all sub-modules, configuration reference, and usage examples |
| [`docs/datasets.md`](docs/datasets.md) | Recommended training datasets with token budget guidance per model size |

---

## The Central Hypothesis

Claude Mythos is suspected to be a **Recurrent-Depth Transformer (RDT)** — also called a Looped Transformer (LT). Rather than stacking hundreds of unique layers, a subset of layers is recycled and run through multiple times per forward pass. Same weights. More loops. Deeper thinking.

This is not chain-of-thought. There is no intermediate token output. All of this reasoning happens **silently, inside a single forward pass**, in continuous latent space.

---

## Architecture

A looped transformer divides its layers into three functional blocks:

```
Input
  ↓
[Prelude P]        — standard transformer layers, run once
  ↓
[Recurrent Block R] — looped T times
  ↑_______↓         (hidden state h updated each loop with input injection e)
  ↓
[Coda C]           — standard transformer layers, run once
  ↓
Output
```

The recurrent block update rule at each loop step t:

```
h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
```

Where:
- `h_t` is the hidden state after loop t
- `e` is the encoded input (from the Prelude), injected at every loop
- `A` and `B` are learned injection parameters
- The Transformer blocks apply attention and MLP as usual

The injection of `e` at every step is what prevents the model from drifting — it keeps the original input signal alive throughout the entire recurrence depth.

The full implementation is in [`open_mythos/main.py`](open_mythos/main.py). See the [`OpenMythos` class reference](docs/open_mythos.md) for a detailed API walkthrough, configuration options, and usage examples.

### Attention Implementations

The attention layer is switchable via `cfg.attn_type`:

| Option | Class | Description |
|---|---|---|
| `"gqa"` | `GQAttention` | Grouped Query Attention (Ainslie et al., 2023) — fewer KV heads than Q heads (`n_kv_heads < n_heads`), reducing KV-cache memory by `n_heads / n_kv_heads`. Uses **Flash Attention 2** (Dao et al., 2023) when `flash-attn>=2.8.3` is installed: GQA is handled natively (no KV head expansion), I/O-bound-optimal, with a transparent fallback to manual scaled dot-product attention when the package is absent. |
| `"mla"` | `MLAttention` | Multi-Latent Attention (DeepSeek-V2) — caches a compressed KV latent (`kv_lora_rank`) rather than full K/V, with split RoPE / no-RoPE head dims for position-aware compression. |

RoPE is applied to Q and K before caching, so cached values do not need to be re-rotated on retrieval.

---

## Why This Explains Mythos

### 1. Systematic Generalization

Vanilla transformers fail to combine knowledge in ways they have never seen during training. Looped transformers pass this test. The ability emerges through a **three-stage grokking process**:

1. Memorization — model fits training distribution
2. In-distribution generalization — model handles known compositions
3. Systematic generalization — model handles novel compositions OOD, abruptly and suddenly

This is why Mythos feels qualitatively different from other models on novel questions — the capability phase-transitions in, rather than emerging gradually.

### 2. Depth Extrapolation

Train on 5-hop reasoning chains. Test on 10-hop. Vanilla transformer fails. Looped transformer succeeds — by running more inference-time loops. This maps directly to the observation that Mythos handles deeply compositional problems (multi-step math, long-horizon planning, layered arguments) without explicit chain-of-thought.

More loops at inference = deeper reasoning chains = harder problems solved.

### 3. Latent Thoughts as Implicit Chain-of-Thought

Each loop iteration is the functional equivalent of one step of chain-of-thought, but operating in continuous latent space rather than token space. A looped model running T loops implicitly simulates T steps of CoT reasoning. This has been formally proven (Saunshi et al., 2025).

Furthermore, continuous latent thoughts — unlike discrete token outputs — can encode **multiple alternative next steps simultaneously**. This allows something closer to breadth-first search over the reasoning space, rather than a single committed reasoning path. The model is effectively exploring many possible directions inside each forward pass before converging.

### 4. No Parameter Explosion

A looped model with k layers run L times achieves the quality of a kL-layer non-looped model, with only k layers worth of parameters. For Mythos-scale deployments, this matters enormously:

- Memory footprint does not grow with reasoning depth
- Inference-time compute scales with loop count, not model size
- This makes deeper reasoning "free" in terms of parameters

---

## The Stability Problem (and How It Was Likely Solved)

Training looped models is notoriously unstable. Two failure modes dominate:

- **Residual explosion** — the hidden state `h_t` grows unboundedly across loops
- **Loss spikes** — training diverges suddenly due to large spectral norms in injection parameters

### The Dynamical Systems View

Recast looping as a discrete linear time-invariant (LTI) dynamical system over the residual stream. Ignoring the nonlinear Transformer contribution, the recurrence becomes:

```
h_{t+1} = A·h_t + B·e
```

For this LTI system, stability is governed entirely by the **spectral radius** of A:
- `ρ(A) < 1` → stable, convergent
- `ρ(A) ≥ 1` → unstable, divergent

Empirically, every divergent training run learns `ρ(A) ≥ 1`. Every convergent run maintains `ρ(A) < 1`.

### The Fix

Constrain the injection parameters so that stability is guaranteed **by construction**:

1. Parameterize A as a continuous negative diagonal matrix
2. Discretize using ZOH/Euler schemes: `A_discrete = exp(Δt · A_continuous)`
3. Enforce negativity via `A := Diag(-exp(log_A))` with a learned scalar `Δt`
4. This ensures `ρ(A) < 1` always holds, regardless of learning rate or batch noise

The result: the looped model becomes significantly more robust to hyperparameter selection and trains cleanly even at high learning rates. This is the Parcae architecture (Prairie et al., 2026), and it represents the most likely class of solution Anthropic used to make Mythos trainable.

---

## Scaling Laws for Looped Models

Parcae establishes the first predictable scaling laws for looped training:

- **Training**: For a fixed FLOP budget with fixed parameters, increasing mean recurrence and reducing token count yields a lower loss than training with minimal loops on more data. Optimal recurrence and optimal token count both follow **power laws** with consistent exponents across scales.
- **Inference**: More test-time loops improves quality following a **predictable, saturating exponential decay** — gains are real but diminishing. This mirrors the inference-time scaling of chain-of-thought.

At 770M parameters, a looped model achieves the downstream quality of a 1.3B fixed-depth Transformer trained on the same data — roughly **half the parameters for the same quality**.

Applied to Mythos: if trained under these scaling laws, Mythos could be dramatically more parameter-efficient than it appears, with a large fraction of its apparent "capability" coming from loop depth rather than raw parameter count.

---

## The Loop Index Embedding Hypothesis

A key open question is whether the looped block behaves **identically** on every iteration, or whether it can learn to do different things at different loop depths.

Without any positional signal across loops, the same weights must handle both early-stage pattern matching and late-stage refinement — a tight constraint. A **RoPE-like embedding of the loop index** injected alongside the input at each step would allow the same parameters to implement functionally distinct operations across iterations, much like how RoPE allows the same attention heads to behave differently at different sequence positions.

If Mythos uses this technique, each loop is not a repetition — it is a distinct computational phase, all sharing weights but operating in different representational regimes. This would substantially increase the expressiveness of the recurrent block without increasing parameter count.

---

## The Overthinking Problem

More loops is not always better. Beyond a certain depth, excessive recurrence **degrades predictions** — the hidden state drifts past the solution and into noise. This is the "overthinking" failure mode.

The original Universal Transformer (Dehghani et al., 2018) addressed this with an **Adaptive Computation Time (ACT)** halting mechanism: a learned scalar per position that dynamically decides when to stop looping. Positions that are harder to process receive more computation; simple tokens halt early.

Mythos almost certainly has some version of this. The model cannot naively run the maximum number of loops on every input — it needs a learned signal for when the answer has converged. The ACT mechanism also makes the model **Turing-complete** under certain assumptions, which has theoretical implications for the class of problems it can solve.

---

## Mixture of Experts — Suspected for Large Parameter Counts

The looped transformer explains the depth of Mythos's reasoning, but not the breadth. Handling wildly different domains — code, math, literature, science, law — with the same weights requires **Mixture of Experts (MoE)**. The suspected design replaces every FFN in the Recurrent Block with a fine-grained MoE layer: each FFN is split into many small experts (1/m the normal size), a router selects the top-mK of them per token via learned affinity scores, and a small number of **shared experts** are always activated regardless of routing to absorb common cross-domain knowledge — syntax, basic reasoning, general context — that would otherwise be redundantly learned by every routed expert. Routing collapse is prevented through a bias term on the router logits adjusted dynamically during training, keeping load balanced across experts without distorting the loss signal. 

As the hidden state `h_t` evolves across loop iterations, the router may select different expert subsets at each depth, making every loop computationally distinct despite shared weights. MoE provides breadth; looping provides depth. If the activation ratio is ~5%, Mythos could hold hundreds of billions of total parameters while activating only a small fraction per token — the true parameter count, if ever disclosed, would be a storage number, not a compute number.

---

## The Memorization-Reasoning Tradeoff

Looped models exhibit an interesting dichotomy: looping improves reasoning but can hurt memorization. The recurrent structure is optimized for iterative composition — running a reasoning chain forward — but does not inherently improve the storage of rote facts.

This maps to an observable characteristic of Mythos: it reasons exceptionally well about novel problems it has never seen, but its factual recall can be inconsistent. The architecture is structurally biased toward composition over memorization.

Looping-based regularization (Saunshi et al., 2025) can be used to balance this tradeoff during training — applying stronger looping constraints for reasoning tasks while relaxing them for retrieval tasks.

---

## Parameter Reuse via LoRA Adaptation

A complementary approach from Relaxed Recursive Transformers (Bae et al., 2024): rather than requiring fully identical weights at every loop, add a small **depth-wise LoRA module** at each iteration. This preserves the compactness of weight sharing while allowing each loop to adapt its behavior slightly.

The result:
- Each loop shares a large common weight matrix (the recursive base)
- A small rank-r adaptation matrix shifts behavior per iteration depth
- The total parameter overhead is minimal

This bridges the gap between pure weight-tying (maximally parameter-efficient, less expressive) and fully distinct layers (maximally expressive, no parameter savings). Mythos likely sits somewhere on this spectrum.

---

## Continuous Depth-wise Batching

A downstream consequence of the recursive architecture: **Continuous Depth-wise Batching**. Because all tokens share the same recurrent block, the model can exit the loop at different depths for different tokens or sequences — processing easy inputs quickly and hard inputs with more iterations, all within the same batch.

Theoretical analysis suggests 2-3x improvements in inference throughput. For a deployed model like Mythos serving many users simultaneously, this would be a substantial efficiency gain.

---

## Summary: What Mythos Probably Is

| Property | Description |
|---|---|
| Architecture | Recurrent-Depth Transformer (Prelude + Looped Recurrent Block + Coda) |
| FFN layer | Suspected MoE — fine-grained experts + always-on shared experts |
| Parameter count | Very large total; small fraction activated per token (~5% estimate) |
| Reasoning mechanism | Implicit multi-hop via iterative latent updates — no token output between steps |
| Inference-time scaling | More loops = deeper reasoning, following predictable exponential decay |
| Training stability | LTI-constrained injection parameters with spectral radius < 1 |
| Loop differentiation | Likely uses loop-index positional embedding (à la RoPE) per iteration |
| Halting | Adaptive Computation Time or learned convergence criterion |
| Attention | GQA (with optional Flash Attention 2) or MLA with compressed KV latent cache |
| Scaling law | Optimal training scales looping and data together, not parameters alone |
| Reasoning vs. memory | Structurally biased toward composition; memorization requires separate treatment |
| Deployment | Continuous Depth-wise Batching enables variable compute per request |

---

## References

### Twitter / X

- Why Claude Mythos is so good — looped transformer theory (Sigrid Jin): https://x.com/realsigridjin/status/2044620031410266276
- LT implicit reasoning over parametric knowledge unlocks generalization (Yuekun Yao): https://x.com/yuekun_yao/status/2044229171627639004
- Looped transformer cyclic trajectories and input injection (rosinality): https://x.com/rosinality/status/2043953033428541853
- Parcae scaling laws for stable looped language models — thread (Hayden Prairie): https://x.com/hayden_prairie/status/2044453231913537927
- RoPE-like loop index embedding idea to differentiate functions across iterations (davidad): https://x.com/davidad/status/2044453231913537927
- On the Looped Transformers Controversy by ChrisHayduk: https://x.com/ChrisHayduk/status/2045947623572688943
- On the Looped Transformers Controversy Summary by @realsigridjin https://x.com/realsigridjin/status/2046012743778766875


### Papers

- Fine-grained expert segmentation and shared expert isolation in MoE: https://arxiv.org/abs/2401.06066
- Loop, Think, & Generalize — Implicit Reasoning in Recurrent Depth Transformers: https://arxiv.org/pdf/2604.07822
- Parcae — Scaling Laws for Stable Looped Language Models: https://arxiv.org/abs/2604.12946
- Parcae blog: https://sandyresearch.github.io/parcae/
- Universal Transformers: https://arxiv.org/pdf/1807.03819
- Reasoning with Latent Thoughts — On the Power of Looped Transformers: https://arxiv.org/abs/2502.17416
- Training Large Language Models to Reason in a Continuous Latent Space: https://arxiv.org/abs/2412.06769
- Relaxed Recursive Transformers — Effective Parameter Sharing with Layer-wise LoRA: https://arxiv.org/pdf/2410.20672
- Mixture-of-Depths Attention: https://arxiv.org/abs/2603.15619
- Hyperloop Transformers: https://arxiv.org/abs/2604.21254
- The Recurrent Transformer: Greater Effective Depth and Efficient Decoding: https://arxiv.org/abs/2604.21215

---

## Citation

If you use OpenMythos in your research or build on this work, please cite:

```bibtex
@software{gomez2026openmythos,
  author    = {Kye Gomez},
  title     = {OpenMythos: A Theoretical Reconstruction of the Claude Mythos Architecture},
  year      = {2026},
  url       = {https://github.com/kyegomez/OpenMythos},
  note      = {Recurrent-Depth Transformer with MoE, MLA, LTI-stable injection, and ACT halting}
}
```

---

## License

MIT License — Copyright (c) 2026 Kye Gomez. See [`LICENSE`](LICENSE) for the full text.
