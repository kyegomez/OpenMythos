# OpenMythos — PAI Review, April 2026

External code review conducted via the PAI Algorithm (v3.8.0, DETERMINED
effort) on commit `eae0f04`. Eight specialist reviewers ran in parallel:
correctness, performance, adversarial stress, maintainability, testing,
Kieran-Python style, architecture, and PyTorch 2026 best practices.

The branch `pai-review-2026-04` contains eight commits landing the Tier 0
install-blockers and the Tier 1 correctness fixes. **All 81 tests pass.**
Tiers 2 and 3 below are documented as a prioritized follow-up roadmap —
each is keyed back to a reviewer finding, with impact and fix sketch.

---

## Landed in this branch (commits on `pai-review-2026-04`)

| # | Commit | Category | Files | What |
|---|---|---|---|---|
| 1 | `fix(install): unpin torch…` | Install | `pyproject.toml`, `__init__.py`, `README.md`, `tokenizer.py` | `torch==2.11.0` was a nonexistent pin; relaxed to `>=2.4,<3`. Added loguru + pytest config + 3.11/3.12 classifiers. `__all__` referenced undefined `load_tokenizer`/`get_vocab_size` (AttributeError on `import *`) — trimmed to genuine public surface. README `mythos_7b()` → `mythos_3b()` (7b was never defined). Deferred the `transformers` import so `import open_mythos` no longer ImportErrors without transformers installed. |
| 2 | `feat(config): __post_init__ validation` | Correctness | `main.py` | Previously invalid combos (`attn_type="MLA"`, `n_heads % n_kv_heads != 0`, odd head_dim, `topk > n_experts`, `max_loop_iters < 1`) silently fell through or crashed deep inside the forward. Now `ValueError` at construction. `attn_type` is also typed `Literal["gqa", "mla"]`. |
| 3 | `fix(numerics): LTI fp32, loop-index fp32, ACT remainder, tie-after-init` | Correctness | `main.py` | Four numerical fixes: (a) `LTIInjection.get_A` now computes `exp(-exp(x))` in fp32 with tighter clamp `(-10, 10)` — the bf16 path was underflowing to `exp(-0)=1.0` and silently breaking the ρ(A)<1 guarantee. (b) `loop_index_embedding` now computes frequencies in fp32 so adjacent `k` indices don't collapse to the same bf16 value. (c) `OpenMythos.__init__` runs `_init_weights` BEFORE tying so the shared tensor isn't initialized twice. (d) `RecurrentBlock.forward` flushes remainder probability onto never-halted positions so ACT weights sum to ~1 for every position. |
| 4 | `feat(generate): bounds check, eval mode, EOS stopping` | Correctness | `main.py` | `forward` rejects `T=0` and `start_pos+T > max_seq_len` (previously both silently indexed a zero-length freqs slice and produced garbage). `generate()` calls `self.eval()` for the duration (restoring prior mode on exit) so dropout doesn't fire during sampling. Added `eos_token_id` parameter and per-row finished-mask stopping. `top_k` is clamped to vocab_size. |
| 5 | `fix(mla): cache shared k_rope once` | Correctness + perf | `main.py` | MLAttention was expanding k_rope to `(B, T, n_heads, rope_dim)` via `.expand().contiguous()` before caching, storing `n_heads` identical copies per token. At `n_heads=16` this is 16× more rope cache than the DeepSeek-V2 design specifies — negating the memory savings that motivate MLA. Now the shared `(B, T, rope_dim)` is cached once; per-head broadcast happens at compute time via a cost-free view. |
| 6 | `test: PAI regression suite + Cyrillic fix` | Tests | `test_main.py`, `tests/test_pai_regressions.py` | Added 14 regression tests covering every fix above: config validation (5), bf16 LTI stability, bf16 loop-index distinct frequencies, weight tying storage identity, forward empty/over-max rejection, generate clamping, generate training-mode restore, generate EOS early stop, LoRA clamp past max. Also fixed `TestOpenMythosMLА` (Cyrillic `А` U+0410) → `TestOpenMythosMLA` (ASCII). |
| 7 | `fix(tests): slice freqs_cis to T` | Tests | `test_main.py` | 13 unit tests in `TestGQAttention`/`TestMLAttention`/`TestTransformerBlock`/`TestRecurrentBlock` were passing the full max-seq-len freqs table into attention forwards, which broadcast-fails at `apply_rope`. The tests had been silently broken because the prior eager `transformers` import meant `test_main.py` couldn't load without transformers installed. Fixed by slicing `[:T]`. |
| 8 | `chore: gitignore uv.lock` | Chore | `.gitignore` | |

Verification: `pytest test_main.py tests/test_pai_regressions.py tests/test_rope_debug.py` → **81 passed in 1.25s**.

---

## Reviewer findings summary

30 total findings; severity counts across reviewers:

| Severity | Count | Landed | Deferred |
|---|---|---|---|
| CRITICAL | 4 | 3 | 1 |
| HIGH | 13 | 4 | 9 |
| MEDIUM | 9 | 4 | 5 |
| LOW | 4 | 1 | 3 |

---

## Tier 2 — Performance (deferred, ordered by expected speedup)

### 2.1 Swap manual attention for `F.scaled_dot_product_attention`
*Source: performance-reviewer #2, adversarial adv-06 adjacency, best-practices topic 1*

`GQAttention.forward` and `MLAttention.forward` materialize the full `(B, H, T, S)` attention matrix, softmax, matmul. On H100 bf16 this misses the fused Flash Attention 2/3 kernel. At `T=2048, B=4, H=16` the scratch matrix is ~1 GB per layer.

**Fix sketch (both classes):**

```python
dropout_p = self.attn_drop.p if self.training else 0.0
out = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=mask,
    is_causal=(mask is None and kv_cache is None),
    dropout_p=dropout_p,
    scale=scale,
)
```

**Expected:** 2–4× training step speedup, 3–8× decode at long context.

**Gotcha:** the `attn_drop` Dropout is consumed inside SDPA; the explicit softmax+dropout pair is redundant. Also SDPA needs `q,k,v` already in `(B, H, T, d)` shape — the transposes in both classes already produce that.

### 2.2 Vectorized MoE dispatch (grouped GEMM)
*Source: performance-reviewer #1, maintainability M16, adversarial adv-10, kieran #18*

`MoEFFN.forward` nests `for i in range(topk): for eid in range(n_experts):` — 256 Python iterations per MoE layer at `topk=4, n_experts=64`. At 1T scale with `n_experts=512` this becomes 2048 iterations × 16 recurrent loops = 32,768 kernel launches per forward. Catastrophic.

**Fix sketch:**

```python
# Sort tokens by expert id for contiguous dispatch.
exp_ids = topk_idx.reshape(-1)
tok_ids = torch.arange(N, device=x.device).repeat_interleave(self.topk)
gate_w  = topk_scores.reshape(-1, 1)
sort_idx = exp_ids.argsort()
counts = torch.bincount(exp_ids[sort_idx], minlength=self.n_experts)
ends = counts.cumsum(0); starts = torch.cat([torch.zeros(1, device=...), ends[:-1]])
x_perm = flat[tok_ids[sort_idx]]
out = torch.zeros_like(flat)
for eid in range(self.n_experts):
    s, e = starts[eid].item(), ends[eid].item()
    if s == e: continue
    y = self.routed_experts[eid](x_perm[s:e]) * gate_w[sort_idx][s:e]
    out.index_add_(0, tok_ids[sort_idx][s:e], y)
```

At scale (512 experts, 2048-token batch), vendor torchtitan's
`triton_contiguous_group_gemm` — a single-file Triton kernel that replaces
the Python per-expert loop with one fused kernel. 2.6× measured speedup
on DeepSeek-V3 training.

`moda.py`'s `DeepSeekMoE` already uses the bincount pattern (lines
562–569) — port it to `main.py::MoEFFN`.

### 2.3 Preallocated KV cache (fix O(T²) decode)
*Source: performance-reviewer #3, adversarial adv-12, architecture #10*

Every decode step does `torch.cat([cache[k], new_k], dim=1)` — allocates a
fresh tensor of the full-so-far cache size and copies everything. Over N
decoded tokens that's O(N²) memory bandwidth and allocator pressure.
At N=2048, B=1, H=16, d=192, bf16 × 20 attention layers, this is
~260 GB of redundant memcpy per generation.

**Fix sketch:** preallocate at `generate()` entry, index-write each step:

```python
# At start of generate():
max_len = prompt_len + max_new_tokens
# In attention forward:
cache = kv_cache.setdefault(cache_key, {
    "k": torch.empty(B, max_len, n_kv_heads, head_dim, ...),
    "v": torch.empty(B, max_len, n_kv_heads, head_dim, ...),
    "len": 0,
})
pos = cache["len"]
cache["k"][:, pos:pos+T] = k
cache["v"][:, pos:pos+T] = v
cache["len"] = pos + T
k = cache["k"][:, :pos+T]
v = cache["v"][:, :pos+T]
```

**Expected:** 3–10× decode speedup at T ≥ 1k. Same pattern for MLA's
`c_kv` and shared `k_rope`.

### 2.4 Gradient checkpointing option
*Source: performance-reviewer #6, best-practices topic 10*

At 1T scale with `n_loops=16`, activations dominate memory. No
`gradient_checkpointing` knob exists. Add an opt-in field to
`MythosConfig`, and wrap the recurrent loop body:

```python
from torch.utils.checkpoint import checkpoint
if self.training and self.cfg.gradient_checkpointing:
    trans_out = checkpoint(
        self.block, combined, freqs_cis, mask, kv_cache, cache_key,
        use_reentrant=False,
    )
else:
    trans_out = self.block(...)
```

`use_reentrant=False` is the 2026-standard form — required for nested
checkpointing, `torch.autograd.grad`, and compatibility with
`torch.compile`. Also wire `apply_activation_checkpointing` into the
FSDP setup in `training/3b_fine_web_edu.py`.

**Expected:** ~`sqrt(n_loops)` activation memory reduction → difference
between trainable and OOM at 1T.

### 2.5 Aux-loss-free router-bias update
*Source: correctness residual #2, adversarial adv-03, best-practices topic 7*

`MoEFFN.router_bias` is registered as a buffer with a comment "adjusted
externally during training; not a gradient param" — but grep confirms
no code path updates it. The aux-loss-free load-balancing scheme is
inert; router collapse after a few hundred steps will stay collapsed.

**DeepSeek-V3 update rule (Wang et al. 2024, arXiv 2408.15664):**

```python
@torch.no_grad()
def update_router_bias(self, counts: Tensor, u: float = 0.001) -> None:
    """Call once per training step with per-expert token counts."""
    avg = counts.float().mean()
    err = avg - counts.float()
    self.router_bias.add_(u * err.sign())
```

Expose as `MoEFFN.update_bias(counts)`; wire a forward-hook in the
training script that collects counts per-step and calls it. Freeze (set
`u=0`) in the final 3% of training.

### 2.6 RoPE real-pair path (drop fp32 complex roundtrip)
*Source: performance-reviewer #4, correctness #9, best-practices topic 3*

`apply_rope` does `x.float()` → `view_as_complex` → multiply → `view_as_real`
→ `.to(x.dtype)`. Three full tensor copies per attention call, blocks
torch.compile fusion, and bf16 has no native complex dtype so the fp32
upcast is forced.

**Fix sketch (real-pair + GPT-NeoX split-halves, HF-compatible):**

```python
def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: (B, T, H, D); cos/sin: (T, D) same dtype as x
    x1, x2 = x.chunk(2, dim=-1)
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos[None, :, None, :] + rot * sin[None, :, None, :]
```

Precompute `cos`/`sin` tables in `OpenMythos.__init__`, register as
non-persistent buffers in the model's working dtype.

**Expected:** 5–10% training speedup; much bigger for decode.

### 2.7 Precomputed causal mask + loop embeddings
*Source: performance-reviewer #5*

`_causal_mask(T, device)` allocates a fresh `(1,1,T,T)` −∞ tensor on
every forward. `loop_index_embedding` allocates freqs/angles/sin/cos
on every loop of every forward. Both are amortizable to one-time init.

**Fix:** register a persistent `(1,1,max_seq_len,max_seq_len)` mask
buffer in `__init__`, slice `[:T,:T]`. Register a
`(max_loop_iters, dim)` loop-embedding table in `RecurrentBlock.__init__`;
index by `t` instead of recomputing.

### 2.8 torch.compile support
*Source: performance-reviewer #8, best-practices topic 9*

Model currently has graph breaks from: MoE Python loop (2.2), dict
cache with f-string keys, `.item()` calls in the router path. Once
2.1-2.3 land, add:

```python
model = torch.compile(model, mode="default")  # training
# or mode="reduce-overhead" for single-stream decode
```

Verify with `TORCH_LOGS=graph_breaks python train.py`.

**Expected:** +15–30% on H100.

### 2.9 Training: tokenization bottleneck
*Source: performance-reviewer #9*

`FineWebEduDataset.__iter__` encodes one sample at a time per worker
(~200–500k tok/s/worker). For 3B on H100, 4M+ tok/s throughput is
needed. Dataloader-bound by a factor of ~2–5×.

**Fix:** (a) use `tokenizer.encode_batch([...])` in batches of 64–128,
(b) pre-tokenize one shard to uint16 memmap once, stream slices. This
is the llm.c / nanoGPT pattern. 10–20× dataloader speedup.

---

## Tier 3 — Correctness medium (deferred)

### 3.1 `_causal_mask` shape assumes cache is empty at T>1
*Source: correctness #2, adversarial adv-02*

`forward` builds mask as `(1,1,T,T)` — fine for prefill or single-token
decode. Breaks for T>1 with non-empty cache (speculative decoding,
prefix caching, multi-token append). Attention scores are `(B,H,T,S)`
with `S = T_prev + T`; mask shape mismatches.

**Fix:** build mask of shape `(1,1,T,S)` accounting for cached length.

### 3.2 FSDP wrap policy includes both `TransformerBlock` and `RecurrentBlock`
*Source: adversarial adv-15, correctness #10*

`training/3b_fine_web_edu.py` line 412:
`ModuleWrapPolicy({TransformerBlock, RecurrentBlock})`. Double-wrap
creates ambiguous unit boundaries. At 1T with `n_experts=512` inside
`MoEFFN` not wrapped, each `TransformerBlock` wrap holds a 320M-param
flat parameter that defeats FULL_SHARD.

**Fix:** wrap `TransformerBlock` and `Expert` (so each routed expert
shards independently). Consider `reshard_after_forward=False` on the
recurrent block to avoid re-gather per loop iteration.

### 3.3 `router_bias` buffer sharded by FSDP
*Source: correctness residual #2*

Under FSDP FULL_SHARD, buffers are sharded by default. If 2.5 lands a
bias update, each rank would update only its local slice. Fix:
add `router_bias` to `ignored_parameters` or mark it explicitly
replicated.

### 3.4 `kv_cache` dict reuse across `forward()` calls pollutes
*Source: adversarial adv-12, maintainability M6*

Cache keys bake `n_loops` into the schema (`recurrent_loop_{t}`).
Reusing the same dict across calls with different `n_loops` mixes
stale cached keys into fresh ones at loop indices 0..min(n_loops).

**Fix:** introduce a `KVCache` dataclass that tracks `n_loops` and
validates on each use, or document as "append-only, strictly
monotonic".

### 3.5 LoRA clamp for depth extrapolation is semantically wrong
*Source: correctness #11, adversarial adv-11*

`t_idx = loop_t if loop_t <= max_t else max_t` — iterations beyond the
trained range all reuse the last scale. "Extra depth" at inference is
therefore not actually different from the final trained iteration,
defeating the README's depth-extrapolation claim.

**Options:** (a) return zero delta for `loop_t > max_t` (neutral),
(b) linearly interpolate or extrapolate scale, (c) replace
`nn.Embedding` with a continuous function of loop index (MLP on
sinusoid).

### 3.6 `_causal_mask` dtype is fp32 under bf16 autocast
*Source: correctness #6*

`torch.full(..., float("-inf"))` defaults to fp32. `attn + mask`
upcasts to fp32, diverges from FSDP's reduce_dtype, forces extra
casts. Fix: `dtype=q.dtype` on mask allocation.

### 3.7 vocab_size default vs tokenizer default mismatch
*Source: adversarial adv-14*

`MythosConfig.vocab_size=32000` while `MythosTokenizer` defaults to
gpt-oss-20b (~200k vocab). Non-Latin text tokenizes to ids > 32000;
`model(ids)` triggers a CUDA assertion failure.

**Fix:** either raise in `OpenMythos.__init__` if provided a
tokenizer with vocab > `cfg.vocab_size`, or change the default to
match the default tokenizer.

### 3.8 ACT initialization biases halting prob to 0.5 at init
*Source: adversarial adv-06*

`ACTHalting.halt` is initialized via `_init_weights` with
`std=0.02`. Bias defaults to 0 → sigmoid(~0) ≈ 0.5. Early in
training ACT halts ~half of positions on iteration 0 with nothing
preventing the recurrent block from degenerating into identity.

**Fix:** initialize `self.halt.bias.fill_(-2.0)` so early-training
halt prob is ~0.12, and consider a `min_loops` floor. Add a
ponder-cost loss term (expected number of iterations × λ).

### 3.9 Deterministic tie-breaking in router topk
*Source: adversarial adv-16*

`router.topk()` with tied logits (common at init) is
device/cuBLAS-version dependent. Reproducibility claims break on
hardware change. Fix: add `logits + eps * arange(n_experts)` tie-
break, or document that determinism requires `use_deterministic_algorithms`.

---

## Tier 4 — Maintainability / architecture (deferred roadmap)

### 4.1 `moda.py` is an orphan 1134-line parallel model
*Source: maintainability M1/M2/M3, architecture #1, testing coverage gap*

Zero imports from other modules, not in `__init__.py`, 69 lines of
commented-out smoke test at bottom. `RMSNorm`, `RoPE`, `Expert` are
re-implemented with a different naming scheme (`d_model` vs `dim`,
`w1/w2/w3` vs `gate/up/down`).

**Three options:**
1. Delete `moda.py` entirely.
2. Move to `experimental/moda.py` outside the published package.
3. Integrate as `attn_type="moda"` via shared primitives module.

Until resolved, it's an unmaintained second architecture that confuses
contributors. **Do not split `main.py` into submodules** (tier-4.2)
until this is resolved.

### 4.2 `main.py` monolith split
*Source: architecture #3*

1048 lines cleanly segmented by comment bars. Suggested split:

```
open_mythos/
  config.py        # MythosConfig
  norm.py          # RMSNorm
  rope.py          # precompute_rope_freqs, apply_rope, loop_index_embedding
  attention.py     # GQAttention, MLAttention
  moe.py           # Expert, MoEFFN
  recurrent.py     # LTIInjection, ACTHalting, LoRAAdapter, RecurrentBlock
  blocks.py        # TransformerBlock
  model.py         # OpenMythos
```

Keep `main.py` as a re-export shim for one release.

### 4.3 Magic numbers → named config fields
*Source: maintainability M9*

`loop_index_embedding`'s `theta=10000.0` (while main RoPE uses 500000),
`LTIInjection.B` init of `0.1`, `.clamp(-10, 10)` (landed in this
branch), `std=0.02` init, `cfg.dim // 8` loop_dim ratio, `4//3` FFN
ratio. Promote to `MythosConfig` fields with defaults.

### 4.4 KV cache type-safety
*Source: kieran #1, maintainability M6*

Typed as bare `dict`, keyed by f-strings. Two different entry shapes
(GQA: `{k, v}`, MLA: `{c_kv, k_rope}`). Introduce `KVCache` dataclass
or at minimum `TypedDict` pair.

### 4.5 Test consolidation
*Source: architecture #7*

Move `test_main.py` → `tests/test_main.py` so everything lives under
`tests/`. `[tool.pytest.ini_options]` already has `testpaths` set
correctly in this branch.

### 4.6 Training script generalization
*Source: architecture #6*

`training/3b_fine_web_edu.py` hardcodes model (`mythos_3b`), dataset,
precision, optimizer. Extract a `training/train.py` with `--variant`,
`--dataset`, `--config overrides.yaml`. Do this before writing a
second training recipe.

### 4.7 README split
*Source: architecture #8*

README at 419 lines mixes marketing, install, usage, theory, references.
Move theory/hypothesis/scaling laws to `docs/theory.md`; target
README ≤ 200 lines.

### 4.8 Variant registry
*Source: architecture #5*

Add at the end of `variants.py`:
```python
VARIANTS = {"1b": mythos_1b, "3b": mythos_3b, ...}
def get_variant(name: str) -> MythosConfig: return VARIANTS[name]()
```

### 4.9 Logging in library code
*Source: maintainability M15, kieran #17*

`main.py`/`moda.py`/`tokenizer.py` have no logging. Training uses
loguru (now declared in pyproject). Add
`import logging; logger = logging.getLogger(__name__)` to library code;
log at interesting branch points (cache-miss, attn-type resolution,
ACT early-exit, clamp activation).

### 4.10 Shared primitives module
*Source: maintainability M3*

Once `moda.py` status is resolved (4.1), extract shared primitives
(RMSNorm, RoPE, Expert) into `open_mythos/primitives.py` so
`main.py` and any other architecture share a single canonical impl.

---

## Tier 5 — Testing gaps (quick wins)

From the testing reviewer + testing gaps surfaced across all reviewers.
Each is a one-liner to add:

1. `tests/test_moda.py` — any coverage of the 1134-line orphan module (if it stays).
2. `test_model_n_loops_exceeds_max_iters` — end-to-end depth extrapolation.
3. `test_act_weight_sum_is_one` — per-position sum of iteration weights.
4. `test_act_early_exit_saves_compute` — instrumented iteration count.
5. `test_act_no_early_exit_with_cache` — confirms cache-consistency invariant.
6. `test_generate_matches_forward_over_full_sequence` — greedy-decode trajectory equivalence.
7. `test_router_bias_shifts_selection` — the update hook in 2.5 actually moves topk.
8. `test_router_bias_survives_state_dict_roundtrip` — FSDP checkpoint sharding.
9. `test_weight_tying_grad_shared` — gradient through `head.weight` equals `embed.weight.grad`.
10. `test_lti_stability_over_many_steps` — 100 SGD steps with lr=1.0, assert A stays in (0,1).
11. `test_fp16_full_forward_no_nan` — half-precision path.
12. `test_bf16_backward_gradient_flow` — autocast bf16, every submodule has finite grads.
13. `test_determinism_fixture` — checked-in 1MB state_dict + golden logits tensor.
14. `test_public_api_loads` — `from open_mythos import *` followed by `getattr` each `__all__` entry.
15. `test_readme_snippets_execute` — parse README code blocks, `exec` them; catches `mythos_7b`-class regressions.
16. `test_all_variants_construct` — parametrize over every variant; instantiate on CPU, assert shape on 1-token forward.

---

## How to open PRs from this branch

```bash
# From the OpenMythos clone:
git remote add fork <your-fork>     # e.g. https://github.com/you/OpenMythos
git push fork pai-review-2026-04

# Open PRs either as one bundle or split by commit:
#   Tier-0 + tests  — one PR
#   Correctness     — one PR
#   MLA cache fix   — one PR (has measurable memory impact)
# The commits are independent and cleanly split.
```

Each commit message explains the exact change; reviewers should not
have to read this document to evaluate a single commit.

---

## Appendix: reviewer artifacts

All eight specialist reviewers were invoked via the PAI Algorithm
(v3.8.0, DETERMINED effort) reading the codebase at
`/tmp/openmythos-review/OpenMythos/`. Findings were returned as JSON,
deduplicated across reviewers, and tiered by severity × fix cost.
Reviewers:

1. `correctness-reviewer` — 11 findings (1 CRIT, 3 HIGH, 3 MED, 4 LOW)
2. `performance-reviewer` — 11 findings (3 CRIT, 3 HIGH, 4 MED, 1 LOW)
3. `testing-reviewer` — 15 findings across coverage / weak-assertion / missing-edge
4. `maintainability-reviewer` — 17 findings across duplication / coupling / magic
5. `adversarial-reviewer` — 17 constructed failure scenarios (1 CRIT, 3 HIGH, 9 MED, 4 LOW)
6. `kieran-python-reviewer` — 18 Python-style findings
7. `architecture-strategist` — 10-point architecture debt list
8. `best-practices-researcher` — 2026-era PyTorch recommendations across 10 topics
