# Stochastic Depth Training (Option B) — Design Spec

**Date:** 2026-04-27
**Status:** Design, not yet implemented
**Related:** issue #5 (ACT depth-binding), `docs/logbook/2026-04-24-eval-and-analysis.md`

---

## Goal

Add a second training recipe — **stochastic depth without ACT weighting** — to the OpenMythos training pipeline, selectable per training run, while keeping the existing ACT recipe fully intact and checkpoint-compatible.

## Motivation

The 1B-token PoC evaluation (2026-04-24) confirmed the upstream finding: with ACT enabled, the model binds tightly to its trained recurrent depth (n_loops=16) and gains nothing from additional inference-time iterations. Depth extrapolation — a core advertised property of recurrent-depth transformers — is unreachable while ACT is on.

Upstream empirical work ([kyegomez/OpenMythos#28](https://github.com/kyegomez/OpenMythos/issues/28), 13-run ablation) showed that the only recipe producing a monotonically decreasing PPL-vs-depth curve was:

- Disable ACT (return the final hidden state directly, no weighted sum)
- Train with random `n_loops` sampled per step

We want this recipe available as an alternative training strategy without abandoning the ACT path. The two should be freely switchable, including mid-training from the same checkpoint, so the model can be trained under different recipes in different phases.

## Design

### Runtime control

Two hyperparameters added directly to `training/1b_poc_fineweb.py` (local variables, not env vars — avoids env-var sprawl):

```python
recurrent_mode = "stochastic_depth"   # "act" or "stochastic_depth"
stochastic_depth_min = 1
stochastic_depth_max = 32
```

The default is `"stochastic_depth"`. To use the current ACT recipe, change to `"act"`.

### Per-step forward

In the training loop, before each forward pass:

- If `recurrent_mode == "stochastic_depth"`: sample `n_loops` uniformly from `[stochastic_depth_min, stochastic_depth_max]` inclusive, and call the model with `bypass_act=True`.
- If `recurrent_mode == "act"`: pass `n_loops=None` (uses `cfg.max_loop_iters`) and `bypass_act=False`.

**Logging:**
- At training startup (master rank only), print a clearly visible banner stating the active `recurrent_mode` and, if stochastic, the `[min, max]` sampling range. Example: `Recurrent mode: stochastic_depth (n_loops sampled from [1, 32])`.
- Add `recurrent_mode`, `stochastic_depth_min`, `stochastic_depth_max` to the ClearML `training_hparams` dict so they appear in the ClearML task configuration.
- Log per-step `n_loops` as a ClearML scalar so the sampling distribution is visible on the dashboard.
- Include `mode=<recurrent_mode>` and `n_loops=<value>` in the per-step stderr step line so they are visible in the job logs.

### Model changes

Two surgical additions to `open_mythos/main.py`:

1. **`OpenMythos.forward()`** — new parameter `bypass_act: bool = False`, plumbed through to `self.recurrent(...)`.
2. **`RecurrentBlock.forward()`** — new parameter `bypass_act: bool = False`:
   - When `False` (default): current behavior unchanged.
   - When `True`: skip ACT weighting accumulation, skip the `halted.all()` FSDP all-reduce, return the final `h` directly after the last iteration.

The `ACTHalting` module stays present in the architecture regardless of mode. When bypassed, its weights simply receive no gradient that step.

### Checkpoint compatibility

The parameter set (state_dict keys and shapes) is **identical across modes**. A checkpoint saved in one mode loads cleanly in the other. This enables:

- Starting from an ACT-trained checkpoint and switching to stochastic depth (current use case — resume from `step_0032000.pt`)
- Curriculum-style training: phases of ACT and phases of stochastic depth interleaved
- Direct A/B comparison on the same initialization

### Stability

Existing architectural guarantees make this design stable:

- **LTI injection** with guaranteed spectral radius < 1 (ZOH discretization) makes the recurrence contractive — hidden state cannot explode across iterations.
- **Input re-injection** at every iteration prevents drift from the input signal.
- **RMSNorm** before every transformer block caps input magnitudes.

Upstream ablation confirmed monotonic PPL across depths 1→16 under this recipe.

Caveats: at `n_loops=32`, gradients through 32 shared blocks may partially vanish in the earliest iterations — not catastrophic, but worth monitoring. When switching modes mid-training, expect a transient loss spike (~0.3–0.5, ~few hundred steps) while the Coda re-adapts to the different hidden-state distribution.

**LoRA depth indexing**: `LoRAAdapter` is initialized with `cfg.max_loop_iters=16` scale embeddings. For `loop_t >= 16`, the adapter already clamps the index (line 641–642) and reuses the depth-15 scale. This means depths 16–31 will share a single LoRA scale rather than having distinct learned scales. Acceptable trade-off: keeps checkpoint compatibility (no shape change in state_dict) and the LoRA delta is a small additive modulation anyway. If per-depth LoRA at extrapolation depths becomes important later, we can bump `cfg.max_loop_iters=32` and pad/re-initialize the LoRA scale embedding in a separate migration.

### Evaluation

No changes needed. `evaluations/eval_checkpoint.py` already runs a depth sweep at `n_loops ∈ {1, 2, 4, 8, 12, 16, 24, 32}`, which gives a direct apples-to-apples comparison between Option A and Option B checkpoints.

## Scope (YAGNI)

**In scope:**
- Runtime mode toggle in the training script
- `bypass_act` flag plumbed through `OpenMythos.forward()` and `RecurrentBlock.forward()`
- Uniform random `n_loops` sampling in the training loop
- ClearML logging of `recurrent_mode` and per-step `n_loops`

**Out of scope (explicitly not doing):**
- Biased / non-uniform depth sampling distributions
- Automatic scheduling between modes (manual switch only)
- Removing or refactoring the ACT path
- Changing `MythosConfig` (no new fields; all control is at training-script level)
- Soft attention over loop outputs (Option C) — separate future design if needed

## Testing

- Unit test: `RecurrentBlock.forward(bypass_act=True)` returns `h` at the requested `n_loops`, with no ACT accumulation applied. Parameter grads match expectation (ACT module receives zero grad).
- Unit test: `bypass_act=False` path produces identical output to the current implementation (regression).
- Unit test: Checkpoint round-trip — save in one mode, load in the other, verify no state_dict mismatch.
- Smoke test: Small-config training loop runs one step in each mode without error.

## Success criteria

1. A single training run can be launched in either `"act"` or `"stochastic_depth"` mode by changing one variable.
2. The current ACT recipe is bit-identical to before when `recurrent_mode="act"`.
3. A checkpoint trained in one mode can be resumed in the other (state_dict loads cleanly; training continues; loss spike is transient).
4. After training ~1B tokens in stochastic_depth mode from a checkpoint, the depth sweep shows non-trivial generation at `n_loops > 16` (i.e., the depth-binding is reduced).

## Open questions

None currently. Range `[1, 32]` chosen based on upstream recipe and compute budget; can be tuned later via the script-level variables without code changes.
