# Stochastic Depth Training (Option B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a runtime-selectable stochastic-depth training recipe (no ACT weighting + random per-step `n_loops`) while keeping the existing ACT recipe fully intact and checkpoint-compatible.

**Architecture:** Thread a boolean `bypass_act` flag through `OpenMythos.forward() -> RecurrentBlock.forward()`. When `True`, skip the ACT weighted-sum accumulation and halting-driven early exit, returning the final hidden state directly. The training script samples `n_loops` uniformly per step when in stochastic-depth mode. `ACTHalting` and `LoRAAdapter` modules remain present in the model unchanged, so checkpoints are bit-compatible across modes.

**Tech Stack:** PyTorch (FSDP, distributed), pytest, loguru logger, ClearML.

**Spec:** `docs/superpowers/specs/2026-04-27-stochastic-depth-training-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `open_mythos/main.py` | Modify | Add `bypass_act` parameter to `RecurrentBlock.forward()` and `OpenMythos.forward()` |
| `training/1b_poc_fineweb.py` | Modify | Add `recurrent_mode` / `stochastic_depth_min` / `stochastic_depth_max` variables; sample `n_loops` per step; log mode and per-step `n_loops` |
| `tests/test_stochastic_depth.py` | Create | New test module for `bypass_act` behavior, regression of ACT path, checkpoint cross-mode compatibility, smoke test of training step |

---

## Task 1: RecurrentBlock `bypass_act` — test first

**Files:**
- Create: `tests/test_stochastic_depth.py`
- Modify: `open_mythos/main.py` (RecurrentBlock.forward signature and body)

- [ ] **Step 1: Write the failing tests**

Create the file `tests/test_stochastic_depth.py` with the following content:

```python
"""Tests for stochastic-depth (Option B) training path: bypass_act flag."""

import pytest
import torch

from open_mythos.main import MythosConfig, OpenMythos, RecurrentBlock


def _small_cfg() -> MythosConfig:
    """Small CPU config used by the existing test suite."""
    return MythosConfig(
        vocab_size=128,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=2,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=64,
        act_threshold=0.99,
        lora_rank=4,
    )


def _build_block_inputs(cfg: MythosConfig, B: int = 2, T: int = 8):
    """Build the (h, e, freqs_cis) inputs needed by RecurrentBlock.forward."""
    torch.manual_seed(0)
    model = OpenMythos(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    x = model.embed(input_ids)
    freqs_cis = model.freqs_cis[:T]
    mask = model._causal_mask(T, x.device, x.dtype)
    for i, layer in enumerate(model.prelude):
        x = layer(x, freqs_cis, mask, None, cache_key=f"prelude_{i}")
    return model.recurrent, x.clone(), x.clone(), freqs_cis, mask


def test_recurrent_block_bypass_act_differs_from_act():
    """bypass_act=True should produce a different output than bypass_act=False."""
    cfg = _small_cfg()
    block, h, e, freqs_cis, mask = _build_block_inputs(cfg)
    torch.manual_seed(1)
    out_act = block(h.clone(), e, freqs_cis, mask, n_loops=4, bypass_act=False)
    torch.manual_seed(1)
    out_bypass = block(h.clone(), e, freqs_cis, mask, n_loops=4, bypass_act=True)
    assert out_act.shape == out_bypass.shape
    assert not torch.allclose(out_act, out_bypass, atol=1e-6), (
        "bypass_act=True should not equal ACT-weighted output"
    )


def test_recurrent_block_bypass_act_runs_full_n_loops():
    """With bypass_act=True there should be no early exit; all n_loops iterations run."""
    cfg = _small_cfg()
    block, h, e, freqs_cis, mask = _build_block_inputs(cfg)
    call_count = {"n": 0}
    original_block = block.block.forward

    def counting_forward(*args, **kwargs):
        call_count["n"] += 1
        return original_block(*args, **kwargs)

    block.block.forward = counting_forward
    try:
        _ = block(h, e, freqs_cis, mask, n_loops=3, bypass_act=True)
    finally:
        block.block.forward = original_block
    assert call_count["n"] == 3, f"expected 3 block calls, got {call_count['n']}"


def test_recurrent_block_bypass_act_returns_final_h():
    """bypass_act=True output should match a manual iteration returning the final h."""
    from open_mythos.main import loop_index_embedding

    cfg = _small_cfg()
    block, h, e, freqs_cis, mask = _build_block_inputs(cfg)
    n_loops = 3

    torch.manual_seed(1)
    h_manual = h.clone()
    for t in range(n_loops):
        h_loop = loop_index_embedding(h_manual, t, block.loop_dim)
        combined = block.norm(h_loop + e)
        trans_out = block.block(combined, freqs_cis, mask, None, f"recurrent_loop_{t}")
        trans_out = trans_out + block.lora(trans_out, t)
        h_manual = block.injection(h_manual, e, trans_out)

    torch.manual_seed(1)
    out_bypass = block(h.clone(), e, freqs_cis, mask, n_loops=n_loops, bypass_act=True)

    assert torch.allclose(out_bypass, h_manual, atol=1e-5), (
        "bypass_act=True should return the final hidden state after n_loops iterations"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stochastic_depth.py -v`
Expected: all three tests FAIL with `TypeError: forward() got an unexpected keyword argument 'bypass_act'`.

- [ ] **Step 3: Add `bypass_act` parameter to `RecurrentBlock.forward()`**

In `open_mythos/main.py`, modify `RecurrentBlock.forward()` (currently around lines 853–941). Replace the current `forward` method with this version:

```python
    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        bypass_act: bool = False,
    ) -> torch.Tensor:
        """
        Run the recurrent loop for up to n_loops iterations.

        Args:
            h          -- initial hidden state from the Prelude, shape (B, T, dim)
            e          -- encoded input frozen for injection each step, shape (B, T, dim)
            freqs_cis  -- precomputed RoPE frequencies
            mask       -- additive causal mask or None
            n_loops    -- number of loop iterations; defaults to cfg.max_loop_iters.
            kv_cache   -- cache dict passed through to the inner TransformerBlock;
                          each loop iteration uses a separate cache key
            bypass_act -- if True, skip ACT weighting and return the final h directly
                          after running all n_loops iterations (used for Option B
                          stochastic-depth training).

        Returns:
            ACT-weighted sum of hidden states across iterations when bypass_act=False,
            or the final hidden state after n_loops iterations when bypass_act=True.
            Shape: (B, T, dim) in both cases.
        """
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape

        if not bypass_act:
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

            if bypass_act:
                continue

            p = self.act(h)  # (B, T)
            still_running = ~halted

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

            if kv_cache is None:
                all_halted = halted.all()
                if torch.distributed.is_initialized():
                    flag = torch.tensor(
                        [all_halted], dtype=torch.int32, device=h.device
                    )
                    torch.distributed.all_reduce(
                        flag, op=torch.distributed.ReduceOp.MIN
                    )
                    all_halted = flag.item() > 0
                if all_halted:
                    break

        if bypass_act:
            return h

        not_halted = ~halted
        if not_halted.any():
            final_remainder = (1.0 - cumulative_p).clamp(min=0) * not_halted.float()
            h_out = h_out + final_remainder.unsqueeze(-1) * h
        return h_out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stochastic_depth.py -v`
Expected: all three `test_recurrent_block_bypass_act_*` tests PASS.

- [ ] **Step 5: Verify the ACT-mode regression — full existing suite**

Run: `pytest tests/test_main.py -v`
Expected: same pass/fail counts as before this task (no newly broken tests; the 14 pre-existing failures remain). The goal is proving `bypass_act=False` (default) did not break the existing ACT behavior.

- [ ] **Step 6: Commit**

```bash
git add open_mythos/main.py tests/test_stochastic_depth.py
git commit -m "feat(model): add bypass_act flag to RecurrentBlock.forward

Skips ACT weighting and returns the final hidden state directly.
Default bypass_act=False preserves the existing ACT code path.
"
```

---

## Task 2: Plumb `bypass_act` through `OpenMythos.forward()`

**Files:**
- Modify: `open_mythos/main.py` (OpenMythos.forward signature and body)
- Modify: `tests/test_stochastic_depth.py` (add one test)

- [ ] **Step 1: Write the failing test**

Append this test to `tests/test_stochastic_depth.py`:

```python
def test_openmythos_forward_bypass_act_propagates():
    """OpenMythos.forward(bypass_act=True) should route through RecurrentBlock with bypass_act=True."""
    cfg = _small_cfg()
    torch.manual_seed(0)
    model = OpenMythos(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))

    torch.manual_seed(1)
    logits_act = model(input_ids, n_loops=3, bypass_act=False)
    torch.manual_seed(1)
    logits_bypass = model(input_ids, n_loops=3, bypass_act=True)

    assert logits_act.shape == logits_bypass.shape
    assert not torch.allclose(logits_act, logits_bypass, atol=1e-6), (
        "bypass_act should change model output"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stochastic_depth.py::test_openmythos_forward_bypass_act_propagates -v`
Expected: FAIL with `TypeError: forward() got an unexpected keyword argument 'bypass_act'`.

- [ ] **Step 3: Add `bypass_act` to `OpenMythos.forward()`**

In `open_mythos/main.py`, locate `OpenMythos.forward()` (currently around lines 1043–1086). Make two edits.

First, update the signature and docstring (around lines 1044–1072). Replace the method definition header with:

```python
    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        start_pos: int = 0,
        bypass_act: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through Prelude → Recurrent Block → Coda.

        Args:
            input_ids  -- token indices of shape (B, T)
            n_loops    -- recurrent loop depth; defaults to cfg.max_loop_iters.
                          Increase at inference to extrapolate to harder problems.
            kv_cache   -- dict mutated in-place for autoregressive KV caching;
                          pass an empty dict {} and reuse across decode steps
            start_pos  -- index of the first token in input_ids within the full
                          sequence; used to select the correct RoPE frequencies
                          during incremental decoding (0 for prefill, prompt_len
                          for each subsequent decode step)
            bypass_act -- if True, RecurrentBlock skips ACT weighting and returns
                          the final hidden state directly. Default False preserves
                          the existing ACT behavior.

        Returns:
            Logits of shape (B, T, vocab_size)
        """
```

Second, update the call to `self.recurrent(...)` — find the line that currently reads:

```python
        x = self.recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)
```

Replace it with:

```python
        x = self.recurrent(x, e, freqs_cis, mask, n_loops, kv_cache, bypass_act)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stochastic_depth.py::test_openmythos_forward_bypass_act_propagates -v`
Expected: PASS.

- [ ] **Step 5: Run the full new test file**

Run: `pytest tests/test_stochastic_depth.py -v`
Expected: all four tests PASS.

- [ ] **Step 6: Commit**

```bash
git add open_mythos/main.py tests/test_stochastic_depth.py
git commit -m "feat(model): plumb bypass_act through OpenMythos.forward"
```

---

## Task 3: Checkpoint round-trip test across modes

**Files:**
- Modify: `tests/test_stochastic_depth.py` (add one test)

- [ ] **Step 1: Write the cross-mode checkpoint test**

Append this test to `tests/test_stochastic_depth.py`:

```python
def test_state_dict_compatible_across_modes(tmp_path):
    """A checkpoint saved before toggling bypass_act should load without key mismatch."""
    cfg = _small_cfg()
    torch.manual_seed(0)
    model_a = OpenMythos(cfg)
    ckpt_path = tmp_path / "model.pt"
    torch.save(model_a.state_dict(), ckpt_path)

    torch.manual_seed(1)
    model_b = OpenMythos(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model_b.load_state_dict(state, strict=True)
    assert not missing, f"unexpected missing keys: {missing}"
    assert not unexpected, f"unexpected extra keys: {unexpected}"

    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    torch.manual_seed(2)
    logits_act = model_b(input_ids, n_loops=3, bypass_act=False)
    torch.manual_seed(2)
    logits_bypass = model_b(input_ids, n_loops=3, bypass_act=True)
    assert logits_act.shape == logits_bypass.shape
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_stochastic_depth.py::test_state_dict_compatible_across_modes -v`
Expected: PASS (no model code changes needed — the parameter set is already mode-independent).

- [ ] **Step 3: Commit**

```bash
git add tests/test_stochastic_depth.py
git commit -m "test: verify state_dict is compatible across ACT / stochastic_depth modes"
```

---

## Task 4: Training script — runtime mode toggle + per-step sampling + logging

**Files:**
- Modify: `training/1b_poc_fineweb.py`

- [ ] **Step 1: Add the `random` import**

In `training/1b_poc_fineweb.py`, locate the import block near the top of the file (around line 29–46). Add `import random` alphabetically among the stdlib imports. For example, after `import os` (or wherever it fits alphabetically):

```python
import random
```

- [ ] **Step 2: Add the three hyperparameters to the hyperparams block**

In `training/1b_poc_fineweb.py`, locate the hyperparameter block that starts around line 398:

```python
    # ------------------------------------------------------------------
    # Hyperparameters (env-var configurable with defaults)
    # ------------------------------------------------------------------
    seq_len = 2048
    micro_batch = 1
```

Immediately before `seq_len = 2048`, insert the three new variables:

```python
    # Recurrent-depth training recipe (Option A: ACT, Option B: stochastic depth).
    # Change recurrent_mode to "act" to use the original ACT halting recipe.
    recurrent_mode = "stochastic_depth"  # "act" or "stochastic_depth"
    stochastic_depth_min = 1
    stochastic_depth_max = 32

```

- [ ] **Step 3: Add startup banner and ClearML hparams**

Locate `training_hparams = {...}` (around line 423). Add the three new keys at the end of the dict (just before the closing `}`):

```python
        "recurrent_mode": recurrent_mode,
        "stochastic_depth_min": stochastic_depth_min,
        "stochastic_depth_max": stochastic_depth_max,
```

Then find the `if master:` block that logs hyperparameters (search for the earliest `logger.info` with "Parameters:" or the config banner near line 484). Immediately after the existing banner lines, add a dedicated mode line. For example, right after:

```python
    logger.info(f"Parameters: {param_count:,}  |  AMP dtype: {amp_dtype}")
```

(The exact wording may differ — find the existing "Parameters:" log line and insert the next line directly after it, inside the same `if master:` guard if present.)

Add:

```python
    if master:
        if recurrent_mode == "stochastic_depth":
            logger.info(
                f"Recurrent mode: stochastic_depth "
                f"(n_loops sampled uniformly from [{stochastic_depth_min}, {stochastic_depth_max}])"
            )
        else:
            logger.info(f"Recurrent mode: act (n_loops = cfg.max_loop_iters = {cfg.max_loop_iters})")
```

- [ ] **Step 4: Sample `n_loops` per step and pass both flags to the forward**

Locate the training loop forward call (around line 555–556):

```python
            with sync, amp_ctx:
                logits = model(x)
```

Replace with:

```python
            if recurrent_mode == "stochastic_depth":
                n_loops_this_step = random.randint(stochastic_depth_min, stochastic_depth_max)
                bypass_act_this_step = True
            else:
                n_loops_this_step = None
                bypass_act_this_step = False

            with sync, amp_ctx:
                logits = model(
                    x,
                    n_loops=n_loops_this_step,
                    bypass_act=bypass_act_this_step,
                )
```

- [ ] **Step 5: Include mode and n_loops in the per-step stderr log and ClearML scalars**

Locate the per-step logging block (around line 572–588). Modify the `logger.info(...)` call to include `mode=` and `n_loops=`.

Replace:

```python
            logger.info(
                f"step {step:6d}/{total_steps} | loss {loss_accum:.4f} "
                f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e} "
                f"| {tok_per_sec / 1e6:.2f}M tok/s "
                f"| {tokens_seen / 1e9:.1f}B tokens seen"
            )
```

with:

```python
            n_loops_display = (
                n_loops_this_step
                if n_loops_this_step is not None
                else cfg.max_loop_iters
            )
            logger.info(
                f"step {step:6d}/{total_steps} | loss {loss_accum:.4f} "
                f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e} "
                f"| {tok_per_sec / 1e6:.2f}M tok/s "
                f"| {tokens_seen / 1e9:.1f}B tokens seen "
                f"| mode={recurrent_mode} n_loops={n_loops_display}"
            )
```

Then, in the block of `log_clearml(...)` calls directly below, add one more scalar:

```python
            log_clearml("n_loops", float(n_loops_display), step)
```

- [ ] **Step 6: Run the full test suite to verify no regression in the training script**

The training script is not directly unit-tested, but a syntax/import error would be caught by import. Run:

```bash
python -c "import ast; ast.parse(open('training/1b_poc_fineweb.py').read()); print('OK')"
```

Expected: `OK`.

- [ ] **Step 7: Commit**

```bash
git add training/1b_poc_fineweb.py
git commit -m "feat(training): add stochastic-depth mode to training script

New local variables (recurrent_mode, stochastic_depth_min, stochastic_depth_max)
control the recipe. Default recurrent_mode='stochastic_depth' samples n_loops
uniformly from [1, 32] and uses bypass_act=True. Set recurrent_mode='act'
for the original ACT halting recipe.

Logs mode and per-step n_loops to stderr and ClearML.
"
```

---

## Task 5: Smoke-test training step in each mode

**Files:**
- Modify: `tests/test_stochastic_depth.py` (add one test)

- [ ] **Step 1: Write the smoke test**

Append to `tests/test_stochastic_depth.py`:

```python
def test_training_step_runs_in_each_mode():
    """One forward+backward+optimizer step works in both modes without error."""
    cfg = _small_cfg()
    torch.manual_seed(0)
    model = OpenMythos(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))

    # ACT mode
    optimizer.zero_grad()
    logits = model(input_ids, n_loops=None, bypass_act=False)
    loss_act = torch.nn.functional.cross_entropy(
        logits.view(-1, cfg.vocab_size), targets.view(-1)
    )
    loss_act.backward()
    optimizer.step()
    assert torch.isfinite(loss_act), "ACT-mode loss must be finite"

    # Stochastic-depth mode
    optimizer.zero_grad()
    logits = model(input_ids, n_loops=3, bypass_act=True)
    loss_sd = torch.nn.functional.cross_entropy(
        logits.view(-1, cfg.vocab_size), targets.view(-1)
    )
    loss_sd.backward()
    optimizer.step()
    assert torch.isfinite(loss_sd), "stochastic-depth-mode loss must be finite"
```

- [ ] **Step 2: Run the smoke test**

Run: `pytest tests/test_stochastic_depth.py::test_training_step_runs_in_each_mode -v`
Expected: PASS.

- [ ] **Step 3: Run the full new test file once more**

Run: `pytest tests/test_stochastic_depth.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 4: Run lint/format**

```bash
black tests/test_stochastic_depth.py training/1b_poc_fineweb.py open_mythos/main.py
ruff check --fix tests/test_stochastic_depth.py training/1b_poc_fineweb.py open_mythos/main.py
```

Expected: no changes required (or only whitespace fixes). If ruff/black makes edits, inspect and commit.

- [ ] **Step 5: Commit**

```bash
git add tests/test_stochastic_depth.py
git commit -m "test: smoke test one training step in each recurrent mode"
```

---

## Task 6: Push and verify end-to-end

- [ ] **Step 1: Confirm all new tests pass**

Run: `pytest tests/test_stochastic_depth.py -v`
Expected: 5 tests PASS.

- [ ] **Step 2: Confirm existing tests have no new failures**

Run: `pytest tests/test_main.py -v`
Expected: the 14 pre-existing failures remain (RoPE + LTI boundary); no new failures introduced.

- [ ] **Step 3: Push to origin**

```bash
git push origin main
```

---

## Post-implementation notes (not part of plan execution)

After this plan is merged, the **currently running 10B training job (56429) will auto-pick up the new default `recurrent_mode="stochastic_depth"` on the next preemption + resubmit** via `bash deploy/bluevela/bsub_1b_10b.sh`. The user has explicitly requested stochastic_depth as the default.

If the current ACT run should continue under ACT instead, set `recurrent_mode = "act"` at the top of `training/1b_poc_fineweb.py` before resubmitting. A mode switch mid-training will cause a transient loss spike of ~0.3–0.5 for a few hundred steps while the Coda re-adapts (documented in spec).
