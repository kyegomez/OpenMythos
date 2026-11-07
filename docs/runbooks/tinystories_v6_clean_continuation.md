# TinyStories v6 Clean-Text Continuation Runbook

This runbook continues the current best local checkpoint, v5, into a v6 run
focused on cleaner data hygiene and story coherence. It keeps the same 15M
OpenMythos small-LM architecture and trains from the v5 weights with
`clean_text=True` enabled by default.

## Goal

- Start from: `checkpoints/openmythos_tinystories_15m_step90000_seq256_eos_weighted_v5.pt`
- Produce: `checkpoints/tinystories_mps_small_mythos_seq256_clean_v6.pt`
- Target step: `100000`
- Keep natural stopping: EOS weighted training with `--eos-loss-weight 4.0`
- Improve data hygiene: default text cleanup before tokenization
- Preserve context length: `seq_len=256`

Expected local speed on Apple Silicon MPS is roughly `5k-7k tok/s`. The full
10k-step run may take around one to two hours depending on thermal state and
other workloads.

## Preflight

Run from the repository root:

```bash
cd /Users/macos/Desktop/Claude/OpenMythos
source .venv-arm64/bin/activate

python - <<'PY'
import torch
print("torch", torch.__version__)
print("mps built", torch.backends.mps.is_built())
print("mps available", torch.backends.mps.is_available())
PY

ls -lh checkpoints/openmythos_tinystories_15m_step90000_seq256_eos_weighted_v5.pt
```

Expected:

- `mps available True`
- v5 checkpoint exists and is about `182M`

## Optional Smoke Run

Use this when you want a short safety check before the full run:

```bash
python -m examples.train_small_lm \
  --dataset roneneldan/TinyStories \
  --dataset-config "" \
  --init-from checkpoints/openmythos_tinystories_15m_step90000_seq256_eos_weighted_v5.pt \
  --out /tmp/openmythos_tinystories_v6_smoke.pt \
  --steps 500 \
  --batch-size 8 \
  --seq-len 256 \
  --device mps \
  --lr 1e-4 \
  --n-loops 4 \
  --log-every 50 \
  --eval-every 250 \
  --eval-batches 20 \
  --save-every 500 \
  --train-tokens 1000000 \
  --eval-tokens 50000 \
  --eos-loss-weight 4.0
```

Proceed to the full run if:

- Training starts from `step=90000` via `[init-from]`
- The setup line prints `clean_text=True`
- Loss remains finite
- MPS throughput is stable after the first few dozen steps

## Full v6 Run

```bash
python -m examples.train_small_lm \
  --dataset roneneldan/TinyStories \
  --dataset-config "" \
  --init-from checkpoints/openmythos_tinystories_15m_step90000_seq256_eos_weighted_v5.pt \
  --out checkpoints/tinystories_mps_small_mythos_seq256_clean_v6.pt \
  --steps 10000 \
  --batch-size 8 \
  --seq-len 256 \
  --device mps \
  --lr 1e-4 \
  --n-loops 4 \
  --log-every 50 \
  --eval-every 500 \
  --eval-batches 20 \
  --save-every 1000 \
  --train-tokens 20000000 \
  --eval-tokens 200000 \
  --eos-loss-weight 4.0
```

Notes:

- `--init-from` loads model weights but starts a fresh optimizer, which is
  intentional for this lower-learning-rate v6 continuation.
- The checkpoint step should advance from `90000` to `100000`.
- If MPS runs out of memory, rerun with `--batch-size 4`; keep the other flags.
- Do not use `--no-clean-text` for v6 unless intentionally running an ablation.

## Stop / Continue Rules

Safe to stop manually after any `[saved]` line. Prefer stopping immediately after
an eval/save boundary such as `95000` or `100000`.

Continue the full run if:

- Validation loss is flat or below the v5 baseline `1.7513`
- Generated samples become less garbled or more coherent
- EOS hit rate stays near `60/60` on the prompt-file suite

Pause and review before more training if:

- Validation loss rises above about `1.82` for multiple evals
- Outputs become noticeably shorter, repetitive, or less story-like
- EOS stops become too early across many prompts

## Evaluation After Training

Batch natural-stop evaluation:

```bash
python -m examples.evaluate_small_lm_generations \
  --checkpoint checkpoints/tinystories_mps_small_mythos_seq256_clean_v6.pt \
  --device mps \
  --prompt-file eval_prompts/tinystories_natural_stop_20.txt \
  --max-new-tokens 256 \
  --stop-at-eos \
  --out-dir runs
```

Single-sample checks:

```bash
python -m examples.generate_small_lm \
  --checkpoint checkpoints/tinystories_mps_small_mythos_seq256_clean_v6.pt \
  --prompt "Mia heard a bell in the forest" \
  --device mps \
  --max-new-tokens 256 \
  --temperature 0.6 \
  --top-k 20 \
  --repetition-penalty 1.18 \
  --no-repeat-ngram-size 3 \
  --stop-at-eos \
  --stop-after-sentences 0 \
  --seed 408

python -m examples.generate_small_lm \
  --checkpoint checkpoints/tinystories_mps_small_mythos_seq256_clean_v6.pt \
  --prompt "Lily found a small door in the garden" \
  --device mps \
  --max-new-tokens 180 \
  --temperature 0.6 \
  --top-k 20 \
  --repetition-penalty 1.18 \
  --no-repeat-ngram-size 3 \
  --stop-at-eos \
  --stop-after-sentences 0 \
  --seed 208
```

## Archive Checklist

Only archive v6 if it beats or matches v5 on the main gates:

- Validation loss is `<= 1.7513`, or samples are visibly better without losing
  EOS reliability
- Prompt-file EOS hit rate is near `60/60`
- No mojibake artifacts appear in representative outputs

Then capture metadata:

```bash
cp checkpoints/tinystories_mps_small_mythos_seq256_clean_v6.pt \
  checkpoints/openmythos_tinystories_15m_step100000_seq256_clean_v6.pt

ls -lh checkpoints/openmythos_tinystories_15m_step100000_seq256_clean_v6.pt
shasum -a 256 checkpoints/openmythos_tinystories_15m_step100000_seq256_clean_v6.pt

python - <<'PY'
import torch
path = "checkpoints/openmythos_tinystories_15m_step100000_seq256_clean_v6.pt"
ckpt = torch.load(path, map_location="cpu", weights_only=False)
for key in ["step", "loss", "append_eos", "eos_loss_weight", "clean_text", "tokenizer"]:
    print(key, ckpt.get(key))
print("max_seq_len", ckpt["config"]["max_seq_len"])
PY
```

After this, create a new model card in `docs/model_cards/` and paste the latest
training/evaluation snippets into the progress log.
