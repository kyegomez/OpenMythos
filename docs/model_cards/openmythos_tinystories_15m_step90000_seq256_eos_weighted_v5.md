# OpenMythos TinyStories 15M Step 90000 Seq256 EOS Weighted v5

This is the sixth archived local OpenMythos small language model trained on
Apple Silicon MPS. It warm-starts from the step 85000 EOS weighted v4 checkpoint,
extends the context length from 128 to 256, and keeps weighted EOS training. It
is the first checkpoint in this local run to reach 12/12 EOS-driven natural
stops under the fixed batch generation evaluation.

## Checkpoint

- File: `checkpoints/openmythos_tinystories_15m_step90000_seq256_eos_weighted_v5.pt`
- Size: `182M`
- SHA256: `588afb3f897c1770c11d32c7bde5d3cc386ace1896c1efa69678357203b153b7`
- Source checkpoint: `checkpoints/tinystories_mps_small_mythos_seq256_eos_weighted.pt`
- Warm-start source: `checkpoints/openmythos_tinystories_15m_step85000_eos_weighted_v4.pt`
- Tokenizer: `gpt2`
- `append_eos`: `True`
- `eos_loss_weight`: `4.0`

## Model

- Parameters: `15,856,162`
- Vocab size: `50,257`
- Hidden dim: `256`
- Attention: `gqa`
- Heads: `8`
- KV heads: `2`
- Max sequence length: `256`
- Prelude layers: `1`
- Recurrent loops: `4`
- Coda layers: `1`
- MoE experts: `8`
- Shared experts: `1`
- Experts per token: `2`
- Expert dim: `256`
- LoRA rank: `8`

## Training

- Dataset: `roneneldan/TinyStories`
- Dataset config: none
- Device: Apple Silicon MPS
- Environment: `.venv-arm64`
- Python: arm64 Python 3.12
- PyTorch: `2.12.0`
- Step: `90000`
- Final train loss: `1.8821`
- Validation loss: `1.7513`
- Observed late-run throughput: about `6.8k tok/s`
- EOS handling: `examples.train_small_lm` appended `tokenizer.eos_token_id`
  after each text sample and multiplied EOS target loss by `4.0`.
- Warm-start: loaded matching weights from v4 and skipped `freqs_cis`, allowing
  RoPE frequencies to be rebuilt for `max_seq_len=256`.

Comparison with earlier baselines:

| Checkpoint | Step | Max seq len | EOS loss weight | Validation loss | EOS stops |
|---|---:|---:|---:|---:|---:|
| v1 | 60000 | 128 | 1.0 | 1.8518 | not measured |
| EOS v2 | 70000 | 128 | 1.0 | 1.8282 | not robust |
| EOS weighted v3 | 80000 | 128 | 4.0 | 1.8320 | 7/12 |
| EOS weighted v4 | 85000 | 128 | 4.0 | 1.8189 | 5/12 |
| Seq256 EOS weighted v5 | 90000 | 256 | 4.0 | 1.7513 | 60/60 |

The jump to `max_seq_len=256` removes the previous generation cap that masked
EOS behavior under 128-token context. Validation loss also improves sharply.

## Recommended Generation

The current generation and evaluation scripts clean common TinyStories mojibake
and spacing artifacts by default (`clean_text=True`). Use `--no-clean-text` only
when comparing raw decoded output.

Balanced natural-stop mode:

```bash
python -m examples.generate_small_lm \
  --checkpoint checkpoints/openmythos_tinystories_15m_step90000_seq256_eos_weighted_v5.pt \
  --prompt "Once upon a time" \
  --device mps \
  --max-new-tokens 256 \
  --temperature 0.6 \
  --top-k 20 \
  --repetition-penalty 1.18 \
  --no-repeat-ngram-size 3 \
  --stop-at-eos \
  --stop-after-sentences 0
```

Conservative short-story mode:

```bash
python -m examples.generate_small_lm \
  --checkpoint checkpoints/openmythos_tinystories_15m_step90000_seq256_eos_weighted_v5.pt \
  --prompt "Tom lost his red ball" \
  --device mps \
  --max-new-tokens 256 \
  --temperature 0.55 \
  --top-k 15 \
  --repetition-penalty 1.22 \
  --no-repeat-ngram-size 4 \
  --stop-at-eos \
  --stop-after-sentences 0
```

Batch evaluation:

```bash
python -m examples.evaluate_small_lm_generations \
  --checkpoint checkpoints/openmythos_tinystories_15m_step90000_seq256_eos_weighted_v5.pt \
  --device mps \
  --max-new-tokens 256 \
  --stop-at-eos
```

Extended prompt-file evaluation:

```bash
python -m examples.evaluate_small_lm_generations \
  --checkpoint checkpoints/openmythos_tinystories_15m_step90000_seq256_eos_weighted_v5.pt \
  --device mps \
  --prompt-file eval_prompts/tinystories_natural_stop_20.txt \
  --max-new-tokens 256 \
  --stop-at-eos
```

## Evaluation Snapshot

Latest local reports used for review:

- `runs/generation_eval_1780721968.md` (`max-new-tokens=220`)
- `runs/generation_eval_1780722076.md` (`max-new-tokens=240`)
- `runs/generation_eval_1780724137.md` (`max-new-tokens=240`, expanded 36 samples)
- `runs/generation_eval_1780796779.md` (`max-new-tokens=240`, prompt-file 60 samples)
- `runs/generation_eval_1780797531.md` (`max-new-tokens=256`, prompt-file 60 samples)

Natural stopping summary from the 60-sample prompt-file report with a 256-token
generation budget:

- Samples: `60`
- `eos_hit=True`: `60`
- `stop_reason=eos`: `60`
- `stop_reason=max_seq_len`: `0`
- `stop_reason=max_new_tokens`: `0`

The previous 240-token-budget report had one non-EOS stop:

- Prompt: `Mia heard a bell in the forest`
- Preset: balanced (`temperature=0.6`, `top_k=20`)
- Seed: `408`
- Result: `generated_tokens=240`, `stop_reason=max_new_tokens`

Single-sample diagnosis with `--max-new-tokens 400` showed the same sample hits
EOS at `generated_tokens=248`. With a 256-token budget, the 60-sample prompt
file evaluation reaches 60/60 EOS stops.

Postprocess smoke after text cleanup:

- Prompt: `Mia heard a bell in the forest`
- Seed: `408`
- Result: `eos_hit=True`, `generated_tokens=248`, `stop_reason=eos`
- Output no longer contains the earlier `â€œ` / `â€™` mojibake or `said,"`
  spacing artifact.

Representative sample:

> Lily found a small door in the garden. She opened it and saw a room with a lot of toys. She was happy and started to play.

Representative sample:

> Once upon a time, there was a little girl named Lily. She had a pet cat named Mittens. One day, Lily and Mittens were playing in the garden.

## Known Limitations

- This is a toy-scale language model, not a general assistant model.
- Outputs are strongest for TinyStories-style short children's stories.
- The expanded local prompt/preset suite reaches 60/60 EOS stops with a
  256-token generation budget; this still does not guarantee stopping across
  arbitrary prompts.
- Some generated stories are still semantically thin or repetitive.
- The model has no instruction tuning, preference tuning, safety tuning, or
  factual grounding.

## Status

Use seq256 EOS weighted v5 as the current best OpenMythos TinyStories checkpoint
for natural stopping. It supersedes v3 for EOS behavior and v4 for validation
loss under the current local evaluation setup.
