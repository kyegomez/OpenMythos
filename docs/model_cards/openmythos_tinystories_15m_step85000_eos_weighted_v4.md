# OpenMythos TinyStories 15M Step 85000 EOS Weighted v4

This is the fifth archived local OpenMythos small language model trained on
Apple Silicon MPS. It continues from the step 80000 EOS weighted v3 checkpoint
with the same EOS target loss weight. It improves held-out validation loss but
does not improve the measured natural stopping rate.

## Checkpoint

- File: `checkpoints/openmythos_tinystories_15m_step85000_eos_weighted_v4.pt`
- Size: `182M`
- SHA256: `7e8890b27b9ee20ae7f577c600fcd0ac089727a0f4320b6a7d5e235fb91dc966`
- Source checkpoint: `checkpoints/tinystories_mps_small_mythos_eos_weighted.pt`
- Previous baseline: `checkpoints/openmythos_tinystories_15m_step80000_eos_weighted_v3.pt`
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
- Max sequence length: `128`
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
- Step: `85000`
- Final train loss: `1.8253`
- Validation loss: `1.8189`
- Observed late-run throughput: about `7.1k tok/s`
- EOS handling: `examples.train_small_lm` appended `tokenizer.eos_token_id`
  after each text sample and multiplied EOS target loss by `4.0`.

Comparison with earlier baselines:

| Checkpoint | Step | EOS data | EOS loss weight | Final train loss | Validation loss | EOS stops |
|---|---:|---:|---:|---:|---:|---:|
| v0 | 40000 | no | 1.0 | 2.0668 | 1.8761 | not measured |
| v1 | 60000 | no | 1.0 | 1.9917 | 1.8518 | not measured |
| EOS v2 | 70000 | yes | 1.0 | 2.0725 | 1.8282 | not robust |
| EOS weighted v3 | 80000 | yes | 4.0 | 2.0096 | 1.8320 | 7/12 |
| EOS weighted v4 | 85000 | yes | 4.0 | 1.8253 | 1.8189 | 5/12 |

The validation loss improves, but the natural-stop hit rate drops from v3. Keep
v3 as the stronger natural-stopping baseline for now, and keep v4 as the
lower-validation-loss comparison point.

## Recommended Generation

Use the same natural-stop settings as v3:

```bash
python -m examples.generate_small_lm \
  --checkpoint checkpoints/openmythos_tinystories_15m_step85000_eos_weighted_v4.pt \
  --prompt "Once upon a time" \
  --device mps \
  --max-new-tokens 180 \
  --temperature 0.6 \
  --top-k 20 \
  --repetition-penalty 1.18 \
  --no-repeat-ngram-size 3 \
  --stop-at-eos \
  --stop-after-sentences 0
```

Batch evaluation:

```bash
python -m examples.evaluate_small_lm_generations \
  --checkpoint checkpoints/openmythos_tinystories_15m_step85000_eos_weighted_v4.pt \
  --device mps \
  --max-new-tokens 180 \
  --stop-at-eos
```

## Evaluation Snapshot

Latest local report used for review:

- `runs/generation_eval_1780719588.md`

Natural stopping summary:

- Samples: `12`
- `eos_hit=True`: `5`
- `stop_reason=eos`: `5`
- `stop_reason=max_seq_len`: `7`
- `stop_reason=max_new_tokens`: `0`

Representative EOS-stopped sample:

> Tom lost his red ball. He went to the park and told his friends. They all played together with the blue ball.
>
> After a while, they found a big, red apple.

Representative max-seq-limited sample:

> Once upon a time, there was a little girl named Lily. She loved to play with her toys and run around outside. One day, she found a big box in the park. It was heavy and had some of the old things inside.

## Known Limitations

- This is a toy-scale language model, not a general assistant model.
- Outputs are strongest for TinyStories-style short children's stories.
- Natural stopping regressed versus v3 under the fixed 12-sample evaluation.
- `max_seq_len=128` can cap generation before `--max-new-tokens` is reached.
- Longer generations can still drift, repeat, or end abruptly.
- The model has no instruction tuning, preference tuning, safety tuning, or
  factual grounding.

## Status

Use v4 when optimizing held-out validation loss. Use v3 when optimizing natural
EOS stopping. The next natural-stopping step should not blindly continue v4;
instead compare sampling settings, add EOS hit-rate summaries to the evaluator,
or train a longer-context model so `max_seq_len` stops masking EOS behavior.
