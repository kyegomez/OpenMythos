# OpenMythos TinyStories 15M Step 70000 EOS v2

This is the third archived local OpenMythos small language model trained on
Apple Silicon MPS. It continues from the step 60000 v1 checkpoint and enables
explicit EOS tokens in the training stream.

## Checkpoint

- File: `checkpoints/openmythos_tinystories_15m_step70000_eos_v2.pt`
- Size: `182M`
- SHA256: `dcf2390f580f475d6de1d5260bf12450baf57b1c3822df4046df92bba2436acd`
- Source checkpoint: `checkpoints/tinystories_mps_small_mythos_eos.pt`
- Previous baseline: `checkpoints/openmythos_tinystories_15m_step60000_v1.pt`
- Tokenizer: `gpt2`
- `append_eos`: `True`

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
- Step: `70000`
- Final train loss: `2.0725`
- Validation loss: `1.8282`
- Observed late-run throughput: about `7.0k tok/s`
- EOS handling: `examples.train_small_lm` appended `tokenizer.eos_token_id`
  after each text sample during the 60000 to 70000 segment.

Comparison with earlier baselines:

| Checkpoint | Step | EOS data | Final train loss | Validation loss |
|---|---:|---:|---:|---:|
| v0 | 40000 | no | 2.0668 | 1.8761 |
| v1 | 60000 | no | 1.9917 | 1.8518 |
| EOS v2 | 70000 | yes | 2.0725 | 1.8282 |

The validation loss continued to improve after adding EOS data. The train loss
is not directly comparable with v1 because the training token stream changed by
adding EOS boundary tokens.

## Recommended Generation

Balanced default with EOS stopping enabled:

```bash
python -m examples.generate_small_lm \
  --checkpoint checkpoints/openmythos_tinystories_15m_step70000_eos_v2.pt \
  --prompt "Once upon a time" \
  --device mps \
  --max-new-tokens 140 \
  --temperature 0.6 \
  --top-k 20 \
  --repetition-penalty 1.18 \
  --no-repeat-ngram-size 3 \
  --stop-at-eos \
  --stop-after-sentences 4
```

Conservative short-story mode:

```bash
python -m examples.generate_small_lm \
  --checkpoint checkpoints/openmythos_tinystories_15m_step70000_eos_v2.pt \
  --prompt "Tom lost his red ball" \
  --device mps \
  --max-new-tokens 120 \
  --temperature 0.55 \
  --top-k 15 \
  --repetition-penalty 1.22 \
  --no-repeat-ngram-size 4 \
  --stop-at-eos \
  --stop-after-sentences 3
```

Batch evaluation:

```bash
python -m examples.evaluate_small_lm_generations \
  --checkpoint checkpoints/openmythos_tinystories_15m_step70000_eos_v2.pt \
  --device mps \
  --max-new-tokens 140 \
  --stop-at-eos
```

## Evaluation Snapshot

Latest local report used for review:

- `runs/generation_eval_1780689867.md`

Representative balanced sample:

> Once upon a time, there was a little girl named Lily. She loved to play outside in the rain. One day, she went outside and it started to rain. The wind came down and the rain made.

Representative conservative sample:

> Tom lost his red ball. He is sad and angry.
>
> "Mom, why did you do that?"

Raw `--stop-at-eos` sample without sentence trimming still continued until the
token budget, so this checkpoint should be considered EOS-aware but not yet a
reliably self-stopping model. `--stop-after-sentences` remains recommended.

## Known Limitations

- This is a toy-scale language model, not a general assistant model.
- Outputs are strongest for TinyStories-style short children's stories.
- EOS was only introduced for the final 10000 training steps, so natural EOS
  stopping is not yet reliable.
- Longer generations can still drift, repeat, or end abruptly.
- The model has no instruction tuning, preference tuning, safety tuning, or
  factual grounding.

## Status

Use EOS v2 as the latest local OpenMythos TinyStories baseline when testing EOS
behavior. Preserve v1 for non-EOS comparison. If improving natural stopping is
the goal, continue EOS training or increase EOS learning signal rather than only
relying on decode-time sentence trimming.
