# OpenMythos TinyStories 15M Step 80000 EOS Weighted v3

This is the fourth archived local OpenMythos small language model trained on
Apple Silicon MPS. It continues from the step 70000 EOS v2 checkpoint and adds
weighted EOS training, making it the first local checkpoint that shows real
EOS-driven natural stopping in batch generation.

## Checkpoint

- File: `checkpoints/openmythos_tinystories_15m_step80000_eos_weighted_v3.pt`
- Size: `182M`
- SHA256: `83efd35dd108bc0550035ceecb5155f84eb83159dcf1a40f7d1514195ee6226f`
- Source checkpoint: `checkpoints/tinystories_mps_small_mythos_eos_weighted.pt`
- Previous baseline: `checkpoints/openmythos_tinystories_15m_step70000_eos_v2.pt`
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
- Step: `80000`
- Final train loss: `2.0096`
- Validation loss: `1.8320`
- Observed late-run throughput: about `5.7k tok/s`
- EOS handling: `examples.train_small_lm` appended `tokenizer.eos_token_id`
  after each text sample and multiplied EOS target loss by `4.0`.

Comparison with earlier baselines:

| Checkpoint | Step | EOS data | EOS loss weight | Final train loss | Validation loss |
|---|---:|---:|---:|---:|---:|
| v0 | 40000 | no | 1.0 | 2.0668 | 1.8761 |
| v1 | 60000 | no | 1.0 | 1.9917 | 1.8518 |
| EOS v2 | 70000 | yes | 1.0 | 2.0725 | 1.8282 |
| EOS weighted v3 | 80000 | yes | 4.0 | 2.0096 | 1.8320 |

The validation loss remains close to EOS v2 while natural stopping improves.
The train loss is not directly comparable with earlier checkpoints because the
training objective now upweights EOS targets.

## Recommended Generation

Balanced natural-stop mode:

```bash
python -m examples.generate_small_lm \
  --checkpoint checkpoints/openmythos_tinystories_15m_step80000_eos_weighted_v3.pt \
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

Conservative short-story mode:

```bash
python -m examples.generate_small_lm \
  --checkpoint checkpoints/openmythos_tinystories_15m_step80000_eos_weighted_v3.pt \
  --prompt "Tom lost his red ball" \
  --device mps \
  --max-new-tokens 180 \
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
  --checkpoint checkpoints/openmythos_tinystories_15m_step80000_eos_weighted_v3.pt \
  --device mps \
  --max-new-tokens 180 \
  --stop-at-eos
```

## Evaluation Snapshot

Latest local report used for review:

- `runs/generation_eval_1780716355.md`

Natural stopping summary:

- Samples: `12`
- `eos_hit=True`: `7`
- `stop_reason=eos`: `7`
- `stop_reason=max_seq_len`: `5`
- `stop_reason=max_new_tokens`: `0`

Representative raw sample without sentence trimming:

> Ben saw a tiny dragon under the bed. He was curious and wanted to play with it. He pulled it out of the head and said, "Hello, I promise! It's my birthday place!"
>
> The dragon opened its eyes and flew away. Ben smiled and said thank you. He liked the surprise too. He decided to be kind and share his gift. He wanted to learn more.

Representative balanced sample:

> Lily found a small door in the garden. It was closed. She felt happy. She opened the door and walked to it.
>
> "Hello, Lily," she said. "What are you doing on here?"
>
> Lily opened her eyes. It wasn't an unknown place. But then she saw a big tree with many leaves. She reached out and grabbed some of her friends. They were all playing together.

## Known Limitations

- This is a toy-scale language model, not a general assistant model.
- Outputs are strongest for TinyStories-style short children's stories.
- Natural stopping improved but is not fully reliable: 5 of 12 review samples
  still stopped at `max_seq_len` rather than EOS.
- `max_seq_len=128` can cap generation before `--max-new-tokens` is reached.
- Longer generations can still drift, repeat, or end abruptly.
- The model has no instruction tuning, preference tuning, safety tuning, or
  factual grounding.

## Status

Use EOS weighted v3 as the latest local OpenMythos TinyStories baseline when
testing natural stopping. It is the first checkpoint in this run that produces
measurable `stop_reason=eos` results, but another short weighted run or a larger
`max_seq_len` training pass may be needed before EOS stopping is robust.
