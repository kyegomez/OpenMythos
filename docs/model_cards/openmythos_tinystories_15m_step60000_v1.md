# OpenMythos TinyStories 15M Step 60000 v1

This is the second archived local OpenMythos small language model trained on
Apple Silicon MPS. It extends the v0 baseline from step 40000 to step 60000.

## Checkpoint

- File: `checkpoints/openmythos_tinystories_15m_step60000_v1.pt`
- Size: `182M`
- SHA256: `102772b5297c6d5a862928410d4b7175a131f23a3b14e54b23e17b5c01c323d7`
- Source checkpoint: `checkpoints/tinystories_mps_small_mythos.pt`
- Previous baseline: `checkpoints/openmythos_tinystories_15m_step40000_v0.pt`
- Tokenizer: `gpt2`

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
- Step: `60000`
- Final train loss: `1.9917`
- Validation loss: `1.8518`
- Observed late-run throughput: about `5.2k tok/s`

Comparison with v0:

| Checkpoint | Step | Final train loss | Validation loss |
|---|---:|---:|---:|
| v0 | 40000 | 2.0668 | 1.8761 |
| v1 | 60000 | 1.9917 | 1.8518 |

The 40000 to 60000 segment still improved validation loss, but the gain was
small (`-0.0243`), so future training should be justified by generation quality
comparisons rather than loss alone.

## Recommended Generation

Balanced default:

```bash
python -m examples.generate_small_lm \
  --checkpoint checkpoints/openmythos_tinystories_15m_step60000_v1.pt \
  --prompt "Once upon a time" \
  --device mps \
  --max-new-tokens 140 \
  --temperature 0.6 \
  --top-k 20 \
  --repetition-penalty 1.18 \
  --no-repeat-ngram-size 3 \
  --stop-after-sentences 4
```

Conservative short-story mode:

```bash
python -m examples.generate_small_lm \
  --checkpoint checkpoints/openmythos_tinystories_15m_step60000_v1.pt \
  --prompt "Tom lost his red ball" \
  --device mps \
  --max-new-tokens 120 \
  --temperature 0.55 \
  --top-k 15 \
  --repetition-penalty 1.22 \
  --no-repeat-ngram-size 4 \
  --stop-after-sentences 3
```

Batch evaluation:

```bash
python -m examples.evaluate_small_lm_generations \
  --checkpoint checkpoints/openmythos_tinystories_15m_step60000_v1.pt \
  --device mps \
  --max-new-tokens 140
```

## Evaluation Snapshot

Latest local report used for review:

- `runs/generation_eval_1780684492.md`

Representative balanced sample:

> Once upon a time, there was a little girl named Lily. She loved to play with her toys and run around outside. One day, she went to the park to play on the swings. She saw a big slide and wanted to try it.

Representative conservative sample:

> Tom lost his red ball. He could not find it. Tom and the boy were sad.

Representative creative sample:

> Ben saw a tiny dragon under the bed. The dragon had wings and a mouth. It was not hungry or lonely. It looked at Ben with its big eyes.

## Known Limitations

- This is a toy-scale language model, not a general assistant model.
- Outputs are strongest for TinyStories-style short children's stories.
- Longer generations can still drift, repeat, or end abruptly.
- Some samples still show local incoherence, weak causality, or odd object substitutions.
- The model has no instruction tuning, preference tuning, safety tuning, or factual grounding.

## Status

Use v1 as the latest local OpenMythos TinyStories baseline. Preserve v0 for
comparison. If continuing training, archive the next checkpoint as a new version
instead of overwriting v1.
