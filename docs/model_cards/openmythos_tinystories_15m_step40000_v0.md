# OpenMythos TinyStories 15M Step 40000 v0

This is the first archived local OpenMythos small language model trained on
Apple Silicon MPS during the local training bring-up.

## Checkpoint

- File: `checkpoints/openmythos_tinystories_15m_step40000_v0.pt`
- Size: `182M`
- SHA256: `dec0974ab0a75f5db3b89248e9f5b7e135ab45124e32d44c1d1509a8bc3600c3`
- Source checkpoint: `checkpoints/tinystories_mps_small_mythos_step40000.pt`
- Tokenizer: `gpt2`

The checkpoint includes:

- `model`: OpenMythos state dict
- `optimizer`: AdamW state dict
- `scaler`: AMP scaler state dict, empty on MPS/no-AMP runs
- `config`: flat `MythosConfig` kwargs
- `tokenizer`: tokenizer id
- `step`: training step
- `loss`: final recorded train loss

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
- Step: `40000`
- Final train loss: `2.0668`
- Validation loss: `1.8761`
- Observed late-run throughput: about `6.2k tok/s`

Resume command used for the final training segment:

```bash
python -m examples.train_small_lm \
  --dataset roneneldan/TinyStories \
  --dataset-config "" \
  --steps 10000 \
  --batch-size 16 \
  --seq-len 128 \
  --device mps \
  --save-every 1000 \
  --eval-every 1000 \
  --log-every 50 \
  --train-tokens 10000000 \
  --eval-tokens 100000 \
  --resume checkpoints/tinystories_mps_small_mythos.pt \
  --out checkpoints/tinystories_mps_small_mythos.pt
```

## Recommended Generation

Balanced default:

```bash
python -m examples.generate_small_lm \
  --checkpoint checkpoints/openmythos_tinystories_15m_step40000_v0.pt \
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
  --checkpoint checkpoints/openmythos_tinystories_15m_step40000_v0.pt \
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
  --checkpoint checkpoints/openmythos_tinystories_15m_step40000_v0.pt \
  --device mps \
  --max-new-tokens 140
```

## Evaluation Snapshot

Latest local report used for review:

- `runs/generation_eval_1780650982.md`

Representative balanced sample:

> Once upon a time, there was a little girl named Lily. She loved to play with her toys and run around outside. One day, she went to the park to play in the park. While playing, she saw something shiny on the ground.

Representative conservative sample:

> Tom lost his red ball. He looked everywhere but he could not find it. He felt sad and lonely.

## Known Limitations

- This is a toy-scale language model, not a general assistant model.
- Outputs are strongest for TinyStories-style short children's stories.
- Longer generations can still drift, repeat, or end abruptly.
- The model has no instruction tuning, preference tuning, safety tuning, or factual grounding.
- The training data and tokenizer do not provide a strong explicit story-stop signal; generation quality benefits from `--stop-after-sentences`.

## Status

Keep this checkpoint as the first local OpenMythos TinyStories baseline.
Future checkpoints should be compared against this file with
`examples.evaluate_small_lm_generations`.
