# OpenMythos TinyStories 15M Step 100000 Seq256 Clean v6

This is the seventh archived local OpenMythos small language model trained on
Apple Silicon MPS. It warm-starts from the step 90000 seq256 EOS weighted v5
checkpoint, keeps the same 15M architecture, and continues training with
`clean_text=True` data hygiene enabled by default.

v6 improves validation loss substantially, but it slightly regresses natural-stop
reliability on the fixed 60-sample prompt-file suite. Treat it as the current
lowest-loss clean continuation, while v5 remains the strongest natural-stop
baseline under the exact local evaluation suite.

## Checkpoint

- File: `checkpoints/openmythos_tinystories_15m_step100000_seq256_clean_v6.pt`
- Size: `182M`
- SHA256: `153cc11b7e58e375750089f187b91a5594c8128fbcd4b3775d30a7b4257a2202`
- Source checkpoint: `checkpoints/tinystories_mps_small_mythos_seq256_clean_v6.pt`
- Warm-start source: `checkpoints/openmythos_tinystories_15m_step90000_seq256_eos_weighted_v5.pt`
- Tokenizer: `gpt2`
- `append_eos`: `True`
- `eos_loss_weight`: `4.0`
- `clean_text`: `True`

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
- Step: `100000`
- Final train loss: `1.7424`
- Validation loss: `1.6520`
- Observed late-run throughput: about `6.26k tok/s`
- Continuation mode: `--init-from` v5 with fresh optimizer
- Learning rate: `1e-4`
- Batch size: `8`
- Sequence length: `256`
- EOS handling: tokenizer EOS appended after each text sample; EOS target loss
  multiplied by `4.0`.
- Text hygiene: `examples.train_small_lm` cleaned common mojibake and spacing
  artifacts before tokenization.

Smoke run before the full continuation reached validation loss `1.7076` at step
`90500`, confirming the setup was healthy before the 10k-step full run.

Comparison with earlier baselines:

| Checkpoint | Step | Max seq len | Clean text | Validation loss | EOS stops |
|---|---:|---:|---|---:|---:|
| v1 | 60000 | 128 | no | 1.8518 | not measured |
| EOS v2 | 70000 | 128 | no | 1.8282 | not robust |
| EOS weighted v3 | 80000 | 128 | no | 1.8320 | 7/12 |
| EOS weighted v4 | 85000 | 128 | no | 1.8189 | 5/12 |
| Seq256 EOS weighted v5 | 90000 | 256 | generation only | 1.7513 | 60/60 |
| Seq256 clean v6 | 100000 | 256 | yes | 1.6520 | 58/60 |

## Recommended Generation

The generation and evaluation scripts clean common TinyStories mojibake and
spacing artifacts by default (`clean_text=True`). Use `--no-clean-text` only when
comparing raw decoded output.

Balanced mode:

```bash
python -m examples.generate_small_lm \
  --checkpoint checkpoints/openmythos_tinystories_15m_step100000_seq256_clean_v6.pt \
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

For user-facing demos, consider `--stop-after-sentences 3` or `4` as an extra
presentation trim. This does not prove natural EOS stopping, but it avoids
showing overly long partial stories if a sample reaches the 256-token context
limit.

Batch evaluation:

```bash
python -m examples.evaluate_small_lm_generations \
  --checkpoint checkpoints/openmythos_tinystories_15m_step100000_seq256_clean_v6.pt \
  --device mps \
  --prompt-file eval_prompts/tinystories_natural_stop_20.txt \
  --max-new-tokens 256 \
  --stop-at-eos \
  --out-dir runs
```

## Evaluation Snapshot

Latest local report:

- `runs/generation_eval_1780810665.md`

Natural stopping summary from the 60-sample prompt-file report with a 256-token
generation budget:

- Samples: `60`
- `eos_hit=True`: `58`
- `stop_reason=eos`: `58`
- `stop_reason=max_seq_len`: `2`
- `stop_reason=max_new_tokens`: `0`

The two failures were concentrated in one prompt:

- Prompt: `Lily found a small door in the garden`
- Conservative preset: `seed=207`, `generated_tokens=248`, `stop_reason=max_seq_len`
- Balanced preset: `seed=208`, `generated_tokens=248`, `stop_reason=max_seq_len`
- Creative preset for the same prompt did stop naturally: `eos_hit=True`

Follow-up context-resize probe:

- Tool: `examples.resize_small_lm_context`
- Probe checkpoint: `/tmp/openmythos_tinystories_v6_seq384_probe.pt`
- Resize: `max_seq_len=256 -> 384`
- Loaded weights: `66`
- Skipped weights: `freqs_cis` only, expected because RoPE frequencies are rebuilt
- Conservative Lily failure reached EOS at `generated_tokens=269`
- Balanced Lily failure reached EOS at `generated_tokens=300`
- Interpretation: the v6 natural-stop regression is primarily a context/budget
  issue, but the long Lily continuations become semantically weak near the end.
  v6.1 should use seq384 continuation training rather than only repackaging the
  checkpoint.

Preset summary:

| Preset | EOS stops |
|---|---:|
| conservative | 19/20 |
| balanced | 19/20 |
| creative | 20/20 |

Single-sample smoke checks:

- `Mia heard a bell in the forest`, balanced seed `408`: `eos_hit=True`,
  `generated_tokens=119`, `stop_reason=eos`
- `Lily found a small door in the garden`, balanced seed `208`,
  `max-new-tokens=180`: `eos_hit=False`, `stop_reason=max_new_tokens`

Representative sample:

> Mia heard a bell in the forest and she started to follow it. She followed the sound and saw a big tree. Mia liked the tree and wanted to climb it.
>
> "Come on, Mia, let's go!" said Tom. He was scared, but he wanted to try. He climbed up the tree, where the tree had a hole. Then, a tall tree came and knocked down the hole. The tree broke into pieces.

Representative failure-mode sample:

> Lily found a small door in the garden. It was dark and shiny. She opened it and saw many things inside. There were books, clothes, toys, and clothes.

## Known Limitations

- This is a toy-scale language model, not a general assistant model.
- Outputs are strongest for TinyStories-style short children's stories.
- v6 improves validation loss but regresses natural stopping from v5's `60/60`
  to `58/60` on the fixed 60-sample prompt-file suite.
- Some generated stories are still semantically thin, repetitive, or logically
  inconsistent.
- The model has no instruction tuning, preference tuning, safety tuning, or
  factual grounding.

## Status

Use v6 when you want the lowest validation loss and clean-text continuation
checkpoint. Use v5 when the priority is maximum natural-stop reliability under
the current local evaluation suite.
