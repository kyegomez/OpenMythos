# OpenMythos TinyStories Mid 38M Step 60000 Seq384 v9

First capacity-scaled small LM (T3b): dim 512 lineage trained from scratch on
Apple Silicon MPS. It supersedes the 15.8M v6.1 on validation loss and EOS
discipline, and matches-or-beats it on story endings and missing-ending
rates when paired with the `mid-v1` sampling preset.

## Checkpoint

- File: `checkpoints/openmythos_tinystories_mid_38m_step60000_seq384_v9.pt`
- SHA256: `43a88b98bbedd3edab503286890f9ad0c34f528664b230cd0fdfe6dd165cc750`
- Working copy: `checkpoints/tinystories_mps_mid_mythos_seq384_v9.pt` (same weights)

## Model

- Params: **37,675,042** (dim 512, n_heads 8, n_kv_heads 2, expert_dim 512,
  8 routed + 1 shared experts, lora_rank 8, 1+1 prelude/coda layers, GQA)
- Tokenizer: gpt2 (vocab 50,257)

## Training

- From scratch (dim mismatch rejects cross-width warm-start by design)
- 60,000 steps, batch 8, seq_len 384, lr 3e-4, eos_loss_weight 4.0,
  stream packing, clean_text, seed 7
- Data budget: 20M tokens (~52k windows, ~9.2 epochs at 60k steps — the
  upper bound before memorization artifacts appear, see Known Issues)
- Val loss curve: 30k `1.5515` → 45k `1.5142` → 60k **`1.5130`** (plateau ~1.51)
- Throughput: ~4.7k tok/s (MPS)

## Recommended Generation (mid-v1 preset)

```bash
.venv-arm64/bin/python -m examples.generate_small_lm \
  --checkpoint checkpoints/openmythos_tinystories_mid_38m_step60000_seq384_v9.pt \
  --device mps --prompt "Lily found a small door in the garden" \
  --max-new-tokens 376 --temperature 0.60 --top-k 25 \
  --repetition-penalty 1.22 --no-repeat-ngram-size 3 --stop-at-eos --seed 7
```

`--preset-set mid-v1` in the evaluator: conservative(0.55/20/1.22/3),
balanced(0.60/25/1.22/3), creative(0.70/30/1.18/3).

## Evaluation (60 samples, coherence prompts)

| metric | v6.1 (15.8M) | v9 60k + mid-v1 |
|---|---|---|
| EOS (strict-256) | 60/60 | **60/60** |
| val loss | 1.6227 | **1.5130** |
| story endings | 33/60 | **35/60** |
| missing_ending | 27 | **25** |
| repeated_trigram | 14 | 17 |
| weak_prompt_adherence | 29 | 35 |
| character_drift | 8 | 10 |
| avg words | 85.9 | 88.1 |

Reports: `runs/generation_eval_1793004646.md` (coherence-v2),
`runs/generation_eval_1793004996.md` / `_1793005163.md` (mid-v1, 376/256).

## Known Issues

- weak_prompt_adherence (35 vs baseline 29) is the main remaining gap;
  structural cause is ~9.2 epochs over a 20M-token budget. Next lever is
  data-side (larger/dedup-fixed corpus), not more steps.
- Endings 35/60 vs the ≥40/60 T3b target: plateaued across 45k→60k with the
  legacy preset; +4 from the mid-v1 preset. Not accepted yet.
- The 45k checkpoint (best generation behavior under the legacy preset) was
  overwritten by the 60k resume — archive before resuming, always.

## Status

Supersedes v6.1 as best local checkpoint on val loss and EOS; T3b acceptance
pending on endings ≥ 40/60 (35/60) and adherence data-side work.
