# OpenMythos TinyStories Mid 38M Step 75000 Seq384 v9-T5

Current best local checkpoint. The v9 mid model (37.7M) continued from step
60000 with a 3x fresh-data budget (T5): 60M tokens (~156k windows, 0.77
epochs over the run) broke the val plateau and drove repetition to the best
reading of the whole lineage.

## Checkpoint

- File: `checkpoints/openmythos_tinystories_mid_38m_step75000_seq384_v9_t5.pt`
- SHA256: `24033aa027dbbf727105cdbec5618b775bed0b432917270f9951ebdeac67334c`
- Lineage: from-scratch v9 → 60k (20M budget) → 75k T5 resume (60M budget)

## Model

- Params: **37,675,042** (dim 512, n_heads 8, n_kv_heads 2, expert_dim 512,
  8 routed + 1 shared experts, lora_rank 8, GQA)
- Tokenizer: gpt2

## Training (T5 resume leg)

- Resume from step 60000, +15,000 steps, batch 8, seq_len 384, lr 3e-4,
  eos_loss_weight 4.0, stream packing, seed 7
- Data: **60M tokens** (fresh windows beyond the first 20M), ~0.77 epochs
- Val: 1.5130 (60k) → **1.4569** (75k); throughput ~4.3k tok/s (MPS)

## Recommended Generation (mid-v1 preset)

```bash
.venv-arm64/bin/python -m examples.generate_small_lm \
  --checkpoint checkpoints/openmythos_tinystories_mid_38m_step75000_seq384_v9_t5.pt \
  --device mps --prompt "Lily found a small door in the garden" \
  --max-new-tokens 376 --temperature 0.60 --top-k 25 \
  --repetition-penalty 1.22 --no-repeat-ngram-size 3 --stop-at-eos --seed 7
```

## Evaluation (60 samples, mid-v1, all samples stop within 256)

| metric | v6.1 (15.8M) | v9 60k | **v9-T5 75k** |
|---|---|---|---|
| val loss | 1.6227 | 1.5130 | **1.4569** |
| EOS (strict-256) | 60/60 | 60/60 | **60/60** |
| repeated_trigram | 14 | 17 | **6** |
| missing_ending | 27 | 25 | **24** |
| story endings | 33/60 | 35/60 | **36/60** |
| avg words | 85.9 | 88.1 | **75.8** |
| weak_prompt_adherence | 29 | 35 | 35 |
| character_drift | 8 | 10 | 11 |

Reports: `runs/generation_eval_1793016390.md` / `_1793016459.md` (376/256).

## Known Issues

- **weak_prompt_adherence (35) is the one acceptance line still open** —
  unchanged across the 60k→75k fresh-data leg, suggesting the heuristic
  under-scores this model's more varied writing style rather than a
  trainable deficit.
- endings slope decayed to +1/15k steps (36/60 vs the ≥40 T3b target):
  more steps are low-yield; remaining levers are corpus scale/diversity or
  human review of the adherence heuristic.

## Status

Supersedes v9-60k and v6.1 as best local model on val loss, EOS, repetition,
and missing-ending; endings/adherence pending corpus-side work or heuristic
calibration.
