# Loop-scaling validation experiments

A minimal, reproducible training + evaluation pipeline that measures how
OpenMythos's validation perplexity changes as you vary the number of
recurrent loops at **inference time**.

The pipeline trains two comparison models (same ~118M-parameter MLA+MoE
backbone, same ~491M tokens of FineWeb-Edu, same optimizer / schedule)
that differ only in their training-time loop strategy:

| Run | Training `n_loops` per step | Role |
|---|---|---|
| `looped_8` | fixed at 8 | the default OpenMythos training style |
| `baseline_1` | fixed at 1 | dense-equivalent ablation |
| `looped_random` (optional) | uniformly sampled from `{4, 6, 8, 12, 16}` | tests whether random-loop training gives monotonic depth extrapolation |

After training, `evaluate.py` sweeps `n_loops ∈ {1, 2, 4, 6, 8, 12, 16}`
at inference on a held-out FineWeb-Edu slice (`--skip_docs 2_000_000`
ensures no train/val overlap) and logs PPL + generation samples.
`plot_results.py` produces three figures: training loss, ρ(A) over
steps, and the inference-time loop-scaling curve.

## Usage

Requires a single GPU with ≥ 48 GB VRAM for `batch_size=32` at `n_loops=8`
(H100 80 GB, A100 80 GB, or A40 48 GB). On H100 SXM each 15k-step run
takes ~4 hours; `looped_random` needs smaller batches to fit `n_loops=16`
and takes ~3.5 hours.

```bash
cd experiments
pip install matplotlib datasets transformers loguru

# Looped (recommended default)
python train.py --run_name looped_8 --max_loop_iters 8 --max_steps 15000

# Baseline for comparison (trains ~3× faster since n_loops=1)
python train.py --run_name baseline_1 --max_loop_iters 1 --max_steps 15000

# Optional: random-loop training for depth-extrapolation ablation
python train.py --run_name looped_random \
    --max_loop_iters 16 \
    --loop_sample_mode random_set --loop_choices 4 6 8 12 16 \
    --batch_size 16 --grad_accum_steps 2 --max_steps 15000

# Inference-time loop sweep + generation samples
python evaluate.py --ckpt /workspace/runs/looped_8/ckpt_15000.pt \
    --loop_grid 1 2 4 6 8 12 16
python evaluate.py --ckpt /workspace/runs/baseline_1/ckpt_15000.pt \
    --loop_grid 1

python plot_results.py --runs_dir /workspace/runs --out_dir /workspace/runs/figs
```

Or run all three phases end-to-end with default settings:

```bash
bash run_all.sh     # drives looped_8 + baseline_1 + evaluate + plot
```

## Files

| File | Purpose |
|---|---|
| `config.py` | `mythos_150m()` MLA+MoE config (actual param count 117.8M) and `TrainConfig` dataclass |
| `data.py` | Streaming FineWeb-Edu loader with `skip_docs` for clean train/val split |
| `train.py` | AdamW + cosine schedule training with per-step `n_loops` logging; supports `--loop_sample_mode {fixed,random_set}` |
| `evaluate.py` | Loads a checkpoint, runs PPL sweep over `--loop_grid`, emits generation samples at trained and 2× loops |
| `plot_results.py` | Parses all `<run>/train.log` + `<run>/eval_ckpt_*.json` under a runs directory and draws three comparison figures |
| `run_all.sh` | Orchestrator: looped_8 → baseline_1 → eval → plot |

## What the logs contain

`train.log` is tab-separated with headers
`step  tokens  n_loops  lr  loss  grad_norm  rho_A  step_s  tok_per_s  gpu_mem_gb`.
The `n_loops` column records the value actually used at each training
step (constant in `fixed` mode, varying in `random_set` mode) so you can
post-hoc slice losses by training-loop-depth.
