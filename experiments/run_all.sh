#!/usr/bin/env bash
# Orchestrate the full OpenMythos loop-scaling experiment on a single H100.
#
# Runs two training jobs sequentially:
#   1. looped_8   (max_loop_iters=8, the OpenMythos architecture)
#   2. baseline_1 (max_loop_iters=1, equivalent to a plain transformer)
# Then evaluates both with varying inference-time loops and plots everything.
#
# Expected wall-clock on H100 PCIe: ~4h train + ~4h train + ~30m eval = ~8.5h
set -e

cd /workspace/OpenMythos/experiments

MAX_STEPS=${MAX_STEPS:-15000}
BATCH_SIZE=${BATCH_SIZE:-32}
GRAD_ACCUM=${GRAD_ACCUM:-1}
RUNS_DIR=/workspace/runs
mkdir -p "$RUNS_DIR"

echo "============================================================"
echo "TRAIN 1/2: looped_8  (max_loop_iters=8)"
echo "============================================================"
python train.py \
    --run_name looped_8 \
    --max_loop_iters 8 \
    --max_steps $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --grad_accum_steps $GRAD_ACCUM \
    --output_dir $RUNS_DIR \
    2>&1 | tee "$RUNS_DIR/looped_8.stdout.log"

echo "============================================================"
echo "TRAIN 2/2: baseline_1  (max_loop_iters=1, plain transformer)"
echo "============================================================"
python train.py \
    --run_name baseline_1 \
    --max_loop_iters 1 \
    --max_steps $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --grad_accum_steps $GRAD_ACCUM \
    --output_dir $RUNS_DIR \
    2>&1 | tee "$RUNS_DIR/baseline_1.stdout.log"

echo "============================================================"
echo "EVAL: loop-scaling sweep"
echo "============================================================"
LOOPED_CKPT=$(ls -t $RUNS_DIR/looped_8/ckpt_*.pt | head -1)
BASELINE_CKPT=$(ls -t $RUNS_DIR/baseline_1/ckpt_*.pt | head -1)

python evaluate.py --ckpt "$LOOPED_CKPT"    \
    --loop_grid 1 2 4 6 8 12 16 \
    2>&1 | tee "$RUNS_DIR/looped_8.eval.log"

python evaluate.py --ckpt "$BASELINE_CKPT" \
    --loop_grid 1 \
    2>&1 | tee "$RUNS_DIR/baseline_1.eval.log"

python plot_results.py --runs_dir $RUNS_DIR --out_dir $RUNS_DIR/figs

echo "============================================================"
echo "DONE.  See $RUNS_DIR/figs/ for plots and summary."
echo "============================================================"
ls -la $RUNS_DIR/figs/
