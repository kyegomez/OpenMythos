"""
OpenMythos Training Script — MLX native causal language model training.

Usage:
  python train.py                          # tiny smoke-test with random data
  python train.py --variant 1b --steps 1000 --lr 3e-4
  python train.py --variant 3b --data path/to/tokens.npy --checkpoint ckpt/
"""

import argparse
import time
import os
import numpy as np
from pathlib import Path
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    logger = logging.getLogger("train")

from open_mythos.main import OpenMythos, MythosConfig
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VARIANTS = {
    "tiny": MythosConfig(
        vocab_size=8192, dim=256, n_heads=4, max_seq_len=128,
        max_loop_iters=4, prelude_layers=1, coda_layers=1,
        n_experts=8, n_shared_experts=1, n_experts_per_tok=2, expert_dim=64,
    ),
    "small": MythosConfig(
        vocab_size=50257, dim=512, n_heads=8, max_seq_len=512,
        max_loop_iters=8, prelude_layers=1, coda_layers=1,
        n_experts=16, n_shared_experts=2, n_experts_per_tok=2, expert_dim=128,
    ),
    "1b": MythosConfig(
        vocab_size=50257, dim=2048, n_heads=16, max_seq_len=1024,
        max_loop_iters=16, prelude_layers=2, coda_layers=2,
        n_experts=16, n_shared_experts=2, n_experts_per_tok=2, expert_dim=256,
    ),
    # Mythos-2B: 823M unique params, ~9.2h/30ksteps @ 927 tok/s on M2 Ultra 64GB
    # batch=1, seq=1024, n_loops=6 verified safe (9.88GB weights+optim, stable Metal pool)
    "2b": MythosConfig(
        vocab_size=50257, dim=3072, n_heads=24, max_seq_len=1024,
        max_loop_iters=24, prelude_layers=2, coda_layers=2,
        n_experts=24, n_shared_experts=2, n_experts_per_tok=2, expert_dim=384,
    ),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class TokenDataset:
    """Flat token array split into (seq_len+1) chunks for causal LM."""

    def __init__(self, tokens: np.ndarray, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_chunks = (len(tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_chunks

    def get_batch(self, indices: np.ndarray) -> tuple[mx.array, mx.array]:
        rows = []
        for i in indices:
            start = i * self.seq_len
            rows.append(self.tokens[start : start + self.seq_len + 1])
        arr = np.stack(rows)
        x = mx.array(arr[:, :-1], dtype=mx.uint32)
        y = mx.array(arr[:, 1:],  dtype=mx.uint32)
        return x, y


def make_random_dataset(vocab_size: int, seq_len: int, n: int = 10_000) -> TokenDataset:
    tokens = np.random.randint(0, vocab_size, size=(n * seq_len + 1,), dtype=np.int32)
    return TokenDataset(tokens, seq_len)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def loss_fn(model: OpenMythos, x: mx.array, y: mx.array, n_loops: int) -> mx.array:
    logits = model(x, n_loops=n_loops)                      # (B, T, V)
    B, T, V = logits.shape
    return mx.mean(
        nn.losses.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))
    )


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model: OpenMythos, optimizer: optim.Adam, step: int, path: str) -> None:
    ckpt_dir = Path(path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights_path = str(ckpt_dir / f"step_{step:06d}.npz")
    model.save_weights(weights_path)
    logger.info(f"Checkpoint saved: {weights_path}")


def load_checkpoint(model: OpenMythos, path: str) -> int:
    ckpts = sorted(Path(path).glob("step_*.npz"))
    if not ckpts:
        return 0
    latest = str(ckpts[-1])
    model.load_weights(latest)
    step = int(ckpts[-1].stem.split("_")[1])
    logger.info(f"Resumed from checkpoint: {latest} (step {step})")
    return step


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    cfg = VARIANTS[args.variant]
    logger.info(f"Variant: {args.variant} | dim={cfg.dim} | experts={cfg.n_experts}")

    model = OpenMythos(cfg)
    mx.eval(model.parameters())

    # Dataset
    if args.data and Path(args.data).exists():
        tokens = np.load(args.data)
        dataset = TokenDataset(tokens, cfg.max_seq_len)
        logger.info(f"Dataset: {args.data} ({len(dataset):,} chunks)")
    else:
        logger.warning("No --data provided, using random tokens for smoke test")
        dataset = make_random_dataset(cfg.vocab_size, cfg.max_seq_len)

    warmup = optim.linear_schedule(0, args.lr, steps=args.warmup_steps)
    decay  = optim.cosine_decay(args.lr, decay_steps=max(args.steps - args.warmup_steps, 1), end=args.lr * 0.1)
    schedule = optim.join_schedules([warmup, decay], [args.warmup_steps])
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.1)

    start_step = 0
    if args.checkpoint and Path(args.checkpoint).exists():
        start_step = load_checkpoint(model, args.checkpoint)

    loss_and_grad = nn.value_and_grad(model, partial(loss_fn, n_loops=args.n_loops))

    rng = np.random.default_rng(42)
    log_loss = 0.0
    t0 = time.time()

    logger.info(f"Training for {args.steps} steps | batch={args.batch} | lr={args.lr} | warmup={args.warmup_steps}")

    for step in range(start_step, start_step + args.steps):
        indices = rng.integers(0, len(dataset), size=args.batch)
        x, y = dataset.get_batch(indices)

        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        log_loss += loss.item()

        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            avg_loss = log_loss / args.log_every
            tokens_per_sec = args.batch * cfg.max_seq_len * args.log_every / elapsed
            logger.info(
                f"step {step+1:6d} | loss {avg_loss:.4f} | "
                f"{tokens_per_sec:,.0f} tok/s | {elapsed:.1f}s"
            )
            log_loss = 0.0
            t0 = time.time()

        if args.checkpoint and (step + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, step + 1, args.checkpoint)

    logger.info("Training complete.")
    if args.checkpoint:
        save_checkpoint(model, optimizer, start_step + args.steps, args.checkpoint)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenMythos MLX Training")
    p.add_argument("--variant",    default="tiny",  choices=list(VARIANTS), help="Model size")
    p.add_argument("--data",       default=None,    help="Path to .npy token file")
    p.add_argument("--checkpoint", default=None,    help="Checkpoint directory")
    p.add_argument("--steps",      type=int,   default=200,   help="Training steps")
    p.add_argument("--batch",      type=int,   default=4,     help="Batch size")
    p.add_argument("--lr",         type=float, default=3e-4,  help="Learning rate")
    p.add_argument("--n_loops",    type=int,   default=4,     help="Recurrent loops during training")
    p.add_argument("--log_every",  type=int,   default=10,    help="Log interval (steps)")
    p.add_argument("--save_every",   type=int,   default=100,  help="Checkpoint interval (steps)")
    p.add_argument("--warmup_steps", type=int,   default=100,  help="Linear warmup steps before cosine decay")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
