"""
OpenMythos Evaluation Script — Perplexity measurement + text generation samples.

Usage:
  python eval.py --checkpoint ckpt/1b-fineweb-edu --data data/fineweb_edu.npy
  python eval.py --checkpoint ckpt/1b-fineweb-edu --prompt "The history of science"
  python eval.py --checkpoint ckpt/1b-fineweb-edu --data data/fineweb_edu.npy --prompt "Once upon a time"
"""

import argparse
import math
import time
import numpy as np
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    logger = logging.getLogger("eval")

from open_mythos.main import OpenMythos, MythosConfig
from train import VARIANTS, TokenDataset


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_latest_checkpoint(model: OpenMythos, ckpt_dir: str) -> int:
    ckpts = sorted(Path(ckpt_dir).glob("step_*.npz"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    latest = str(ckpts[-1])
    model.load_weights(latest)
    mx.eval(model.parameters())
    step = int(ckpts[-1].stem.split("_")[1])
    logger.info(f"Loaded checkpoint: {latest} (step {step})")
    return step


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity(
    model: OpenMythos,
    dataset: TokenDataset,
    n_loops: int,
    n_batches: int = 50,
    batch_size: int = 4,
) -> float:
    """Estimate perplexity over random batches from the dataset."""
    rng = np.random.default_rng(0)
    total_loss = 0.0
    total_tokens = 0

    logger.info(f"Computing perplexity over {n_batches} batches (batch={batch_size})...")
    t0 = time.time()

    for i in range(n_batches):
        indices = rng.integers(0, len(dataset), size=batch_size)
        x, y = dataset.get_batch(indices)

        logits = model(x, n_loops=n_loops)          # (B, T, V)
        B, T, V = logits.shape
        loss = mx.mean(
            nn.losses.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))
        )
        mx.eval(loss)
        total_loss += loss.item() * B * T
        total_tokens += B * T

        if (i + 1) % 10 == 0:
            logger.info(f"  batch {i+1}/{n_batches} | running ppl: {math.exp(total_loss / total_tokens):.2f}")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    elapsed = time.time() - t0
    logger.info(f"Perplexity: {ppl:.2f} (avg loss: {avg_loss:.4f}) [{elapsed:.1f}s]")
    return ppl


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate_samples(
    model: OpenMythos,
    tokenizer,
    prompts: list[str],
    n_loops: int,
    max_new_tokens: int,
    temperature: float,
) -> None:
    logger.info(f"Generating samples (n_loops={n_loops}, max_new_tokens={max_new_tokens}, temp={temperature})")
    print()

    for prompt in prompts:
        print(f"{'─' * 60}")
        print(f"PROMPT: {prompt!r}")
        print()

        input_ids = tokenizer.encode(prompt)
        tokens = mx.array([input_ids], dtype=mx.uint32)

        t0 = time.time()
        # Token bias: penalize repeated whitespace tokens to encourage code content.
        # GPT-2 token 220 = ' ' (space), 197 = '\t', 198 = '\n'
        SPACE_TOKENS = [220, 197]
        SPACE_PENALTY = 5.0  # logit penalty applied when previous token is also whitespace

        for step_i in range(max_new_tokens):
            logits = model(tokens, n_loops=n_loops)
            next_logits = logits[:, -1, :].astype(mx.float32)

            # Apply space-repeat penalty: suppress pure whitespace runs
            if step_i > 0:
                prev_token = int(tokens[0, -1].item())
                if prev_token in SPACE_TOKENS:
                    penalty = mx.zeros_like(next_logits)
                    for sp in SPACE_TOKENS:
                        # Build index tensor and scatter penalty
                        idx = mx.array([[sp]])
                        penalty = penalty.at[0:1, sp:sp+1].add(-SPACE_PENALTY)
                    next_logits = next_logits + penalty

            if temperature > 0:
                next_logits = next_logits / temperature
                probs = mx.softmax(next_logits, axis=-1)
                # Top-p (nucleus) sampling with p=0.9
                sorted_indices = mx.argsort(probs, axis=-1)[:, ::-1]
                sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
                cumsum = mx.cumsum(sorted_probs, axis=-1)
                # Mask tokens beyond cumulative probability 0.9
                mask = (cumsum - sorted_probs) < 0.9
                filtered_probs = mx.where(mask, sorted_probs, mx.zeros_like(sorted_probs))
                # Sample from filtered distribution
                filtered_sum = mx.sum(filtered_probs, axis=-1, keepdims=True)
                normalized = filtered_probs / (filtered_sum + 1e-8)
                # Multinomial sampling via Gumbel-max trick
                gumbel = -mx.log(-mx.log(mx.random.uniform(shape=normalized.shape) + 1e-10) + 1e-10)
                sample_idx = mx.argmax(mx.log(normalized + 1e-10) + gumbel, axis=-1, keepdims=True)
                next_token = mx.take_along_axis(sorted_indices, sample_idx, axis=-1)
            else:
                next_token = mx.argmax(next_logits, axis=-1, keepdims=True)
            tokens = mx.concatenate([tokens, next_token], axis=1)
            mx.eval(tokens)
            if int(next_token.item()) == (tokenizer.eos_token_id or 50256):
                break

        elapsed = time.time() - t0
        generated_ids = tokens[0].tolist()
        text = tokenizer.decode(generated_ids)
        tps = max_new_tokens / elapsed

        print(f"OUTPUT:\n{text}")
        print(f"\n[{max_new_tokens} tokens, {tps:.1f} tok/s]")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenMythos Evaluation")
    p.add_argument("--checkpoint", required=True,        help="Checkpoint directory")
    p.add_argument("--variant",    default="1b",         choices=list(VARIANTS))
    p.add_argument("--data",       default=None,         help="Path to .npy token file for perplexity")
    p.add_argument("--tokenizer",  default="gpt2",       help="HF tokenizer name")
    p.add_argument("--n_loops",    type=int, default=4,  help="Recurrent loops")
    p.add_argument("--ppl_batches", type=int, default=50, help="Batches for perplexity estimate")
    p.add_argument("--batch_size",  type=int, default=4,  help="Batch size for perplexity")
    p.add_argument("--prompt",     type=str, default=None,
                   help="Text prompt for generation (comma-separated for multiple)")
    p.add_argument("--max_new_tokens", type=int, default=128, help="Tokens to generate per prompt")
    p.add_argument("--temperature",    type=float, default=0.8, help="Sampling temperature (0=greedy)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = VARIANTS[args.variant]
    model = OpenMythos(cfg)
    step = load_latest_checkpoint(model, args.checkpoint)
    logger.info(f"Model: {args.variant} | dim={cfg.dim} | step={step}")

    # --- Perplexity ---
    if args.data and Path(args.data).exists():
        tokens = np.load(args.data)
        dataset = TokenDataset(tokens, cfg.max_seq_len)
        logger.info(f"Dataset: {args.data} ({len(dataset):,} chunks)")
        ppl = compute_perplexity(model, dataset, args.n_loops, args.ppl_batches, args.batch_size)
        print(f"\n{'='*60}")
        print(f"  Perplexity : {ppl:.2f}")
        print(f"  Step       : {step:,}")
        print(f"  Variant    : {args.variant}")
        print(f"{'='*60}\n")
    else:
        logger.info("No --data provided, skipping perplexity.")

    # --- Generation ---
    if args.prompt:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        prompts = [p.strip() for p in args.prompt.split("|||")]
        generate_samples(model, tokenizer, prompts, args.n_loops, args.max_new_tokens, args.temperature)
    else:
        logger.info("No --prompt provided, skipping generation. Use --prompt 'text' to generate.")


if __name__ == "__main__":
    main()
