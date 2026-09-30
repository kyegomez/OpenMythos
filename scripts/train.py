#!/usr/bin/env python3
"""
Consolidated OpenMythos / Sentinel Grid trainer.

Replaces the ad-hoc train_fixed.py / train_100bt.py / train_overnight.py /
train_massive.py / train_extended.py / train_tools.py / train_final.py /
train_mythos1b.py / train_mps.py / resume_train.py scripts (now in archive/)
with one config-driven script. Recipe (grad-accum, clip, AdamW, cosine
schedule, checkpoint dict shape) is carried over unchanged from
train_massive.py, the proven lineage that produced the current production
checkpoint.

Examples:
  # Continue the proven cyber lineage on a bigger/merged corpus
  python3 scripts/train.py --resume-from overnight_final.pt \\
      --corpus merged_corpus.jsonl --output-name sentinel_v2 --epochs 3

  # Fresh run from scratch
  python3 scripts/train.py --resume-from none --corpus cyber_corpus.jsonl \\
      --output-name from_scratch
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths  # noqa: E402  (must run before transformers import, sets HF_HOME)

import torch  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.nn import CrossEntropyLoss  # noqa: E402
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup  # noqa: E402

from open_mythos.main import OpenMythos, MythosConfig  # noqa: E402
from open_mythos import variants as _variants  # noqa: E402


# ── Config presets ──────────────────────────────────────────────────────
# "sentinel_gqa" is the exact architecture used by every proven checkpoint
# in the current lineage (overnight_final.pt -> massive_best.pt etc). Keep
# this as the default so resuming from those checkpoints works unmodified.
# Other names delegate to open_mythos/variants.py; vocab_size is always
# overridden from the tokenizer regardless of preset (variants.py hardcodes
# vocab_size=32000, which is the mismatch bug earlier scripts had to patch
# around with clamp() hacks -- always deriving it from the tokenizer avoids
# resurrecting that).
def build_config(preset: str, vocab_size: int) -> MythosConfig:
    if preset == "sentinel_gqa":
        cfg = MythosConfig(
            vocab_size=vocab_size, dim=2048, n_heads=16, n_kv_heads=4,
            max_seq_len=256, max_loop_iters=8, prelude_layers=2, coda_layers=2,
            attn_type="gqa", n_experts=16, n_shared_experts=1,
            n_experts_per_tok=2, expert_dim=512, lora_rank=16,
            act_threshold=0.99,
        )
    else:
        variant_fn = getattr(_variants, preset, None)
        if variant_fn is None:
            raise ValueError(
                f"Unknown config preset '{preset}'. Use 'sentinel_gqa' or one "
                f"of the open_mythos.variants functions (mythos_1b, mythos_3b, ...)."
            )
        cfg = variant_fn()
        cfg.vocab_size = vocab_size
    return cfg


def resolve_path(base_dir: str, name_or_path: str) -> str:
    """A bare filename resolves inside base_dir; an absolute/relative path with a
    separator is used as-is."""
    if os.sep in name_or_path or name_or_path == "none":
        return name_or_path
    return os.path.join(base_dir, name_or_path)


def load_corpus(corpus_path: str, min_len: int = 100):
    texts = []
    with open(corpus_path) as f:
        for line in f:
            t = json.loads(line)["text"]
            if len(t.strip()) > min_len:
                texts.append(t)
    return texts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-preset", default="sentinel_gqa",
                    help="'sentinel_gqa' (default, matches existing checkpoints) or a name from open_mythos/variants.py")
    p.add_argument("--corpus", required=True,
                    help="Corpus filename (resolved inside CORPUS_DIR) or full path to a .jsonl file")
    p.add_argument("--resume-from", default="none",
                    help="Checkpoint filename (resolved inside CHECKPOINT_DIR), full path, or 'none' for a fresh model")
    p.add_argument("--output-name", required=True,
                    help="Base name for output checkpoints, e.g. 'sentinel_v2' -> sentinel_v2_best.pt / _last.pt / _step_N.pt")
    p.add_argument("--epochs", type=float, default=3.0, help="Target epochs over the training split")
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--warmup-steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=1, help="Per-step batch size (kept at 1; scale via --grad-accum)")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--n-loops", type=int, default=4, help="Recurrent loop count used during training")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--val-min", type=int, default=400, help="Minimum held-out validation set size")
    p.add_argument("--val-samples", type=int, default=60, help="Number of val examples sampled per validation check")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--patience", type=int, default=6, help="Early-stop after this many non-improving val checks")
    p.add_argument("--time-budget-hours", type=float, default=None,
                    help="Stop after roughly this many hours (checked once per step, after each optimizer step), saving a final checkpoint before exiting. Use this instead of babysitting/killing a long run.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    cfg = build_config(args.config_preset, vocab_size)

    resume_from = resolve_path(paths.CHECKPOINT_DIR, args.resume_from)
    if resume_from != "none":
        print(f"Loading: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model = OpenMythos(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        resumed_step = ckpt["step"]
        tokens_seen = ckpt.get("tokens_seen", 0)
        print(f"Resumed from step {resumed_step}")
    else:
        model = OpenMythos(cfg).to(device)
        resumed_step = 0
        tokens_seen = 0
        print("Starting from a freshly initialized model")

    corpus_path = resolve_path(paths.CORPUS_DIR, args.corpus)
    all_texts = load_corpus(corpus_path)
    random.seed(args.seed)
    random.shuffle(all_texts)
    val_size = max(args.val_min, int(len(all_texts) * args.val_frac))
    val_size = max(1, min(val_size, len(all_texts) - 1))
    val_texts = all_texts[:val_size]
    train_texts = all_texts[val_size:]
    print(f"Corpus: {corpus_path}")
    print(f"Total entries: {len(all_texts)} | Train: {len(train_texts)} | Val (held out): {len(val_texts)}")

    max_seq_len = cfg.max_seq_len

    def encode(text):
        return tokenizer(
            text, max_length=max_seq_len + 1, truncation=True,
            padding="max_length", return_tensors="pt",
        )["input_ids"].squeeze(0)

    steps_per_epoch = max(1, len(train_texts) // (args.batch_size * args.grad_accum))
    max_steps = resumed_step + int(steps_per_epoch * args.epochs)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=max(1, max_steps - resumed_step),
    )
    loss_fn = CrossEntropyLoss()

    print(f"\nTarget: {args.epochs} epochs = {max_steps - resumed_step} steps")
    print(f"Steps: {resumed_step} -> {max_steps}")
    print(f"LR: {args.lr} | Validation every {args.val_every} steps")
    print(f"Output: {paths.CHECKPOINT_DIR}/{args.output_name}_{{best,last,step_N}}.pt\n")

    def run_validation():
        model.eval()
        n = min(args.val_samples, len(val_texts))
        sample = random.sample(val_texts, n)
        val_loss = 0.0
        with torch.no_grad():
            for text in sample:
                tokens = encode(text).unsqueeze(0).to(device)
                x, y = tokens[:, :-1], tokens[:, 1:]
                logits = model(x, n_loops=args.n_loops)
                loss = loss_fn(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
                val_loss += loss.item()
        model.train()
        return val_loss / n

    def save_checkpoint(name, extra=None, retries=3, retry_delay=5.0):
        """Write to a .tmp path and atomically rename on success, retrying on
        failure. Guards against the exact failure mode seen in practice: a
        transient volume/network drop truncating torch.save mid-write and
        leaving a corrupt file at the final checkpoint name."""
        payload = {
            "step": step, "tokens_seen": tokens_seen,
            "vocab_size": vocab_size, "config": cfg.__dict__,
            "model": model.state_dict(), "optim": optimizer.state_dict(),
        }
        if extra:
            payload.update(extra)
        final_path = os.path.join(paths.CHECKPOINT_DIR, name)
        tmp_path = final_path + ".tmp"
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                torch.save(payload, tmp_path)
                os.replace(tmp_path, final_path)
                return
            except OSError as e:
                last_err = e
                print(f"  ! checkpoint write failed (attempt {attempt}/{retries}): {e}")
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                if attempt < retries:
                    time.sleep(retry_delay)
        print(f"  ! giving up on checkpoint '{name}' after {retries} attempts ({last_err}) -- continuing training, not saved")

    model.train()
    step = resumed_step
    accum_step = 0
    total_loss = 0.0
    accum_loss = 0.0
    start_time = time.time()
    best_val = float("inf")
    patience = 0

    train_pool = train_texts.copy()
    random.shuffle(train_pool)
    pool_idx = 0

    while step < max_steps:
        if pool_idx >= len(train_pool):
            random.shuffle(train_pool)
            pool_idx = 0
        text = train_pool[pool_idx]
        pool_idx += 1

        tokens = encode(text).unsqueeze(0).to(device)
        x, y = tokens[:, :-1], tokens[:, 1:]

        logits = model(x, n_loops=args.n_loops)
        loss = loss_fn(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        loss = loss / args.grad_accum
        loss.backward()

        accum_loss += loss.item()
        accum_step += 1
        if accum_step < args.grad_accum:
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += accum_loss
        tokens_seen += args.grad_accum * max_seq_len
        accum_loss = 0.0
        accum_step = 0
        step += 1

        if step % args.log_every == 0:
            avg = total_loss / args.log_every
            elapsed = time.time() - start_time
            sps = (step - resumed_step) / elapsed if elapsed > 0 else 0
            eta = (max_steps - step) / sps if sps > 0 else 0
            lr_now = scheduler.get_last_lr()[0]
            print(f"Step {step:>6} | train: {avg:.4f} | {sps:.2f} sps | LR: {lr_now:.2e} | ETA: {eta/3600:.2f}h")
            total_loss = 0.0

        if step % args.val_every == 0:
            vl = run_validation()
            if vl < best_val:
                best_val = vl
                patience = 0
                save_checkpoint(f"{args.output_name}_best.pt", {"val_loss": vl})
                flag = " <-- new best, saved"
            else:
                patience += 1
                flag = f" <-- no improvement ({patience}/{args.patience})"
            print(f"  VAL @ step {step}: {vl:.4f} (best: {best_val:.4f}){flag}")
            if patience >= args.patience:
                print(f"\nEarly stopping -- no val improvement for {args.patience} checks")
                break

        if step % args.save_every == 0:
            save_checkpoint(f"{args.output_name}_step_{step}.pt")
            print(f"  -> Checkpoint saved at step {step}")

        if args.time_budget_hours is not None and (time.time() - start_time) / 3600 >= args.time_budget_hours:
            print(f"\nTime budget of {args.time_budget_hours}h reached at step {step} -- stopping")
            break

    save_checkpoint(f"{args.output_name}_last.pt")
    elapsed = time.time() - start_time
    print(f"\nDone. Last checkpoint -> {paths.CHECKPOINT_DIR}/{args.output_name}_last.pt")
    print(f"Best val checkpoint -> {paths.CHECKPOINT_DIR}/{args.output_name}_best.pt (val_loss: {best_val:.4f})")
    print(f"Steps: {step} | Tokens: {tokens_seen/1e6:.1f}M | Time: {elapsed/3600:.2f}h")


if __name__ == "__main__":
    main()
