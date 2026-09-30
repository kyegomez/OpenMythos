#!/usr/bin/env python3
"""
Phase 4 pilot: compare MoDAModel (open_mythos/moda.py -- Mixture-of-Depths
Attention + DeepSeek MoE, never trained before) against the proven OpenMythos
architecture (open_mythos/main.py) at roughly equivalent scale.

This is a validation pass, not a real training run: both models start from
fresh random init (MoDAModel has no compatible checkpoint to resume from) and
train for a short, fixed number of steps on a small corpus slice, purely to
compare loss-curve stability and per-step time on this machine's MPS backend.
Not a claim about final model quality either architecture could reach.

Usage:
  python3 scripts/moda_pilot.py --corpus scratch_pilot_corpus.jsonl --steps 150
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths  # noqa: E402

import torch  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from open_mythos.main import OpenMythos, MythosConfig  # noqa: E402
from open_mythos.moda import MoDAModel, MoDAConfig  # noqa: E402


def load_corpus(path, min_len=100):
    texts = []
    with open(path) as f:
        for line in f:
            t = json.loads(line)["text"]
            if len(t.strip()) > min_len:
                texts.append(t)
    return texts


def run_pilot(name, model, encode_fn, forward_fn, texts, steps, lr, device):
    model.to(device)
    model.train()
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses = []
    step_times = []
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {name} ({n_params:,} params) ===")

    for step in range(1, steps + 1):
        text = texts[step % len(texts)]
        tokens = encode_fn(text).unsqueeze(0).to(device)
        x, y = tokens[:, :-1], tokens[:, 1:]

        t0 = time.time()
        loss = forward_fn(model, x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        t1 = time.time()

        losses.append(loss.item())
        step_times.append(t1 - t0)

        if step % 25 == 0 or step == 1:
            recent = losses[-25:]
            print(f"  step {step:>4} | loss: {sum(recent)/len(recent):.4f} | "
                  f"avg step time: {sum(step_times[-25:])/len(step_times[-25:]):.3f}s")

    first_avg = sum(losses[:10]) / min(10, len(losses))
    last_avg = sum(losses[-10:]) / min(10, len(losses))
    diverged = any(l != l or l > 50 for l in losses)  # NaN or blown-up loss
    return {
        "name": name, "n_params": n_params,
        "first_loss": first_avg, "last_loss": last_avg,
        "avg_step_time": sum(step_times) / len(step_times),
        "diverged": diverged,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True, help="Corpus filename (resolved inside CORPUS_DIR) or full path")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    corpus_path = args.corpus if "/" in args.corpus else f"{paths.CORPUS_DIR}/{args.corpus}"
    texts = load_corpus(corpus_path)
    print(f"Corpus: {corpus_path} ({len(texts)} entries)")

    max_seq_len = 256

    def encode(text):
        return tokenizer(
            text, max_length=max_seq_len + 1, truncation=True,
            padding="max_length", return_tensors="pt",
        )["input_ids"].squeeze(0)

    results = []

    # --- OpenMythos (main.py), same architecture family as production ---
    torch.manual_seed(args.seed)
    mythos_cfg = MythosConfig(
        vocab_size=vocab_size, dim=2048, n_heads=16, n_kv_heads=4,
        max_seq_len=max_seq_len, max_loop_iters=8, prelude_layers=2, coda_layers=2,
        attn_type="gqa", n_experts=16, n_shared_experts=1, n_experts_per_tok=2,
        expert_dim=512, lora_rank=16, act_threshold=0.99,
    )
    mythos_model = OpenMythos(mythos_cfg)
    loss_fn = torch.nn.CrossEntropyLoss()

    def mythos_forward(model, x, y):
        logits = model(x, n_loops=4)
        return loss_fn(logits.reshape(-1, mythos_cfg.vocab_size), y.reshape(-1))

    results.append(run_pilot("OpenMythos (main.py)", mythos_model, encode, mythos_forward, texts, args.steps, args.lr, device))
    del mythos_model
    if device.type == "mps":
        torch.mps.empty_cache()

    # --- MoDAModel (moda.py), roughly equivalent scale ---
    torch.manual_seed(args.seed)
    moda_cfg = MoDAConfig(
        vocab_size=vocab_size, d_model=2048, n_layers=4, n_heads_q=16, n_heads_kv=4,
        head_dim=128, max_seq_len=max_seq_len,
        n_shared_experts=1, n_routed_experts=16, n_activated_experts=2,
        expert_hidden_dim=512, moe_balance_alpha=0.001, moe_score_func="softmax",
    )
    moda_model = MoDAModel(moda_cfg)

    def moda_forward(model, x, y):
        _, loss = model(x, labels=y)
        return loss

    results.append(run_pilot("MoDAModel (moda.py)", moda_model, encode, moda_forward, texts, args.steps, args.lr, device))

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for r in results:
        status = "DIVERGED" if r["diverged"] else "stable"
        print(f"  {r['name']:<24} params: {r['n_params']:>13,} | "
              f"loss {r['first_loss']:.2f} -> {r['last_loss']:.2f} | "
              f"avg step: {r['avg_step_time']:.3f}s | {status}")


if __name__ == "__main__":
    main()
