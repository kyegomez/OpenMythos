#!/usr/bin/env python3
"""
Tiny text model for OpenMythos: does the looped architecture learn to write?

Trains a very small OpenMythos on plain text, one byte per token, on a CPU.
The point is not a useful language model. It is to watch the real architecture
(Prelude -> looped Recurrent Block with MoE, LoRA depth adapters and ACT
halting -> Coda) learn real text, sample from it, and see what changing the
loop count at inference does.

Quick start (CPU, about 15 minutes):

    python examples/tiny_text.py --preset tiny --minutes 15

Longer run (a few hours, better text):

    python examples/tiny_text.py --preset small --minutes 180

Sample from a saved run, with a different loop count than it trained on:

    python examples/tiny_text.py --generate runs/tiny_text --prompt "ROMEO:" --loops 8

Data: any UTF-8 text file via --data. With no --data, the script downloads
the 1 MB "tiny Shakespeare" file once and keeps it under the output folder.
Tokens are raw bytes (vocabulary of 256), so there is no tokenizer to install.

Everything is written under --out (default runs/tiny_text, which the repo's
.gitignore already excludes).
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.request

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_mythos import MythosConfig, OpenMythos  # noqa: E402

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
VOCAB = 256  # one token per byte

# Sizes are picked for CPU training. Measured on a 10-thread desktop CPU:
# "tiny" (0.6M parameters) runs about 1.7 steps/s, "small" (2.4M) about
# 0.6 steps/s. A 4-core VM is roughly 2-3 times slower, so use --minutes.
PRESETS = {
    "tiny": dict(dim=128, heads=4, kv_heads=2, prelude=1, coda=1, experts=4, shared_experts=1,
                 experts_per_tok=2, expert_dim=128, lora_rank=8, seq_len=128, batch=32,
                 lr=1e-3, warmup=200, steps=3000),
    "small": dict(dim=256, heads=8, kv_heads=4, prelude=1, coda=1, experts=4, shared_experts=1,
                  experts_per_tok=2, expert_dim=256, lora_rank=16, seq_len=192, batch=24,
                  lr=6e-4, warmup=300, steps=6000),
}
OVERRIDABLE = ("seq_len", "batch", "lr", "warmup", "steps")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    ap.add_argument("--data", help="UTF-8 text file to train on (default: download tiny Shakespeare)")
    ap.add_argument("--out", default="runs/tiny_text", help="folder for the model, log and downloaded data")
    ap.add_argument("--train-loops", type=int, default=4, help="recurrent loop count used in training")
    ap.add_argument("--minutes", type=float, default=0,
                    help="stop after this many minutes even if --steps is not reached (0 = no limit)")
    ap.add_argument("--steps", type=int, help="training steps (default: from the preset)")
    ap.add_argument("--batch", type=int, help="sequences per step")
    ap.add_argument("--seq-len", type=int, help="bytes of context per sequence")
    ap.add_argument("--lr", type=float, help="peak learning rate")
    ap.add_argument("--warmup", type=int, help="linear warmup steps")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--act-final-remainder", action="store_true",
                    help="give unused halting mass to the last loop (see MythosConfig)")
    ap.add_argument("--eval-every", type=int, default=250, help="steps between validation checks")
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--sample-every", type=int, default=500, help="steps between printed samples (0 = off)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = PyTorch default)")
    ap.add_argument("--seed", type=int, default=0)
    g = ap.add_argument_group("sampling")
    g.add_argument("--generate", metavar="DIR", help="skip training: load the model in DIR and write text")
    g.add_argument("--prompt", default="\n", help="text the model continues")
    g.add_argument("--tokens", type=int, default=400, help="bytes to generate")
    g.add_argument("--loops", type=int, help="loop count for sampling (default: the training value)")
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=40)
    return ap


def load_bytes(args):
    path = args.data
    if not path:
        path = os.path.join(args.out, "data", "tinyshakespeare.txt")
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"downloading tiny Shakespeare to {path}")
            with urllib.request.urlopen(DATA_URL, timeout=60) as resp, open(path, "wb") as fh:
                fh.write(resp.read())
    with open(path, "rb") as fh:
        raw = fh.read()
    if len(raw) < 20_000:
        sys.exit(f"{path} is too small to train on ({len(raw)} bytes); use at least a few hundred KB")
    data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
    cut = int(len(data) * 0.9)
    return path, data[:cut], data[cut:]


def get_batch(data, batch, seq_len, gen):
    starts = torch.randint(0, len(data) - seq_len - 1, (batch,), generator=gen)
    x = torch.stack([data[s:s + seq_len] for s in starts])
    y = torch.stack([data[s + 1:s + seq_len + 1] for s in starts])
    return x, y


def make_config(args, p):
    return MythosConfig(
        vocab_size=VOCAB,
        dim=p["dim"],
        n_heads=p["heads"],
        n_kv_heads=p["kv_heads"],
        max_seq_len=p["seq_len"],
        max_loop_iters=args.train_loops,
        prelude_layers=p["prelude"],
        coda_layers=p["coda"],
        attn_type="gqa",
        n_experts=p["experts"],
        n_shared_experts=p["shared_experts"],
        n_experts_per_tok=p["experts_per_tok"],
        expert_dim=p["expert_dim"],
        lora_rank=p["lora_rank"],
        act_final_remainder=args.act_final_remainder,
        dropout=args.dropout,
    )


@torch.no_grad()
def estimate_loss(model, data, batch, seq_len, loops, n_batches, gen):
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        x, y = get_batch(data, batch, seq_len, gen)
        logits = model(x, n_loops=loops)
        total += F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1)).item()
    model.train()
    return total / n_batches


@torch.no_grad()
def sample(model, prompt, tokens, loops, seq_len, temperature, top_k, gen):
    """Generate bytes after `prompt`.

    The model only ever saw seq_len bytes of context, so each step feeds the
    last seq_len bytes again instead of using generate()'s KV cache. Slower,
    but it can write text of any length.
    """
    model.eval()
    ids = torch.tensor([list(prompt.encode("utf-8")) or [10]], dtype=torch.long)
    for _ in range(tokens):
        logits = model(ids[:, -seq_len:], n_loops=loops)[:, -1, :] / max(temperature, 1e-6)
        if top_k > 0:
            kth = logits.topk(min(top_k, VOCAB)).values[:, -1:]
            logits[logits < kth] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=gen)
        ids = torch.cat([ids, nxt], dim=1)
    model.train()
    return bytes(ids[0].tolist()).decode("utf-8", errors="replace")


def lr_at(step, steps, warmup, peak, time_used=0.0):
    """Linear warmup, then cosine decay to 5% of the peak.

    time_used is the fraction of a --minutes budget already spent; the decay
    follows whichever is further along, steps or time, so a run that will be
    cut off by the clock still finishes at a low learning rate.
    """
    if step <= warmup:
        return peak * step / max(warmup, 1)
    progress = max((step - warmup) / max(steps - warmup, 1), time_used)
    return peak * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


def run_generate(args):
    meta_path = os.path.join(args.generate, "tiny_text.json")
    if not os.path.exists(meta_path):
        sys.exit(f"no tiny_text.json in {args.generate}; train a model first")
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    model = OpenMythos.from_pretrained(args.generate)
    loops = args.loops or meta["train_loops"]
    gen = torch.Generator().manual_seed(args.seed)
    print(f"model: {model.num_parameters():,} parameters, trained with {meta['train_loops']} loops, "
          f"validation loss {meta.get('val_loss', float('nan')):.3f}; sampling with {loops} loops\n")
    print(sample(model, args.prompt, args.tokens, loops, meta["seq_len"],
                 args.temperature, args.top_k, gen))


def main():
    args = build_parser().parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.generate:
        run_generate(args)
        return

    p = dict(PRESETS[args.preset])
    for key in OVERRIDABLE:
        if getattr(args, key) is not None:
            p[key] = getattr(args, key)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    path, train, val = load_bytes(args)
    model = OpenMythos(make_config(args, p))
    os.makedirs(args.out, exist_ok=True)
    print(f"data: {path}  ({len(train):,} training bytes, {len(val):,} validation bytes)")
    print(f"model: preset {args.preset}, {model.num_parameters():,} parameters, "
          f"{args.train_loops} loops, context {p['seq_len']} bytes, {torch.get_num_threads()} CPU threads")
    print(f"plan: up to {p['steps']} steps of {p['batch']} sequences"
          + (f", or {args.minutes:g} minutes" if args.minutes else "")
          + f"; a model that guesses bytes uniformly scores {math.log(VOCAB):.2f}\n")

    decay = [q for q in model.parameters() if q.requires_grad and q.dim() >= 2]
    no_decay = [q for q in model.parameters() if q.requires_grad and q.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=p["lr"])
    params = decay + no_decay

    def save(val_loss, step):
        model.save_pretrained(args.out)
        with open(os.path.join(args.out, "tiny_text.json"), "w", encoding="utf-8") as fh:
            json.dump({"preset": args.preset, "seq_len": p["seq_len"], "train_loops": args.train_loops,
                       "step": step, "val_loss": val_loss, "data": os.path.basename(path)}, fh, indent=2)

    log_path = os.path.join(args.out, "log.jsonl")
    log = open(log_path, "w", encoding="utf-8")
    best, step, t0 = float("inf"), 0, time.time()
    deadline = t0 + args.minutes * 60 if args.minutes else None
    model.train()
    try:
        for step in range(1, p["steps"] + 1):
            used = (time.time() - t0) / (args.minutes * 60) if args.minutes else 0.0
            for group in opt.param_groups:
                group["lr"] = lr_at(step, p["steps"], p["warmup"], p["lr"], used)
            x, y = get_batch(train, p["batch"], p["seq_len"], gen)
            logits = model(x, n_loops=args.train_loops)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            model.update_router_bias()  # aux-loss-free MoE load balancing

            if step % args.log_every == 0 or step == 1:
                rate = step / (time.time() - t0)
                print(f"step {step:>6}  loss {loss.item():.3f}  lr {opt.param_groups[0]['lr']:.1e}  "
                      f"{rate:.2f} steps/s  {(time.time() - t0) / 60:.1f} min")
            out_of_time = deadline is not None and time.time() >= deadline
            last = step == p["steps"] or out_of_time
            if step % args.eval_every == 0 or last:
                val_loss = estimate_loss(model, val, p["batch"], p["seq_len"], args.train_loops,
                                         args.eval_batches, gen)
                mark = ""
                if val_loss < best:
                    best = val_loss
                    save(val_loss, step)
                    mark = "  (best so far, saved)"
                print(f"  validation loss {val_loss:.3f}{mark}")
                log.write(json.dumps({"step": step, "train_loss": loss.item(), "val_loss": val_loss,
                                      "minutes": (time.time() - t0) / 60}) + "\n")
                log.flush()
            if args.sample_every and (step % args.sample_every == 0 or last):
                text = sample(model, "\n", 200, args.train_loops, p["seq_len"], 0.8, 40, gen)
                print("  ---- sample ----\n  " + text.strip().replace("\n", "\n  ") + "\n  ----------------")
            if out_of_time:
                print(f"\ntime limit of {args.minutes:g} minutes reached at step {step}")
                break
    except KeyboardInterrupt:
        print(f"\nstopped by Ctrl+C at step {step}; the best model so far is saved")
    finally:
        log.close()

    if best == float("inf"):
        print("stopped before the first validation check, nothing saved")
        return
    print(f"\ntraining time {(time.time() - t0) / 60:.1f} min, best validation loss {best:.3f}")

    # The architecture's claim is that loop count is a dial. Show what it does
    # on text: same saved weights, validation loss at other loop counts.
    model = OpenMythos.from_pretrained(args.out)
    print("\nvalidation loss of the saved model at different loop counts "
          f"(it trained with {args.train_loops}):")
    for loops in sorted({1, 2, args.train_loops, args.train_loops * 2, args.train_loops * 4}):
        vl = estimate_loss(model, val, p["batch"], p["seq_len"], loops, args.eval_batches,
                           torch.Generator().manual_seed(1))
        print(f"  {loops:>3} loops: {vl:.3f}" + ("   <- training value" if loops == args.train_loops else ""))
    print(f"\nmodel saved in {args.out}. To write more text:\n"
          f"  python examples/tiny_text.py --generate {args.out} --prompt \"ROMEO:\"")


if __name__ == "__main__":
    main()
