#!/usr/bin/env python3
"""
Depth-extrapolation demo for OpenMythos.

The whole thesis of a Recurrent-Depth Transformer is that *looping the same
weights more times buys more reasoning*. This script tries to actually show
that on a task that provably needs sequential compute, rather than just
asserting it in a docstring.

Task: iterated pointer chasing (compute P^k(s))
-----------------------------------------------
Each example gives the model a fresh random permutation P of M states,
written out as a sequence, followed by a start state s and a hop count k:

    [ P[0] P[1] ... P[M-1] | SEP | s | k ]   ->  predict  P^k(s)

Because P is different every example, the model cannot memorize answers. To
produce P^k(s) it must read P with attention and then apply it k times in
sequence. Applying a function k times is exactly the kind of work that extra
loop iterations are supposed to provide.

What the script measures
------------------------
1. The model trains at a FIXED loop count and should solve small k.
2. At inference we sweep n_loops and print accuracy per hop count k, including
   k values never seen in training. If recurrent depth is doing real work,
   accuracy should rise with more loops, and extra loops should help the
   unseen (harder) k the most.

The printed numbers are whatever actually happens on this run (seeded). No
result is hard-coded. Runs on CPU in a couple of minutes.

    python examples/depth_extrapolation.py
"""

import argparse
import tempfile

import torch
import torch.nn.functional as F

from open_mythos.main import MythosConfig, OpenMythos


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class PointerChase:
    """Generator for the P^k(s) pointer-chasing task.

    Vocabulary layout:
        ids 0 .. M-1          -- state / value tokens (also the answer space)
        id  M                 -- SEP
        ids M+1 .. M+k_max     -- hop-count tokens (k encoded as id M+k)
    """

    def __init__(self, n_states: int, k_max: int):
        self.M = n_states
        self.k_max = k_max
        self.sep = n_states
        self.vocab_size = n_states + 1 + k_max
        self.seq_len = n_states + 3  # P (M) + SEP + s + k

    def k_token(self, k: int) -> int:
        return self.M + k  # k in 1..k_max -> id M+1..M+k_max

    def batch(self, batch_size: int, k_values, generator: torch.Generator):
        """Return (input_ids [B, L], target [B]) sampling k uniformly from k_values."""
        M = self.M
        inputs = torch.empty(batch_size, self.seq_len, dtype=torch.long)
        targets = torch.empty(batch_size, dtype=torch.long)
        kv = torch.tensor(k_values)
        for b in range(batch_size):
            P = torch.randperm(M, generator=generator)
            s = int(torch.randint(0, M, (1,), generator=generator))
            k = int(kv[torch.randint(0, len(kv), (1,), generator=generator)])
            # apply P k times
            cur = s
            for _ in range(k):
                cur = int(P[cur])
            inputs[b, :M] = P
            inputs[b, M] = self.sep
            inputs[b, M + 1] = s
            inputs[b, M + 2] = self.k_token(k)
            targets[b] = cur
        return inputs, targets


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def accuracy(model, task, k, n_loops, n_samples, gen):
    """Fraction of P^k(s) predictions that are correct, over n_samples examples."""
    model.eval()
    correct = 0
    seen = 0
    with torch.no_grad():
        while seen < n_samples:
            bs = min(256, n_samples - seen)
            x, y = task.batch(bs, [k], gen)
            logits = model(x, n_loops=n_loops)[:, -1, : task.M]
            correct += (logits.argmax(-1) == y).sum().item()
            seen += bs
    return correct / n_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=6, help="number of states M")
    ap.add_argument("--k-train", type=int, default=3, help="max hops seen in training")
    ap.add_argument("--k-test", type=int, default=5, help="max hops at evaluation")
    ap.add_argument("--train-loops", type=int, default=6, help="loop count during training")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    task = PointerChase(args.states, args.k_test)
    cfg = MythosConfig(
        vocab_size=task.vocab_size,
        dim=96,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=task.seq_len + 2,
        max_loop_iters=args.train_loops,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=64,
        lora_rank=8,
        act_threshold=0.999,
    )
    model = OpenMythos(cfg)
    print(
        f"Task: P^k(s) with M={args.states} states, "
        f"train k in 1..{args.k_train}, test k in 1..{args.k_test}"
    )
    print(f"Model: {model.num_parameters():,} params, trained at n_loops={args.train_loops}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    train_ks = list(range(1, args.k_train + 1))

    model.train()
    for step in range(1, args.steps + 1):
        x, y = task.batch(args.batch, train_ks, gen)
        logits = model(x, n_loops=args.train_loops)[:, -1, :]
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        model.update_router_bias()  # aux-loss-free MoE load balancing
        if step % 500 == 0 or step == 1:
            print(f"step {step:5d}  loss {loss.item():.4f}")

    # ---- Evaluation: accuracy as a function of loop count ----
    print("\nAccuracy (%) by hop count k and inference loop count:")
    loop_sweep = [1, 2, args.train_loops, args.train_loops * 2]
    header = "  k \\ loops | " + " | ".join(f"{n:>4d}" for n in loop_sweep)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for k in range(1, args.k_test + 1):
        row = []
        for n in loop_sweep:
            acc = accuracy(model, task, k, n_loops=n, n_samples=512, gen=gen)
            row.append(f"{100 * acc:4.0f}")
        tag = "   " if k <= args.k_train else " * "  # * = unseen in training
        print(f"  {tag}{k}     | " + " | ".join(row))
    print(f"\n  rows marked * use k values never seen during training (k>{args.k_train})")
    print("  'loops' = recurrent depth at inference; training used "
          f"n_loops={args.train_loops}")

    # ---- Demonstrate safe save / load round-trip ----
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        reloaded = OpenMythos.from_pretrained(d)
        x, y = task.batch(64, [args.k_train], gen)
        with torch.no_grad():
            a = model.eval()(x, n_loops=args.train_loops)
            b = reloaded.eval()(x, n_loops=args.train_loops)
        max_diff = (a - b).abs().max().item()
        print(f"\nsafetensors save/load round-trip: max logit diff = {max_diff:.2e}")


if __name__ == "__main__":
    main()
