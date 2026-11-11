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
3. For every sweep column it also prints how many loops the model *actually*
   ran before ACT halted everything. If that number stays at 1-2 no matter
   what n_loops you ask for, extra depth is never exercised and a flat table
   is uninformative; raise --act-threshold (1.0 = never halt early).

The printed numbers are whatever actually happens on this run (seeded). No
result is hard-coded.

Small machine (CPU, a few minutes):

    python examples/depth_extrapolation.py

Big machine (GPU, the setting the thesis actually needs):

    python examples/depth_extrapolation.py --preset big --device cuda \
        --steps 40000 --save-dir runs/depth_big

`--preset big` changes the model-size defaults and the learning rate and warmup
that go with them; every value can still be overridden by its own flag. See README "Depth-extrapolation demo".
"""

import argparse
import json
import os
import tempfile
import time

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
        """Return (input_ids [B, L], target [B]) sampling k uniformly from k_values.

        Fully vectorized: P^k(s) is computed by k gathers over the batch, so
        large batches on the big machine do not pay a Python loop per example.
        """
        M = self.M
        P = torch.argsort(torch.rand(batch_size, M, generator=generator), dim=1)
        s = torch.randint(0, M, (batch_size,), generator=generator)
        kv = torch.tensor(k_values)
        k = kv[torch.randint(0, len(kv), (batch_size,), generator=generator)]

        cur = s.clone()
        for hop in range(1, int(k.max()) + 1):
            nxt = P.gather(1, cur.unsqueeze(1)).squeeze(1)
            cur = torch.where(k >= hop, nxt, cur)

        inputs = torch.empty(batch_size, self.seq_len, dtype=torch.long)
        inputs[:, :M] = P
        inputs[:, M] = self.sep
        inputs[:, M + 1] = s
        inputs[:, M + 2] = self.M + k
        return inputs, cur


# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

# "small" reproduces the original CPU demo exactly (271k params).
# "big" is a starting point for a 32 GB / GPU box, not a tuned optimum.
PRESETS = {
    "small": dict(dim=96, heads=4, kv_heads=2, prelude=1, coda=1,
                  experts=4, shared_experts=1, experts_per_tok=2,
                  expert_dim=64, lora_rank=8),
    "big": dict(dim=512, heads=8, kv_heads=4, prelude=2, coda=2,
                experts=8, shared_experts=2, experts_per_tok=2,
                expert_dim=256, lora_rank=32),
}

# Learning rate and warmup that go with each preset. The big model does not
# learn at the small model's 3e-3 (loss stays at chance); 3e-4 with a short
# warmup does.
TRAIN_DEFAULTS = {
    "small": dict(lr=3e-3, warmup=0),
    "big": dict(lr=3e-4, warmup=500),
}


class LoopCounter:
    """Counts how many recurrent loop iterations ran in the last forward pass.

    The recurrent block breaks out of its loop once every position has halted,
    so counting calls to the inner TransformerBlock per forward gives the
    depth the model actually used, independent of the n_loops you requested.
    """

    def __init__(self, model: OpenMythos):
        self.count = 0
        self._handle = model.recurrent.block.register_forward_hook(self._hook)

    def _hook(self, *_):
        self.count += 1

    def reset(self):
        self.count = 0

    def remove(self):
        self._handle.remove()


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def accuracy(model, task, k, n_loops, n_samples, gen, device, counter=None):
    """Return (accuracy, max loops actually run) over n_samples examples of P^k(s)."""
    model.eval()
    correct = 0
    seen = 0
    loops_run = 0
    with torch.no_grad():
        while seen < n_samples:
            bs = min(256, n_samples - seen)
            x, y = task.batch(bs, [k], gen)
            x, y = x.to(device), y.to(device)
            if counter is not None:
                counter.reset()
            logits = model(x, n_loops=n_loops)[:, -1, : task.M]
            if counter is not None:
                loops_run = max(loops_run, counter.count)
            correct += (logits.argmax(-1) == y).sum().item()
            seen += bs
    return correct / n_samples, loops_run


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = ap.add_argument_group("task")
    g.add_argument("--states", type=int, default=6, help="number of states M")
    g.add_argument("--k-train", type=int, default=3, help="max hops seen in training")
    g.add_argument("--k-test", type=int, default=5, help="max hops at evaluation")

    g = ap.add_argument_group("model (defaults come from --preset; any flag overrides)")
    g.add_argument("--preset", choices=sorted(PRESETS), default="small")
    g.add_argument("--dim", type=int)
    g.add_argument("--heads", type=int)
    g.add_argument("--kv-heads", type=int)
    g.add_argument("--prelude", type=int, help="prelude (non-recurrent) layers")
    g.add_argument("--coda", type=int, help="coda (non-recurrent) layers")
    g.add_argument("--experts", type=int, help="routed MoE experts")
    g.add_argument("--shared-experts", type=int)
    g.add_argument("--experts-per-tok", type=int)
    g.add_argument("--expert-dim", type=int)
    g.add_argument("--lora-rank", type=int, help="per-loop LoRA rank")
    g.add_argument("--train-loops", type=int, default=6, help="loop count during training")
    g.add_argument(
        "--act-threshold", type=float, default=0.999,
        help="ACT cumulative-halt threshold; 1.0 disables early halting so every "
             "requested loop runs (recommended when probing depth)",
    )
    g.add_argument("--dropout", type=float, default=0.0)

    g = ap.add_argument_group("training")
    g.add_argument("--steps", type=int, default=3000)
    g.add_argument("--batch", type=int, default=128)
    g.add_argument("--lr", type=float,
                   help="peak learning rate (default: 3e-3 for small, 3e-4 for big)")
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--warmup", type=int,
                   help="linear LR warmup steps (default: 0 for small, 500 for big)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="auto", help="cpu, cuda, cuda:1, mps, or auto")
    g.add_argument("--threads", type=int, help="torch CPU threads (default: torch's choice)")
    g.add_argument("--log-every", type=int, default=500)

    g = ap.add_argument_group("evaluation")
    g.add_argument(
        "--eval-loops", type=int, nargs="+",
        help="inference loop counts to sweep (default: 1 2 train-loops 2*train-loops)",
    )
    g.add_argument("--eval-samples", type=int, default=512, help="examples per table cell")
    g.add_argument("--eval-every", type=int, default=0,
                   help="also print the table every N steps (0 = only at the end)")

    g = ap.add_argument_group("output")
    g.add_argument("--save-dir", help="save the final model (safetensors) and results.json here")
    g.add_argument("--resume", help="load a save-dir checkpoint and skip training")
    return ap


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_config(args, task) -> MythosConfig:
    p = dict(PRESETS[args.preset])
    for key in p:
        override = getattr(args, key)
        if override is not None:
            p[key] = override
    return MythosConfig(
        vocab_size=task.vocab_size,
        dim=p["dim"],
        n_heads=p["heads"],
        n_kv_heads=p["kv_heads"],
        max_seq_len=task.seq_len + 2,
        max_loop_iters=args.train_loops,
        prelude_layers=p["prelude"],
        coda_layers=p["coda"],
        attn_type="gqa",
        n_experts=p["experts"],
        n_shared_experts=p["shared_experts"],
        n_experts_per_tok=p["experts_per_tok"],
        expert_dim=p["expert_dim"],
        lora_rank=p["lora_rank"],
        act_threshold=args.act_threshold,
        dropout=args.dropout,
    )


def eval_table(model, task, args, loop_sweep, gen, device, counter):
    """Print accuracy by (k, n_loops) plus loops actually run; return rows as dicts."""
    print("\nAccuracy (%) by hop count k and inference loop count:")
    header = "  k \\ loops | " + " | ".join(f"{n:>4d}" for n in loop_sweep)
    print(header)
    print("  " + "-" * (len(header) - 2))
    rows = []
    ran = {n: 0 for n in loop_sweep}
    for k in range(1, args.k_test + 1):
        cells = []
        for n in loop_sweep:
            acc, used = accuracy(model, task, k, n, args.eval_samples, gen, device, counter)
            ran[n] = max(ran[n], used)
            cells.append(f"{100 * acc:4.0f}")
            rows.append({"k": k, "n_loops": n, "accuracy": acc, "loops_run": used})
        tag = "   " if k <= args.k_train else " * "  # * = unseen in training
        print(f"  {tag}{k}     | " + " | ".join(cells))
    print("  loops run | " + " | ".join(f"{ran[n]:>4d}" for n in loop_sweep))
    print(f"\n  rows marked * use k values never seen during training (k>{args.k_train})")
    print("  'loops' = recurrent depth requested at inference; training used "
          f"n_loops={args.train_loops}")
    print("  'loops run' = depth actually executed before ACT halted every position")
    return rows


def main():
    args = build_parser().parse_args()
    for key, value in TRAIN_DEFAULTS[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    task = PointerChase(args.states, args.k_test)
    cfg = make_config(args, task)
    if args.resume:
        model = OpenMythos.from_pretrained(args.resume)
        cfg = model.cfg
    else:
        model = OpenMythos(cfg)
    model.to(device)
    counter = LoopCounter(model)

    print(
        f"Task: P^k(s) with M={args.states} states, "
        f"train k in 1..{args.k_train}, test k in 1..{args.k_test}"
    )
    print(
        f"Model: preset={args.preset} dim={cfg.dim} experts={cfg.n_experts} "
        f"lora_rank={cfg.lora_rank} act_threshold={cfg.act_threshold} "
        f"-> {model.num_parameters():,} params, trained at n_loops={args.train_loops}"
    )
    print(
        f"Device: {device}  batch={args.batch}  steps={args.steps}  "
        f"lr={args.lr:g}  warmup={args.warmup}\n"
    )

    loop_sweep = args.eval_loops or [1, 2, args.train_loops, args.train_loops * 2]
    train_ks = list(range(1, args.k_train + 1))
    history = []

    if not args.resume:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, args.steps - args.warmup)
        )
        t0 = time.time()
        model.train()
        for step in range(1, args.steps + 1):
            if args.warmup and step <= args.warmup:
                for pg in opt.param_groups:
                    pg["lr"] = args.lr * step / args.warmup
            x, y = task.batch(args.batch, train_ks, gen)
            x, y = x.to(device), y.to(device)
            logits = model(x, n_loops=args.train_loops)[:, -1, :]
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if not args.warmup or step > args.warmup:
                cosine.step()
            model.update_router_bias()  # aux-loss-free MoE load balancing
            if step % args.log_every == 0 or step == 1:
                elapsed = time.time() - t0
                rate = step / max(elapsed, 1e-9)
                eta = (args.steps - step) / max(rate, 1e-9)
                print(
                    f"step {step:6d}  loss {loss.item():.4f}  "
                    f"{rate:6.1f} step/s  eta {eta / 60:6.1f} min",
                    flush=True,
                )
                history.append({"step": step, "loss": loss.item()})
            if args.eval_every and step % args.eval_every == 0 and step < args.steps:
                eval_table(model, task, args, loop_sweep, gen, device, counter)
                model.train()
        print(f"\ntraining time: {(time.time() - t0) / 60:.1f} min")

    rows = eval_table(model, task, args, loop_sweep, gen, device, counter)

    # ---- Demonstrate safe save / load round-trip ----
    save_dir = args.save_dir or tempfile.mkdtemp()
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    reloaded = OpenMythos.from_pretrained(save_dir).to(device)
    x, _ = task.batch(64, [args.k_train], gen)
    x = x.to(device)
    with torch.no_grad():
        a = model.eval()(x, n_loops=args.train_loops)
        b = reloaded.eval()(x, n_loops=args.train_loops)
    max_diff = (a - b).abs().max().item()
    print(f"\nsafetensors save/load round-trip: max logit diff = {max_diff:.2e}")

    if args.save_dir:
        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(
                {"args": vars(args), "params": model.num_parameters(),
                 "history": history, "eval": rows, "roundtrip_max_diff": max_diff},
                f, indent=2,
            )
        print(f"saved model + results.json to {save_dir}")
    counter.remove()


if __name__ == "__main__":
    main()
