"""
OpenMythos loop-scaling validation — training script.

Usage:
    python train.py --run_name looped_8   --max_loop_iters 8   --max_steps 30000
    python train.py --run_name baseline_1 --max_loop_iters 1   --max_steps 30000

Writes:
    /workspace/runs/<run_name>/train.log      (plaintext per-step metrics)
    /workspace/runs/<run_name>/config.json
    /workspace/runs/<run_name>/ckpt_*.pt
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from open_mythos.main import OpenMythos

from config import TrainConfig, mythos_150m
from data import build_loader, get_tokenizer


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.learning_rate - cfg.min_lr) * coeff


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", type=str, required=True)
    ap.add_argument("--max_loop_iters", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=30000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--output_dir", type=str, default="/workspace/runs")
    ap.add_argument(
        "--loop_sample_mode",
        type=str,
        default="fixed",
        choices=["fixed", "random_set"],
        help="fixed: always use --max_loop_iters. random_set: uniformly sample each step from --loop_choices.",
    )
    ap.add_argument(
        "--loop_choices",
        type=int,
        nargs="+",
        default=[4, 6, 8, 12, 16],
        help="Set of n_loops values to uniformly sample from (only used in random_set mode).",
    )
    args = ap.parse_args()

    cfg = TrainConfig(
        run_name=args.run_name,
        max_loop_iters=args.max_loop_iters,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        seq_len=args.seq_len,
        learning_rate=args.lr,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        output_dir=args.output_dir,
    )

    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(0)

    loop_mode = args.loop_sample_mode
    loop_choices = args.loop_choices if loop_mode == "random_set" else [cfg.max_loop_iters]
    print(f"==> run_name={cfg.run_name}  max_loop_iters={cfg.max_loop_iters}  "
          f"loop_mode={loop_mode}  loop_choices={loop_choices}")
    print(f"==> output_dir={out_dir}")

    tok = get_tokenizer(cfg.tokenizer)
    vocab_size = tok.vocab_size

    mcfg = mythos_150m(max_loop_iters=cfg.max_loop_iters)
    mcfg.vocab_size = vocab_size
    mcfg.max_seq_len = cfg.seq_len
    model = OpenMythos(mcfg).to(device=device, dtype=dtype)

    total, trainable = count_params(model)
    print(f"==> params total={total/1e6:.1f}M  trainable={trainable/1e6:.1f}M")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    loader = build_loader(
        tokenizer=tok,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        num_workers=2,
    )
    loader_iter = iter(loader)

    log_path = out_dir / "train.log"
    log_f = log_path.open("w", buffering=1)
    log_f.write("step\ttokens\tn_loops\tlr\tloss\tgrad_norm\trho_A\tstep_s\ttok_per_s\tgpu_mem_gb\n")

    rng = random.Random(0)

    model.train()
    tokens_seen = 0
    ema_loss = None
    t_start = time.time()
    step_start = time.time()
    accum_loss = 0.0

    for step in range(cfg.max_steps):
        lr = cosine_lr(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        if loop_mode == "random_set":
            n_loops_step = rng.choice(loop_choices)
        else:
            n_loops_step = cfg.max_loop_iters

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(cfg.grad_accum_steps):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x, n_loops=n_loops_step)
            loss = F.cross_entropy(
                logits.float().view(-1, mcfg.vocab_size),
                y.view(-1),
            )
            (loss / cfg.grad_accum_steps).backward()
            accum_loss += loss.item() / cfg.grad_accum_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        opt.step()

        tokens_seen += cfg.batch_size * cfg.seq_len * cfg.grad_accum_steps
        ema_loss = accum_loss if ema_loss is None else 0.98 * ema_loss + 0.02 * accum_loss

        if (step + 1) % cfg.log_every == 0 or step == 0:
            torch.cuda.synchronize()
            dt = time.time() - step_start
            tok_per_s = cfg.batch_size * cfg.seq_len * cfg.grad_accum_steps * cfg.log_every / dt if step > 0 else 0
            rho_A = model.recurrent.injection.get_A().max().item()
            mem = torch.cuda.max_memory_allocated() / 1e9
            line = (
                f"{step+1}\t{tokens_seen}\t{n_loops_step}\t{lr:.2e}\t{accum_loss:.4f}\t{grad_norm:.3f}\t"
                f"{rho_A:.4f}\t{dt/max(1,cfg.log_every):.3f}\t{tok_per_s:.0f}\t{mem:.1f}"
            )
            log_f.write(line + "\n")
            print(
                f"step {step+1}/{cfg.max_steps}  "
                f"n_loops={n_loops_step}  "
                f"loss={accum_loss:.3f}  ema={ema_loss:.3f}  "
                f"lr={lr:.2e}  gnorm={grad_norm:.2f}  "
                f"rho={rho_A:.3f}  "
                f"tok/s={tok_per_s:.0f}  mem={mem:.1f}GB"
            )
            step_start = time.time()

        if (step + 1) % cfg.ckpt_every == 0 or step + 1 == cfg.max_steps:
            ckpt_path = out_dir / f"ckpt_{step+1}.pt"
            torch.save(
                {
                    "step": step + 1,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "cfg": cfg.to_dict(),
                    "mcfg": mcfg.__dict__,
                    "tokens_seen": tokens_seen,
                },
                ckpt_path,
            )
            print(f"==> saved ckpt {ckpt_path}  tokens={tokens_seen/1e6:.1f}M")

    log_f.close()
    total_time = time.time() - t_start
    print(f"==> done in {total_time/3600:.2f}h  tokens={tokens_seen/1e9:.2f}B")


if __name__ == "__main__":
    main()
