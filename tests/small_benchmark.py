#!/usr/bin/env python3
"""
Side-by-side training + benchmark of OpenMythos vs. a vanilla transformer on a
small HuggingFace dataset (TinyStories by default, streamed).

Both models share the same tiny MLA config and see the exact same batches in
the same order, so per-step train loss and throughput are directly comparable.
The baseline is a dense stack of the same TransformerBlock primitive with
`use_moe=False`; its unique-layer depth matches the recurrent block's
unique-parameter depth (prelude + 1 + coda), so total parameter counts land in
the same ballpark. Attention kernel is shared (MLA in both models), so any
measured delta reflects the looped recurrent-depth architecture rather than
kernel differences.

What the script measures
------------------------
1. Per-step training loss + tokens/sec for both models, fed identical batches.
2. Periodic held-out eval loss on a separate dataset split (--eval-every).
3. Depth-extrapolation sweep at the end: OpenMythos is trained at
   cfg.max_loop_iters, then evaluated at n_loops in --depth-sweep
   (default 1,2,4,8,16). This is the experiment the recurrent-depth
   architecture is designed to win — eval loss should keep dropping past
   the trained depth if depth extrapolation is working.
4. Summary table with initial/final/avg train loss, wall-clock, avg tok/s,
   and sec/step for both models.

Defaults are tuned for a laptop CPU run in reasonable time; pass --device cuda
and bump --steps / --batch-size / --seq-len for a real comparison.

    # Default CPU smoke run (TinyStories, 1k steps, batch 32, seq 256)
    python tests/small_benchmark.py

    # Heavier GPU run
    python tests/small_benchmark.py --steps 5000 --batch-size 64 --seq-len 512 --device cuda

    # Wikitext instead of TinyStories
    python tests/small_benchmark.py --dataset wikitext --dataset-config wikitext-2-raw-v1

    # Aggressive depth extrapolation sweep
    python tests/small_benchmark.py --depth-sweep 1,2,4,8,16,32
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from open_mythos import MythosConfig, OpenMythos
from open_mythos.main import (
    RMSNorm,
    TransformerBlock,
    precompute_rope_freqs,
)


# ---------------------------------------------------------------------------
# Baseline: dense GQA + SwiGLU transformer
# ---------------------------------------------------------------------------


class BaselineTransformer(nn.Module):
    """Vanilla decoder-only transformer with dense SwiGLU FFNs.

    Reuses OpenMythos's TransformerBlock (attention + FFN kernels are identical)
    so any measured delta reflects the looped recurrent-depth architecture, not
    kernel differences. Supports both attn_type="gqa" and "mla".
    """

    def __init__(self, cfg: MythosConfig, n_layers: int):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        # MLA applies RoPE to qk_rope_head_dim only; GQA rotates the full head_dim.
        rope_dim = (
            cfg.qk_rope_head_dim if cfg.attn_type == "mla" else cfg.dim // cfg.n_heads
        )
        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs(rope_dim, cfg.max_seq_len, cfg.rope_theta),
            persistent=False,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((1, 1, T, T), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        T = input_ids.shape[1]
        x = self.embed(input_ids)
        freqs_cis = self.freqs_cis[:T]
        mask = self._causal_mask(T, x.device) if T > 1 else None
        for i, layer in enumerate(self.layers):
            x = layer(x, freqs_cis, mask, cache_key=f"layer_{i}")
        return self.head(self.norm(x))


# ---------------------------------------------------------------------------
# Dataset: tokenize once, pack into fixed-length next-token pairs
# ---------------------------------------------------------------------------


class PackedLMDataset(Dataset):
    """Flatten an HF text dataset into one token buffer, slice fixed-length pairs.

    Accepts either map-style or streaming (`IterableDataset`) HF datasets —
    iteration stops once `max_tokens` are collected, so large corpora like
    TinyStories can be streamed without downloading the whole thing.
    """

    def __init__(
        self,
        hf_ds,
        tokenizer,
        seq_len: int,
        max_tokens: int,
        text_field: str = "text",
    ):
        buf: list[int] = []
        for sample in hf_ds:
            text = sample[text_field]
            if not text or not text.strip():
                continue
            buf.extend(tokenizer.encode(text, add_special_tokens=False))
            if len(buf) >= max_tokens:
                break
        self.seq_len = seq_len
        n_pairs = max(1, (len(buf) - 1) // seq_len)
        buf = buf[: n_pairs * seq_len + 1]
        self.data = torch.tensor(buf, dtype=torch.long)

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx: int):
        s = idx * self.seq_len
        chunk = self.data[s : s + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class Metrics:
    total_loss: float = 0.0
    total_tokens: int = 0
    total_time: float = 0.0
    steps: int = 0
    first_losses: list[float] = field(default_factory=list)
    last_losses: Deque[float] = field(default_factory=lambda: deque(maxlen=10))

    def update(self, loss: float, tokens: int, seconds: float) -> None:
        self.total_loss += loss
        self.total_tokens += tokens
        self.total_time += seconds
        self.steps += 1
        if len(self.first_losses) < 10:
            self.first_losses.append(loss)
        self.last_losses.append(loss)

    @property
    def avg_loss(self) -> float:
        return self.total_loss / max(1, self.steps)

    @property
    def tok_per_sec(self) -> float:
        return self.total_tokens / max(1e-9, self.total_time)

    @property
    def initial_loss(self) -> float:
        return sum(self.first_losses) / max(1, len(self.first_losses))

    @property
    def final_loss(self) -> float:
        return sum(self.last_losses) / max(1, len(self.last_losses))


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def train_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    vocab_size: int,
) -> tuple[float, float]:
    """Run one optimizer step; return (loss, wall-clock seconds)."""
    t0 = time.perf_counter()
    model.train()
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return loss.item(), time.perf_counter() - t0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    vocab_size: int,
    max_batches: int | None = None,
    n_loops: int | None = None,
) -> float:
    """Mean cross-entropy over (up to `max_batches`) of the loader.

    `n_loops` is only forwarded to OpenMythos; for any other module the kwarg
    is dropped, so the same function benchmarks baseline and mythos uniformly.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if isinstance(model, OpenMythos):
            logits = model(x, n_loops=n_loops)
        else:
            logits = model(x)
        # sum-reduction so we weight by token count, not batch count
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
    return total_loss / max(1, total_tokens)


# ---------------------------------------------------------------------------
# Config + utilities
# ---------------------------------------------------------------------------


def build_tiny_cfg(vocab_size: int, seq_len: int) -> MythosConfig:
    """Tiny shared config with MLA attention — runs in reasonable time on CPU.

    MLA LoRA ranks and head dims scale with `dim=128` instead of the
    2048-dim-sized defaults (q_lora_rank=1536, qk_nope_head_dim=128, ...),
    which would otherwise dominate the parameter count at this scale.
    """
    return MythosConfig(
        vocab_size=vocab_size,
        dim=128,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=seq_len,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        attn_type="mla",
        kv_lora_rank=64,
        q_lora_rank=128,
        qk_rope_head_dim=16,
        qk_nope_head_dim=32,
        v_head_dim=32,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=128,
        lora_rank=4,
        rope_theta=10000.0,
        dropout=0.0,
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def fmt_count(n: float) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}T"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    # Defaults point at TinyStories — simpler vocabulary + shorter documents
    # lets a dim=128 model actually reach a meaningful loss in modest time.
    p.add_argument("--dataset", default="roneneldan/TinyStories")
    p.add_argument(
        "--dataset-config",
        default="",
        help="pass '' for datasets with no config (e.g. TinyStories)",
    )
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="validation")
    p.add_argument(
        "--train-tokens",
        type=int,
        default=5_000_000,
        help="max tokens to materialize for the training buffer",
    )
    p.add_argument(
        "--eval-tokens",
        type=int,
        default=200_000,
        help="max tokens to materialize for the held-out eval buffer",
    )
    p.add_argument("--text-field", default="text")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument(
        "--eval-every",
        type=int,
        default=200,
        help="run held-out eval every N steps (0 disables)",
    )
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument(
        "--depth-sweep",
        default="1,2,4,8,16",
        help="comma-separated n_loops values for OpenMythos depth-extrapolation eval",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def load_text_ds(name: str, config: str, split: str):
    """Streaming `load_dataset` with optional config (empty string == no config)."""
    if config:
        return load_dataset(name, config, split=split, streaming=True)
    return load_dataset(name, split=split, streaming=True)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(
        f"[setup] device={device}  batch={args.batch_size}  "
        f"seq_len={args.seq_len}  steps={args.steps}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # AutoTokenizer.vocab_size can be smaller than the head size for BPE
    # tokenizers with added tokens; use len(tokenizer) to be safe.
    vocab_size = len(tokenizer)
    print(f"[setup] tokenizer={args.tokenizer}  vocab_size={vocab_size:,}")

    # ------------------------------------------------------------------
    # Data: streamed train + held-out eval splits
    # ------------------------------------------------------------------
    print(f"[setup] dataset={args.dataset}  config={args.dataset_config or '∅'}")
    raw_train = load_text_ds(args.dataset, args.dataset_config, args.train_split)
    train_ds = PackedLMDataset(
        raw_train, tokenizer, args.seq_len, args.train_tokens, args.text_field
    )
    raw_eval = load_text_ds(args.dataset, args.dataset_config, args.eval_split)
    eval_ds = PackedLMDataset(
        raw_eval, tokenizer, args.seq_len, args.eval_tokens, args.text_field
    )
    print(
        f"[setup] train tokens={train_ds.data.numel():,}  pairs={len(train_ds)}  |  "
        f"eval tokens={eval_ds.data.numel():,}  pairs={len(eval_ds)}"
    )

    torch.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    # ------------------------------------------------------------------
    # Models — same init seed so both start from the same embedding
    # ------------------------------------------------------------------
    cfg = build_tiny_cfg(vocab_size, args.seq_len)

    torch.manual_seed(args.seed)
    mythos = OpenMythos(cfg).to(device)

    # Parameter-matched depth: prelude + one unique recurrent block + coda.
    baseline_layers = cfg.prelude_layers + 1 + cfg.coda_layers
    torch.manual_seed(args.seed)
    baseline = BaselineTransformer(cfg, n_layers=baseline_layers).to(device)

    n_m, n_b = count_params(mythos), count_params(baseline)
    print(
        f"[setup] OpenMythos params  = {fmt_count(n_m)}  ({n_m:,})\n"
        f"[setup] Baseline  params  = {fmt_count(n_b)}  ({n_b:,})  "
        f"[{baseline_layers} layers]"
    )
    print(
        f"[setup] Mythos runtime depth = prelude({cfg.prelude_layers}) + "
        f"loops({cfg.max_loop_iters}) + coda({cfg.coda_layers}) = "
        f"{cfg.prelude_layers + cfg.max_loop_iters + cfg.coda_layers}"
    )

    opt_m = torch.optim.AdamW(
        mythos.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    opt_b = torch.optim.AdamW(
        baseline.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    mm, bm = Metrics(), Metrics()
    eval_history: list[tuple[int, float, float]] = []  # (step, mythos_eval, base_eval)

    header = (
        f"\n{'step':>6} | {'mythos loss':>12} | {'base loss':>10} | "
        f"{'mythos tok/s':>13} | {'base tok/s':>11}"
    )
    print(header)
    print("-" * len(header))

    # ------------------------------------------------------------------
    # Training loop with periodic held-out eval
    # ------------------------------------------------------------------
    data_iter = iter(train_loader)
    t_total = time.perf_counter()
    for step in range(1, args.steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        tokens = x.numel()

        loss_m, dt_m = train_step(mythos, x, y, opt_m, device, vocab_size)
        loss_b, dt_b = train_step(baseline, x, y, opt_b, device, vocab_size)

        mm.update(loss_m, tokens, dt_m)
        bm.update(loss_b, tokens, dt_b)

        if step == 1 or step % args.log_every == 0:
            print(
                f"{step:>6} | {loss_m:>12.4f} | {loss_b:>10.4f} | "
                f"{tokens / dt_m:>13,.0f} | {tokens / dt_b:>11,.0f}"
            )

        if args.eval_every and step % args.eval_every == 0:
            eval_m = evaluate(
                mythos, eval_loader, device, vocab_size, args.eval_batches
            )
            eval_b = evaluate(
                baseline, eval_loader, device, vocab_size, args.eval_batches
            )
            eval_history.append((step, eval_m, eval_b))
            print(
                f"  [eval @ step {step}]  mythos {eval_m:.4f}   baseline {eval_b:.4f}   "
                f"(Δ = {eval_m - eval_b:+.4f})"
            )

    total_wall = time.perf_counter() - t_total

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    bar = "=" * 70
    print(f"\n{bar}\nSummary ({args.steps} steps, wall clock {total_wall:.1f}s)\n{bar}")
    print(f"  {'':<24} {'OpenMythos':>16}   {'Baseline':>16}")
    print(f"  {'params':<24} {fmt_count(n_m):>16}   {fmt_count(n_b):>16}")
    print(
        f"  {'initial train (first 10)':<24} "
        f"{mm.initial_loss:>16.4f}   {bm.initial_loss:>16.4f}"
    )
    print(
        f"  {'final train (last 10)':<24} "
        f"{mm.final_loss:>16.4f}   {bm.final_loss:>16.4f}"
    )
    print(
        f"  {'avg train (all steps)':<24} "
        f"{mm.avg_loss:>16.4f}   {bm.avg_loss:>16.4f}"
    )
    print(
        f"  {'train time (sec)':<24} "
        f"{mm.total_time:>16.2f}   {bm.total_time:>16.2f}"
    )
    print(
        f"  {'avg tok/s':<24} " f"{mm.tok_per_sec:>16,.0f}   {bm.tok_per_sec:>16,.0f}"
    )
    print(
        f"  {'sec/step':<24} "
        f"{mm.total_time / max(1, mm.steps):>16.4f}   "
        f"{bm.total_time / max(1, bm.steps):>16.4f}"
    )

    # ------------------------------------------------------------------
    # Depth extrapolation: OpenMythos eval loss as a function of n_loops.
    # Trained at cfg.max_loop_iters; we run inference with a sweep to see
    # whether additional loops keep improving (depth extrapolation) or the
    # model collapses outside the trained regime.
    # ------------------------------------------------------------------
    loops_sweep = sorted({int(s) for s in args.depth_sweep.split(",") if s.strip()})
    print(f"\n{bar}\nDepth extrapolation (held-out eval, full eval set)\n{bar}")
    baseline_eval = evaluate(baseline, eval_loader, device, vocab_size)
    print(f"  Baseline (fixed depth)          : eval loss = {baseline_eval:.4f}")
    # First collect all sweep losses, then print with deltas vs. the trained depth.
    sweep: list[tuple[int, float]] = []
    for nl in loops_sweep:
        sweep.append(
            (nl, evaluate(mythos, eval_loader, device, vocab_size, n_loops=nl))
        )
    trained_loss = next((loss for nl, loss in sweep if nl == cfg.max_loop_iters), None)
    print(f"  OpenMythos (trained at n_loops={cfg.max_loop_iters}):")
    print(f"    {'n_loops':>8}  {'eval loss':>10}  {'Δ vs trained':>14}")
    for nl, loss in sweep:
        if trained_loss is None or nl == cfg.max_loop_iters:
            delta_str = ""
        else:
            delta_str = f"{loss - trained_loss:+.4f}"
        marker = " ←trained" if nl == cfg.max_loop_iters else ""
        print(f"    {nl:>8}  {loss:>10.4f}  {delta_str:>14}{marker}")


if __name__ == "__main__":
    main()
