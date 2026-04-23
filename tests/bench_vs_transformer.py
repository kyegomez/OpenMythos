"""
OpenMythos vs. vanilla GQA+MoE transformer benchmark.

Compares OpenMythos (Prelude + looped Recurrent Block + Coda with ACT halting,
LTI-stable injection, LoRA depth adapter) against a parameter-matched vanilla
transformer built from the same GQAttention + MoEFFN building blocks stacked
non-recurrently. The baseline reuses OpenMythos primitives so the comparison
isolates the recurrent-depth architecture, not the kernels.

Metrics reported:
    - Parameter counts (total, MoE-active approximation)
    - Prefill latency + throughput at several sequence lengths
    - Decode (autoregressive step) latency with KV cache
    - Peak memory (CUDA only)
    - OpenMythos depth-scaling sweep: latency vs. n_loops

Run:
    python benchmarks/bench_vs_transformer.py                     # small CPU/GPU smoke test
    python benchmarks/bench_vs_transformer.py --size 1b --device cuda
    python benchmarks/bench_vs_transformer.py --seq-lens 128,512,2048 --n-loops 1,4,8,16
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from open_mythos import MythosConfig, OpenMythos, mythos_1b
from open_mythos.main import (
    RMSNorm,
    TransformerBlock,
    precompute_rope_freqs,
)


# ---------------------------------------------------------------------------
# Baseline: non-looped GQA + MoE transformer
# ---------------------------------------------------------------------------


class BaselineTransformer(nn.Module):
    """
    Vanilla decoder-only transformer with GQA attention and MoE FFNs, stacked
    non-recurrently. Shares TransformerBlock / GQAttention / MoEFFN kernels
    with OpenMythos so any speed delta is attributable to the recurrent-depth
    architecture rather than the underlying attention/FFN implementation.
    """

    def __init__(self, cfg: MythosConfig, n_layers: int):
        super().__init__()
        self.cfg = cfg
        self.n_layers = n_layers
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=True) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        head_dim = cfg.dim // cfg.n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs(head_dim, cfg.max_seq_len, cfg.rope_theta),
            persistent=False,
        )

    @staticmethod
    def _causal_mask(T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((1, 1, T, T), float("-inf"), device=device, dtype=dtype)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: Optional[dict] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        T = input_ids.shape[1]
        x = self.embed(input_ids)
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]
        mask = self._causal_mask(T, x.device, x.dtype) if T > 1 else None
        for i, layer in enumerate(self.layers):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"layer_{i}")
        return self.head(self.norm(x))


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_fn(fn, device: torch.device, warmup: int = 2, trials: int = 5) -> float:
    """Returns median wall-clock seconds over `trials` after `warmup` runs."""
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(trials):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def peak_mem_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def reset_mem(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------


@dataclass
class ParamCounts:
    total: int
    moe_active_est: int  # active per token (shared + top-k routed)


def count_params(model: nn.Module, cfg: MythosConfig) -> ParamCounts:
    total = sum(p.numel() for p in model.parameters())
    # Rough active-per-token count for MoE layers: shared + top-k routed fraction.
    # For simplicity we report total and an estimated activation ratio separately.
    active_ratio = (cfg.n_shared_experts + cfg.n_experts_per_tok) / (
        cfg.n_shared_experts + cfg.n_experts
    )
    # Only FFN parameters shrink under activation; attention + embed/head are always on.
    # This is a coarse lower bound on active params.
    ffn_params = 0
    other_params = 0
    for name, p in model.named_parameters():
        if ".ffn." in name or name.startswith("ffn.") or ".experts." in name:
            ffn_params += p.numel()
        else:
            other_params += p.numel()
    active_est = other_params + int(ffn_params * active_ratio)
    return ParamCounts(total=total, moe_active_est=active_est)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def bench_prefill(
    model: nn.Module,
    vocab_size: int,
    batch: int,
    seq_len: int,
    device: torch.device,
    n_loops: Optional[int] = None,
) -> tuple[float, float]:
    """Returns (median_seconds, tokens_per_sec)."""
    ids = torch.randint(0, vocab_size, (batch, seq_len), device=device)

    if isinstance(model, OpenMythos):

        def run() -> None:
            with torch.no_grad():
                model(ids, n_loops=n_loops)

    else:

        def run() -> None:
            with torch.no_grad():
                model(ids)

    secs = time_fn(run, device)
    tps = (batch * seq_len) / secs
    return secs, tps


def bench_decode(
    model: nn.Module,
    vocab_size: int,
    batch: int,
    prompt_len: int,
    decode_steps: int,
    device: torch.device,
    n_loops: Optional[int] = None,
) -> tuple[float, float]:
    """
    Prefill a `prompt_len` prompt, then time `decode_steps` single-token decode
    steps with KV cache. Returns (avg_seconds_per_step, decode_tokens_per_sec).
    """
    prompt = torch.randint(0, vocab_size, (batch, prompt_len), device=device)

    def one_run() -> None:
        kv_cache: dict = {}
        with torch.no_grad():
            if isinstance(model, OpenMythos):
                model(prompt, n_loops=n_loops, kv_cache=kv_cache, start_pos=0)
            else:
                model(prompt, kv_cache=kv_cache, start_pos=0)
            for i in range(decode_steps):
                next_tok = torch.randint(0, vocab_size, (batch, 1), device=device)
                if isinstance(model, OpenMythos):
                    model(
                        next_tok,
                        n_loops=n_loops,
                        kv_cache=kv_cache,
                        start_pos=prompt_len + i,
                    )
                else:
                    model(next_tok, kv_cache=kv_cache, start_pos=prompt_len + i)

    secs = time_fn(one_run, device, warmup=1, trials=3)
    per_step = secs / decode_steps
    tps = batch * decode_steps / secs
    return per_step, tps


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def small_cfg() -> MythosConfig:
    """Tiny config for smoke tests — runs on CPU in seconds."""
    return MythosConfig(
        vocab_size=1024,
        dim=256,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=1024,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=8,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=128,
        lora_rank=4,
        dropout=0.0,
    )


def get_cfg(size: str) -> MythosConfig:
    size = size.lower()
    if size == "small":
        return small_cfg()
    if size == "1b":
        cfg = mythos_1b()
        # GQA for apples-to-apples; MLA changes KV shape semantics.
        cfg.attn_type = "gqa"
        return cfg
    raise ValueError(f"unknown size: {size!r} (use 'small' or '1b')")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def fmt_count(n: int) -> str:
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}P"


def print_header(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--size", default="small", choices=["small", "1b"])
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp32", "bf16", "fp16"],
        help="'auto' picks fp32 on CPU and bf16 on CUDA",
    )
    p.add_argument("--batch", type=int, default=1)
    p.add_argument(
        "--seq-lens",
        default="128,512",
        help="comma-separated prefill sequence lengths",
    )
    p.add_argument(
        "--n-loops",
        default="1,4,8",
        help="comma-separated loop counts to sweep (OpenMythos only)",
    )
    p.add_argument(
        "--decode-steps",
        type=int,
        default=32,
        help="number of autoregressive decode steps after prefill",
    )
    p.add_argument(
        "--decode-prompt-len",
        type=int,
        default=128,
        help="prefill length before decode",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype_arg = args.dtype
    if dtype_arg == "auto":
        dtype_arg = "bf16" if device.type == "cuda" else "fp32"
    dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[dtype_arg]

    seq_lens = [int(s) for s in args.seq_lens.split(",") if s.strip()]
    n_loops_sweep = [int(s) for s in args.n_loops.split(",") if s.strip()]

    cfg = get_cfg(args.size)
    print_header(f"Config: size={args.size}  device={device}  dtype={dtype_arg}")
    print(
        f"  dim={cfg.dim}  n_heads={cfg.n_heads}  n_kv_heads={cfg.n_kv_heads}  "
        f"prelude={cfg.prelude_layers}  coda={cfg.coda_layers}  "
        f"max_loop_iters={cfg.max_loop_iters}\n"
        f"  experts={cfg.n_experts}  shared={cfg.n_shared_experts}  "
        f"top_k={cfg.n_experts_per_tok}  expert_dim={cfg.expert_dim}"
    )

    # Build models. Baseline depth = prelude + 1 (one unique recurrent block) + coda
    # to match the unique-parameter depth of OpenMythos (parameter-matched baseline).
    baseline_n_layers = cfg.prelude_layers + 1 + cfg.coda_layers

    torch.manual_seed(0)
    mythos = OpenMythos(cfg).to(device=device, dtype=dtype).eval()
    torch.manual_seed(0)
    baseline = (
        BaselineTransformer(cfg, n_layers=baseline_n_layers)
        .to(device=device, dtype=dtype)
        .eval()
    )

    m_params = count_params(mythos, cfg)
    b_params = count_params(baseline, cfg)
    print_header(
        "Parameters (block-matched: baseline depth = prelude + 1 recurrent + coda)"
    )
    print(
        f"  OpenMythos : total={fmt_count(m_params.total):>10}   "
        f"active/tok≈{fmt_count(m_params.moe_active_est):>10}"
    )
    print(
        f"  Baseline   : total={fmt_count(b_params.total):>10}   "
        f"active/tok≈{fmt_count(b_params.moe_active_est):>10}"
    )
    print(
        f"  Baseline unique layers = {baseline_n_layers}  "
        f"(Mythos total runtime depth at max_loops = "
        f"{cfg.prelude_layers + cfg.max_loop_iters + cfg.coda_layers})"
    )

    # ---- Prefill ----
    print_header("Prefill latency (batch={batch})".format(batch=args.batch))
    header = f"  {'model':<26} {'seq':>6} {'sec':>10} {'tok/s':>12} {'peak MB':>10}"
    print(header)
    for seq_len in seq_lens:
        if seq_len > cfg.max_seq_len:
            print(f"  skip seq_len={seq_len} (> max_seq_len={cfg.max_seq_len})")
            continue

        reset_mem(device)
        secs, tps = bench_prefill(baseline, cfg.vocab_size, args.batch, seq_len, device)
        mem = peak_mem_mb(device)
        print(
            f"  {'Baseline (stacked)':<26} {seq_len:>6} "
            f"{secs*1000:>9.2f}ms {tps:>12,.0f} {mem:>10.1f}"
        )

        for nl in n_loops_sweep:
            reset_mem(device)
            secs, tps = bench_prefill(
                mythos, cfg.vocab_size, args.batch, seq_len, device, n_loops=nl
            )
            mem = peak_mem_mb(device)
            print(
                f"  {'OpenMythos (loops=' + str(nl) + ')':<26} {seq_len:>6} "
                f"{secs*1000:>9.2f}ms {tps:>12,.0f} {mem:>10.1f}"
            )

    # ---- Decode ----
    print_header(
        f"Decode latency (prefill {args.decode_prompt_len} tokens + "
        f"{args.decode_steps} decode steps, batch={args.batch})"
    )
    print(f"  {'model':<26} {'sec/step':>12} {'decode tok/s':>14}")

    reset_mem(device)
    per_step, tps = bench_decode(
        baseline,
        cfg.vocab_size,
        args.batch,
        args.decode_prompt_len,
        args.decode_steps,
        device,
    )
    print(f"  {'Baseline (stacked)':<26} {per_step*1000:>10.2f}ms {tps:>14,.1f}")

    for nl in n_loops_sweep:
        reset_mem(device)
        per_step, tps = bench_decode(
            mythos,
            cfg.vocab_size,
            args.batch,
            args.decode_prompt_len,
            args.decode_steps,
            device,
            n_loops=nl,
        )
        print(
            f"  {'OpenMythos (loops=' + str(nl) + ')':<26} "
            f"{per_step*1000:>10.2f}ms {tps:>14,.1f}"
        )

    # ---- Depth scaling ----
    print_header(
        "OpenMythos depth scaling (fixed seq={}, batch={})".format(
            seq_lens[0], args.batch
        )
    )
    print(f"  {'n_loops':>8} {'sec':>10} {'tok/s':>12} {'Δ vs loops=1':>14}")
    base_secs = None
    for nl in n_loops_sweep:
        reset_mem(device)
        secs, tps = bench_prefill(
            mythos, cfg.vocab_size, args.batch, seq_lens[0], device, n_loops=nl
        )
        if base_secs is None:
            base_secs = secs
            delta = "1.00x"
        else:
            delta = f"{secs / base_secs:.2f}x"
        print(f"  {nl:>8} {secs*1000:>9.2f}ms {tps:>12,.0f} {delta:>14}")

    print("\nDone.")


if __name__ == "__main__":
    main()
