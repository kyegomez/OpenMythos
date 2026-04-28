"""OpenMythos Benchmarking & Profiling Utilities (100x Enhanced Edition).

Provides:
  - Throughput benchmark (tokens/sec forward/generate)
  - Latency distribution (p50/p90/p99)
  - Memory profiling (peak VRAM, parameter footprint)
  - MoE routing entropy analysis
  - ACT halting depth analysis
  - Comparison table across model sizes
"""
from __future__ import annotations

import gc
import math
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    """Container for benchmark results."""
    name: str
    throughput_tps: float          # tokens / second
    latency_p50_ms: float          # median latency per step (ms)
    latency_p90_ms: float
    latency_p99_ms: float
    peak_memory_mb: float          # peak VRAM or RAM in MB
    total_params_m: float          # total parameters in millions
    active_params_m: float         # active parameters per token (MoE)
    notes: str = ""

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Throughput : {self.throughput_tps:>10.1f} tok/s\n"
            f"  Latency p50: {self.latency_p50_ms:>10.2f} ms\n"
            f"  Latency p90: {self.latency_p90_ms:>10.2f} ms\n"
            f"  Latency p99: {self.latency_p99_ms:>10.2f} ms\n"
            f"  Peak memory: {self.peak_memory_mb:>10.1f} MB\n"
            f"  Params (M) : {self.total_params_m:>10.1f} total / "
            f"{self.active_params_m:.1f} active/tok\n"
            + (f"  Notes      : {self.notes}\n" if self.notes else "")
        )


# ---------------------------------------------------------------------------
# Memory utilities
# ---------------------------------------------------------------------------

@contextmanager
def peak_memory_tracker(device: torch.device):
    """
    Context manager that tracks peak memory usage on CUDA or CPU.

    On CUDA, uses torch.cuda.max_memory_allocated().
    On CPU, uses a simple before/after delta (less accurate).

    Yields:
        A dict with key 'peak_mb' populated after the context exits.
    """
    result = {"peak_mb": 0.0}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        yield result
        result["peak_mb"] = torch.cuda.max_memory_allocated(device) / 1e6
    else:
        try:
            import psutil
            proc = psutil.Process()
            before = proc.memory_info().rss / 1e6
            yield result
            after = proc.memory_info().rss / 1e6
            result["peak_mb"] = max(after - before, 0.0)
        except ImportError:
            yield result
            result["peak_mb"] = 0.0


def model_memory_mb(model: nn.Module, dtype: torch.dtype = torch.float32) -> float:
    """Estimate model weight memory in MB for a given dtype."""
    bytes_per_param = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2, torch.int8: 1}
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * bytes_per_param.get(dtype, 4) / 1e6


def count_active_params(model: nn.Module) -> int:
    """
    Count parameters activated per token for a MoE model.
    Counts: embedding, prelude, coda, recurrent non-expert, + top-k experts.
    """
    from open_mythos.main import OpenMythos, MoEFFN
    if not isinstance(model, OpenMythos):
        return sum(p.numel() for p in model.parameters())

    cfg = model.cfg
    active = 0
    # Embedding lookup: 1 row
    active += cfg.dim
    # Prelude + Coda: dense blocks, all active
    n_dense = cfg.prelude_layers + cfg.coda_layers
    for layer in list(model.prelude) + list(model.coda):
        active += sum(p.numel() for p in layer.parameters())
    # Recurrent block: attn + injection + lora + act + norm (always active)
    rec = model.recurrent
    active += sum(p.numel() for p in rec.block.attn.parameters())
    active += sum(p.numel() for p in rec.injection.parameters())
    active += sum(p.numel() for p in rec.lora.parameters())
    active += sum(p.numel() for p in rec.act.parameters())
    active += sum(p.numel() for p in rec.norm.parameters())
    # MoE: shared experts always + top-k routed experts
    moe = rec.block.ffn
    for shared in moe.shared_experts:
        active += sum(p.numel() for p in shared.parameters())
    # top-k experts (average)
    expert_params = sum(p.numel() for p in moe.routed_experts[0].parameters())
    active += expert_params * cfg.n_experts_per_tok
    # LM head (tied with embedding, so 0 extra)
    return active


# ---------------------------------------------------------------------------
# Throughput benchmark
# ---------------------------------------------------------------------------

def benchmark_forward(
    model: nn.Module,
    batch_size: int = 4,
    seq_len: int = 512,
    n_loops: int = 8,
    n_warmup: int = 3,
    n_runs: int = 20,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> BenchResult:
    """
    Benchmark forward-pass throughput and latency.

    Args:
        model      -- model to benchmark
        batch_size -- batch size
        seq_len    -- sequence length
        n_loops    -- recurrent loop depth
        n_warmup   -- warmup runs (not measured)
        n_runs     -- measured runs
        device     -- target device
        dtype      -- tensor dtype for input

    Returns:
        BenchResult with throughput and latency statistics
    """
    device = device or next(model.parameters()).device
    model.eval()
    vocab_size = model.cfg.vocab_size if hasattr(model, "cfg") else 32000

    ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(ids, n_loops=n_loops)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    # Measure
    latencies = []
    with peak_memory_tracker(device) as mem_ctx:
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = model(ids, n_loops=n_loops)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                latencies.append((time.perf_counter() - t0) * 1000)  # ms

    total_tokens = batch_size * seq_len
    median_ms = statistics.median(latencies)
    latencies_sorted = sorted(latencies)
    p90 = latencies_sorted[int(0.9 * len(latencies_sorted))]
    p99 = latencies_sorted[int(0.99 * len(latencies_sorted))]
    tps = total_tokens / (median_ms / 1000)

    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_active = count_active_params(model) / 1e6

    return BenchResult(
        name=f"forward B={batch_size} T={seq_len} loops={n_loops}",
        throughput_tps=tps,
        latency_p50_ms=median_ms,
        latency_p90_ms=p90,
        latency_p99_ms=p99,
        peak_memory_mb=mem_ctx["peak_mb"],
        total_params_m=n_total,
        active_params_m=n_active,
    )


def benchmark_generate(
    model: nn.Module,
    prompt_len: int = 64,
    gen_len: int = 128,
    n_loops: int = 8,
    temperature: float = 1.0,
    n_warmup: int = 2,
    n_runs: int = 5,
    device: Optional[torch.device] = None,
) -> BenchResult:
    """
    Benchmark autoregressive generation throughput.

    Args:
        model      -- model to benchmark
        prompt_len -- prompt token count
        gen_len    -- number of tokens to generate
        n_loops    -- recurrent loop depth
        temperature-- sampling temperature
        n_warmup   -- warmup runs
        n_runs     -- measured runs
        device     -- target device

    Returns:
        BenchResult with generation throughput statistics
    """
    device = device or next(model.parameters()).device
    model.eval()
    vocab_size = model.cfg.vocab_size if hasattr(model, "cfg") else 32000

    prompt = torch.randint(0, vocab_size, (1, prompt_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model.generate(prompt, max_new_tokens=gen_len, n_loops=n_loops, temperature=temperature)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    latencies = []
    with peak_memory_tracker(device) as mem_ctx:
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = model.generate(prompt, max_new_tokens=gen_len, n_loops=n_loops, temperature=temperature)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                latencies.append((time.perf_counter() - t0) * 1000)

    median_ms = statistics.median(latencies)
    latencies_sorted = sorted(latencies)
    p90 = latencies_sorted[int(0.9 * len(latencies_sorted))]
    p99 = latencies_sorted[int(0.99 * len(latencies_sorted))]
    tps = gen_len / (median_ms / 1000)  # generated tokens per second

    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_active = count_active_params(model) / 1e6

    return BenchResult(
        name=f"generate prompt={prompt_len} gen={gen_len} loops={n_loops}",
        throughput_tps=tps,
        latency_p50_ms=median_ms,
        latency_p90_ms=p90,
        latency_p99_ms=p99,
        peak_memory_mb=mem_ctx["peak_mb"],
        total_params_m=n_total,
        active_params_m=n_active,
    )


# ---------------------------------------------------------------------------
# MoE routing entropy analysis
# ---------------------------------------------------------------------------

def analyze_routing_entropy(
    model: nn.Module,
    n_tokens: int = 1024,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Measure MoE routing entropy and load balance on random inputs.

    Higher entropy = more balanced routing (ideal).
    Lower entropy = routing collapse (problematic).

    Args:
        model    -- OpenMythos model
        n_tokens -- number of random tokens to analyze
        device   -- target device

    Returns:
        Dict with routing_entropy, load_balance_score, max_expert_load
    """
    from open_mythos.main import OpenMythos, MoEFFN
    if not isinstance(model, OpenMythos):
        return {}

    device = device or next(model.parameters()).device
    model.eval()

    vocab_size = model.cfg.vocab_size
    ids = torch.randint(0, vocab_size, (1, n_tokens), device=device)

    expert_counts = torch.zeros(model.cfg.n_experts, device=device)
    hooks = []

    def make_hook(moe_layer):
        def hook(module, input, output):
            x = input[0]
            flat = x.view(-1, x.shape[-1])
            logits = module.router(flat)
            _, topk_idx = (logits + module.router_bias).topk(module.topk, dim=-1)
            for idx in topk_idx.reshape(-1):
                expert_counts[idx.item()] += 1
        return hook

    moe = model.recurrent.block.ffn
    if isinstance(moe, MoEFFN):
        h = moe.register_forward_hook(make_hook(moe))
        hooks.append(h)

    with torch.no_grad():
        model(ids, n_loops=4)

    for h in hooks:
        h.remove()

    # Normalize to probability distribution
    total = expert_counts.sum().item()
    if total == 0:
        return {"routing_entropy": 0.0, "load_balance_score": 0.0, "max_expert_load": 0.0}

    probs = (expert_counts / total).cpu()
    # Entropy (bits)
    entropy = -(probs * (probs + 1e-9).log2()).sum().item()
    max_entropy = math.log2(model.cfg.n_experts)
    balance_score = entropy / max_entropy  # 1.0 = perfectly balanced

    return {
        "routing_entropy_bits": entropy,
        "max_entropy_bits": max_entropy,
        "load_balance_score": balance_score,
        "max_expert_load_pct": probs.max().item() * 100,
        "min_expert_load_pct": probs.min().item() * 100,
    }


# ---------------------------------------------------------------------------
# ACT depth analysis
# ---------------------------------------------------------------------------

def analyze_act_depth(
    model: nn.Module,
    prompts: Optional[List[str]] = None,
    n_tokens: int = 128,
    n_loops: int = 16,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Analyze Adaptive Computation Time halting depths.

    Returns statistics on how many loop iterations tokens use on average,
    which indicates reasoning difficulty and ACT effectiveness.

    Args:
        model    -- OpenMythos model
        prompts  -- unused placeholder (tokenization not built in)
        n_tokens -- number of random tokens to test
        n_loops  -- maximum loop iterations
        device   -- target device

    Returns:
        Dict with mean/max/min halt iteration statistics
    """
    device = device or next(model.parameters()).device
    model.eval()

    vocab_size = model.cfg.vocab_size
    ids = torch.randint(0, vocab_size, (1, n_tokens), device=device)

    with torch.no_grad():
        model(ids, n_loops=n_loops)

    stats = model.get_halt_stats()
    return stats or {"mean_halt_iter": 0.0, "max_halt_iter": 0.0, "min_halt_iter": 0.0}


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def benchmark_all_variants(
    seq_len: int = 64,
    n_loops: int = 4,
    device: Optional[torch.device] = None,
) -> str:
    """
    Run forward-pass benchmark across all model size variants and print a table.

    Args:
        seq_len -- sequence length for benchmark
        n_loops -- recurrent loop depth
        device  -- target device

    Returns:
        Formatted table string
    """
    from open_mythos.variants import mythos_1b, mythos_3b, mythos_10b
    from open_mythos.main import OpenMythos

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = [("1B", mythos_1b), ("3B", mythos_3b), ("10B", mythos_10b)]

    rows = []
    header = f"{'Variant':<8} {'Params(M)':>10} {'Active(M)':>10} {'TPS':>10} {'p50(ms)':>10} {'Mem(MB)':>10}"
    rows.append(header)
    rows.append("-" * len(header))

    for name, cfg_fn in variants:
        try:
            model = OpenMythos(cfg_fn()).to(device)
            result = benchmark_forward(
                model, batch_size=1, seq_len=seq_len, n_loops=n_loops,
                n_warmup=1, n_runs=5, device=device
            )
            rows.append(
                f"{name:<8} {result.total_params_m:>10.1f} {result.active_params_m:>10.1f} "
                f"{result.throughput_tps:>10.1f} {result.latency_p50_ms:>10.2f} "
                f"{result.peak_memory_mb:>10.1f}"
            )
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            rows.append(f"{name:<8} ERROR: {e}")

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Quick benchmark entrypoint
# ---------------------------------------------------------------------------

def run_quick_benchmark(
    model: nn.Module,
    device: Optional[torch.device] = None,
    n_loops: int = 8,
) -> None:
    """
    Run a quick comprehensive benchmark and print results.

    Args:
        model   -- OpenMythos model
        device  -- target device (auto-detected if None)
        n_loops -- recurrent loop depth
    """
    device = device or next(model.parameters()).device
    print("\n" + "=" * 60)
    print("  OpenMythos Quick Benchmark")
    print("=" * 60)

    print("\n[1/4] Forward throughput (B=4, T=128)...")
    fwd = benchmark_forward(model, batch_size=4, seq_len=128, n_loops=n_loops,
                            n_warmup=2, n_runs=10, device=device)
    print(fwd)

    print("[2/4] Generation throughput (prompt=32, gen=64)...")
    gen = benchmark_generate(model, prompt_len=32, gen_len=64, n_loops=n_loops,
                             n_warmup=1, n_runs=3, device=device)
    print(gen)

    print("[3/4] MoE routing entropy...")
    routing = analyze_routing_entropy(model, n_tokens=256, device=device)
    for k, v in routing.items():
        print(f"  {k}: {v:.4f}")

    print("\n[4/4] ACT halting depth...")
    act = analyze_act_depth(model, n_tokens=128, n_loops=n_loops, device=device)
    for k, v in act.items():
        print(f"  {k}: {v:.2f}")

    print("\n" + "=" * 60)
    print(f"  Model size (fp32): {model_memory_mb(model):.1f} MB")
    print(f"  Model size (bf16): {model_memory_mb(model, torch.bfloat16):.1f} MB")
    print("=" * 60 + "\n")
