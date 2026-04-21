"""
Plot training curves + loop-scaling PPL sweep for OpenMythos experiments.

Reads:
    /workspace/runs/looped_8/train.log       (step loss rho_A etc)
    /workspace/runs/baseline_1/train.log
    /workspace/runs/looped_8/eval_ckpt_*.json
    /workspace/runs/baseline_1/eval_ckpt_*.json

Writes:
    figs/training_loss.png
    figs/rho_A.png
    figs/loop_scaling.png
    figs/summary.md
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_log(path: Path):
    rows = []
    with path.open() as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != len(header):
                continue
            row = {}
            for k, v in zip(header, parts):
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return rows


def ema(xs, alpha=0.9):
    out = []
    s = None
    for x in xs:
        s = x if s is None else alpha * s + (1 - alpha) * x
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="/workspace/runs")
    ap.add_argument("--out_dir", type=str, default="/workspace/runs/figs")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = {
        p.name: load_log(p / "train.log")
        for p in sorted(runs_dir.glob("*"))
        if (p / "train.log").exists()
    }
    print(f"loaded runs: {list(runs.keys())}")

    # Training loss
    plt.figure(figsize=(7, 4.5))
    for name, rows in runs.items():
        steps = [r["step"] for r in rows]
        losses = [r["loss"] for r in rows]
        plt.plot(steps, losses, alpha=0.3, color="C0" if "looped" in name else "C1")
        plt.plot(steps, ema(losses), label=name, color="C0" if "looped" in name else "C1", lw=2)
    plt.xlabel("step")
    plt.ylabel("train loss")
    plt.title("OpenMythos training loss — looped vs baseline")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "training_loss.png", dpi=140)
    plt.close()

    # rho_A
    plt.figure(figsize=(7, 4.5))
    for name, rows in runs.items():
        steps = [r["step"] for r in rows]
        rhos = [r["rho_A"] for r in rows]
        plt.plot(steps, rhos, label=name)
    plt.axhline(1.0, color="r", linestyle="--", alpha=0.5, label="instability bound")
    plt.xlabel("step")
    plt.ylabel(r"max element of $A_{\mathrm{discrete}}$  (= $\rho(A)$)")
    plt.title("LTI injection spectral radius over training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "rho_A.png", dpi=140)
    plt.close()

    # Loop scaling sweep
    evals = {}
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        for ev in sorted(run_dir.glob("eval_ckpt_*.json")):
            with ev.open() as f:
                data = json.load(f)
            evals[run_dir.name] = data

    if evals:
        plt.figure(figsize=(7, 4.5))
        for name, data in evals.items():
            sw = data["loop_sweep"]
            xs = [r["n_loops"] for r in sw]
            ys = [r["ppl"] for r in sw]
            trained_at = data["trained_with_max_loop_iters"]
            plt.plot(xs, ys, marker="o", label=f"{name} (trained loops={trained_at})")
            plt.axvline(trained_at, color="gray", linestyle=":", alpha=0.4)
        plt.xlabel("n_loops at inference")
        plt.ylabel("validation perplexity")
        plt.title("Test-time loop scaling — does more compute help?")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "loop_scaling.png", dpi=140)
        plt.close()

    # Summary
    lines = ["# OpenMythos Experiment Summary\n"]
    for name, rows in runs.items():
        if not rows:
            continue
        last = rows[-1]
        lines.append(f"## {name}\n")
        lines.append(f"- final step: {int(last['step'])}")
        lines.append(f"- final loss: {last['loss']:.3f}")
        lines.append(f"- final rho(A): {last['rho_A']:.4f}")
        lines.append(f"- tokens seen: {last['tokens']/1e6:.1f}M")
        lines.append("")
    if evals:
        lines.append("## Loop-scaling sweep\n")
        for name, data in evals.items():
            lines.append(f"### {name}\n")
            lines.append("| n_loops | ppl | nll | rho_A |")
            lines.append("|---|---|---|---|")
            for r in data["loop_sweep"]:
                lines.append(f"| {r['n_loops']} | {r['ppl']:.3f} | {r['nll']:.4f} | {r['rho_A']:.4f} |")
            lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines))
    print(f"==> wrote figs to {out_dir}")


if __name__ == "__main__":
    main()
