"""Measure the mean ACT halt step of a small LM checkpoint.

Feeds a deterministic token batch through the model, captures per-loop
halting probabilities via a forward hook on the recurrent block's ACT head,
and reports the mean halt step plus a per-step histogram. Run it on two
checkpoints with identical settings and diff the means to gate quantization
iterations (Q3 acceptance: |shift| <= 0.1 steps).

CPU smoke:
    python -m examples.measure_act_halt --checkpoint <ckpt> --device cpu
"""

from __future__ import annotations

import argparse

import torch

from examples.generate_small_lm import config_from_checkpoint
from open_mythos import OpenMythos


def halt_steps_from_probabilities(
    probs: list[torch.Tensor], threshold: float
) -> torch.Tensor:
    """Per-position halt step (1-based) from per-loop halting probabilities.

    ``probs[t]`` has shape (B, T); a position halts at the first loop where
    the cumulative probability reaches ``threshold``. Positions that never
    halt get ``len(probs)``.
    """
    cumulative = torch.zeros_like(probs[0])
    steps = torch.full_like(probs[0], len(probs), dtype=torch.long)
    for loop_idx, p in enumerate(probs, start=1):
        hit = (cumulative + p >= threshold) & (steps == len(probs))
        steps = torch.where(hit, torch.full_like(steps, loop_idx), steps)
        cumulative = cumulative + p
    return steps


def measure_halt_steps(
    model: OpenMythos, input_ids: torch.Tensor, n_loops: int
) -> torch.Tensor:
    captured: list[torch.Tensor] = []
    handle = model.recurrent.act.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach())
    )
    try:
        with torch.no_grad():
            model(input_ids, n_loops=n_loops)
    finally:
        handle.remove()
    if len(captured) != n_loops:
        raise ValueError(
            f"captured {len(captured)} halting probabilities, expected {n_loops}"
        )
    return halt_steps_from_probabilities(captured, model.cfg.act_threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--n-loops", type=int, default=0,
                        help="loop depth; 0 uses cfg.max_loop_iters")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = config_from_checkpoint(ckpt)
    model = OpenMythos(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    n_loops = args.n_loops or cfg.max_loop_iters
    torch.manual_seed(args.seed)
    input_ids = torch.randint(
        0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device
    )
    steps = measure_halt_steps(model, input_ids, n_loops)

    print(f"[halt] checkpoint={args.checkpoint}")
    print(f"[halt] n_loops={n_loops} threshold={cfg.act_threshold} positions={steps.numel()}")
    print(f"[halt] mean={steps.float().mean().item():.4f} max={steps.max().item()}")
    histogram = torch.bincount(steps.flatten(), minlength=n_loops)
    for loop_idx, count in enumerate(histogram.tolist(), start=1):
        print(f"[halt] step {loop_idx}: {count}")


if __name__ == "__main__":
    main()
