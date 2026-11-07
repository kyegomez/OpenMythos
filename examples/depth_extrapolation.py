"""Depth-extrapolation experiment: does inference-time loop count buy accuracy?

Pre-registered design: docs/depth_extrapolation_experiment.md. Trains a
tiny-vocab OpenMythos on an algorithmic task (carry-chain addition or parity
scan) with a fixed training loop depth, then sweeps inference-time n_loops
and reports exact-match accuracy overall and per difficulty bucket.

Train (MPS):
    python -m examples.depth_extrapolation train --task add --loops-train 4 \
        --steps 4000 --device mps --out checkpoints/depth_add_l4.pt
Sweep:
    python -m examples.depth_extrapolation eval --task add \
        --checkpoint checkpoints/depth_add_l4.pt --loops 2,4,8,16,32
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from examples.train_small_lm import compute_loss
from open_mythos import MythosConfig, OpenMythos

DIGIT_CHARS = "0123456789"
CHAR_TO_ID = {c: i for i, c in enumerate(DIGIT_CHARS + "+=")}
ID_TO_CHAR = {i: c for c, i in CHAR_TO_ID.items()}
EOS_ID = len(CHAR_TO_ID)
VOCAB_SIZE = len(CHAR_TO_ID) + 1


@dataclass(frozen=True)
class TaskInstance:
    input_text: str
    answer_text: str
    bucket: str


def char_id(char: str) -> int:
    return CHAR_TO_ID[char]


def id_char(idx: int) -> str:
    return ID_TO_CHAR.get(idx, "")


def make_addition_instance(rng: torch.Generator, max_digits: int) -> TaskInstance:
    def draw() -> int:
        # Uniform digit count: skew toward 4-digit gives ~90% one bucket.
        digits = int(torch.randint(1, max_digits + 1, (1,), generator=rng).item())
        low = 1 if digits == 1 else 10 ** (digits - 1)
        return int(torch.randint(low, 10**digits, (1,), generator=rng).item())

    a, b = draw(), draw()
    return TaskInstance(
        input_text=f"{a}+{b}=",
        answer_text=str(a + b)[::-1],
        bucket=f"{max(len(str(a)), len(str(b)))}-digit",
    )


def parse_addition(input_text: str) -> tuple[int, int] | None:
    body = input_text.rstrip("=")
    left, sep, right = body.partition("+")
    if not sep or not left.isdigit() or not right.isdigit():
        return None
    return int(left), int(right)


def make_parity_instance(rng: torch.Generator, max_bits: int) -> TaskInstance:
    # Quantized lengths keep difficulty buckets statistically meaningful.
    step = max(1, max_bits // 4)
    length = int(torch.randint(1, 5, (1,), generator=rng).item()) * step
    bits = "".join(str(int(torch.randint(0, 2, (1,), generator=rng).item())) for _ in range(length))
    parity = bits.count("1") % 2
    return TaskInstance(
        input_text=f"{bits}=",
        answer_text=str(parity),
        bucket=f"{length}b",
    )


def encode_instance(
    instance: TaskInstance, seq_len: int, eos_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = [char_id(c) for c in instance.input_text]
    first_answer = len(ids)
    ids += [char_id(c) for c in instance.answer_text] + [eos_id]
    answer_end = len(ids)  # last trained target = the single terminating EOS
    if len(ids) > seq_len + 1:
        raise ValueError(
            f"instance needs {len(ids)} positions > seq_len+1={seq_len + 1}"
        )
    ids += [eos_id] * (seq_len + 1 - len(ids))
    total = torch.tensor(ids, dtype=torch.long)
    mask = torch.zeros(seq_len, dtype=torch.float32)
    mask[first_answer - 1 : answer_end - 1] = 1.0
    return total[:-1], total[1:], mask


def build_batch(
    rng: torch.Generator,
    task: str,
    batch_size: int,
    seq_len: int,
    max_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    make = make_addition_instance if task == "add" else make_parity_instance
    xs, ys, masks = [], [], []
    for _ in range(batch_size):
        x, y, mask = encode_instance(make(rng, max_size), seq_len, EOS_ID)
        xs.append(x)
        ys.append(y)
        masks.append(mask)
    return torch.stack(xs), torch.stack(ys), torch.stack(masks)


def decode_answer(ids: list[int], eos_id: int) -> str:
    chars = []
    for idx in ids:
        if idx == eos_id:
            break
        chars.append(id_char(idx))
    return "".join(chars)


def check_prediction(prediction: str, expected: str) -> bool:
    return prediction == expected


def build_model(loops_train: int, seq_len: int) -> OpenMythos:
    cfg = MythosConfig(
        vocab_size=VOCAB_SIZE,
        dim=512,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=seq_len + 1,
        max_loop_iters=loops_train,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=512,
        lora_rank=8,
        dropout=0.0,
    )
    return OpenMythos(cfg)


def train_model(
    task: str,
    steps: int,
    batch_size: int,
    seq_len: int,
    max_size: int,
    loops_train: int,
    lr: float,
    device: torch.device,
    seed: int,
    log_every: int,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    model = build_model(loops_train, seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    start = time.perf_counter()
    for step in range(1, steps + 1):
        x, y, mask = build_batch(rng, task, batch_size, seq_len, max_size)
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = compute_loss(logits, y, VOCAB_SIZE, None, mask)
        loss.backward()
        optimizer.step()
        if step == 1 or step % log_every == 0:
            elapsed = time.perf_counter() - start
            print(
                f"step {step:5d} | loss {float(loss.item()):.4f} | "
                f"tok/s {(step * batch_size * seq_len) / max(elapsed, 1e-9):,.0f}"
            )
    return model


@torch.no_grad()
def greedy_answer(
    model: OpenMythos,
    input_text: str,
    loops: int,
    max_new_tokens: int,
    device: torch.device,
    eos_id: int = EOS_ID,
) -> str:
    was_training = model.training
    model.eval()
    ids = [char_id(c) for c in input_text]
    context = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(context, n_loops=loops)
        nxt = int(torch.argmax(logits[0, -1]).item())
        if nxt == eos_id:
            break
        ids.append(nxt)
        context = torch.tensor([ids], dtype=torch.long, device=device)
    if was_training:
        model.train()
    return decode_answer(ids[len(input_text) :], eos_id)


@torch.no_grad()
def eval_sweep(
    model: OpenMythos,
    task: str,
    loops_list: list[int],
    n_eval: int,
    max_size: int,
    max_new_tokens: int,
    seed: int,
    device: torch.device = torch.device("cpu"),
) -> list[dict]:
    make = make_addition_instance if task == "add" else make_parity_instance
    rows = []
    for loops in loops_list:
        rng = torch.Generator().manual_seed(seed)  # same test set for every T
        correct = 0
        buckets: dict[str, list[bool]] = {}
        for _ in range(n_eval):
            instance = make(rng, max_size)
            prediction = greedy_answer(model, instance.input_text, loops, max_new_tokens, device)
            hit = check_prediction(prediction, instance.answer_text)
            correct += hit
            buckets.setdefault(instance.bucket, []).append(hit)
        rows.append(
            {
                "loops": loops,
                "accuracy": correct / max(1, n_eval),
                "by_bucket": {
                    key: sum(hits) / len(hits) for key, hits in sorted(buckets.items())
                },
            }
        )
    return rows


def load_eval_model(checkpoint: str, device: torch.device) -> OpenMythos:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = MythosConfig(**ckpt["config"])
    model = OpenMythos(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--task", choices=["add", "parity"], required=True)
    train_p.add_argument("--steps", type=int, default=4000)
    train_p.add_argument("--batch-size", type=int, default=64)
    train_p.add_argument("--seq-len", type=int, default=48)
    train_p.add_argument("--max-size", type=int, default=4)
    train_p.add_argument("--loops-train", type=int, default=4)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--device", default="cpu")
    train_p.add_argument("--seed", type=int, default=7)
    train_p.add_argument("--log-every", type=int, default=100)
    train_p.add_argument("--out", required=True)

    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--task", choices=["add", "parity"], required=True)
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--loops", default="2,4,8,16,32")
    eval_p.add_argument("--n-eval", type=int, default=300)
    eval_p.add_argument("--max-size", type=int, default=4)
    eval_p.add_argument("--max-new-tokens", type=int, default=8)
    eval_p.add_argument("--device", default="cpu")
    eval_p.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()
    if args.mode == "train":
        device = torch.device(args.device)
        model = train_model(
            args.task,
            args.steps,
            args.batch_size,
            args.seq_len,
            args.max_size,
            args.loops_train,
            args.lr,
            device,
            args.seed,
            args.log_every,
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "config": {
                    "vocab_size": VOCAB_SIZE,
                    "dim": 512,
                    "n_heads": 8,
                    "n_kv_heads": 2,
                    "max_seq_len": args.seq_len + 1,
                    "max_loop_iters": args.loops_train,
                    "prelude_layers": 1,
                    "coda_layers": 1,
                    "attn_type": "gqa",
                    "n_experts": 4,
                    "n_shared_experts": 1,
                    "n_experts_per_tok": 2,
                    "expert_dim": 512,
                    "lora_rank": 8,
                    "dropout": 0.0,
                },
            },
            out,
        )
        print(f"[saved] {out}")
        return

    device = torch.device(args.device)
    model = load_eval_model(args.checkpoint, device)
    loops_list = [int(part) for part in args.loops.split(",")]
    rows = eval_sweep(
        model,
        args.task,
        loops_list,
        args.n_eval,
        args.max_size,
        args.max_new_tokens,
        args.seed,
        device,
    )
    print(f"[sweep] task={args.task} checkpoint={args.checkpoint}")
    for row in rows:
        buckets = " ".join(
            f"{bucket}:{acc:.2f}" for bucket, acc in row["by_bucket"].items()
        )
        print(f"[sweep] loops={row['loops']:3d} acc={row['accuracy']:.3f} | {buckets}")


if __name__ == "__main__":
    main()
