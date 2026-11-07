"""Train a tiny OpenMythos checkpoint on an offline byte-level corpus.

This is a CPU-friendly end-to-end example: train, evaluate, save a checkpoint,
then use ``examples/generate_tiny_mythos.py`` to load it for generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from open_mythos import MythosConfig, OpenMythos


CORPUS = """
OpenMythos is a recurrent-depth transformer. It reads a prompt, loops over a
shared recurrent block, and turns the final hidden state into next-token
predictions. Tiny training runs are not useful language models, but they are
excellent wiring tests: data moves through the model, gradients flow, loss
falls, checkpoints save, and generation can reload the learned weights.

The model learns byte by byte. The alphabet is simple, the corpus is small, and
the point is clarity. A larger run can swap this built-in text for a real
tokenizer and dataset while keeping the same train-save-load shape.
"""


CONFIG_KWARGS = {
    "vocab_size": 256,
    "dim": 128,
    "n_heads": 4,
    "n_kv_heads": 2,
    "max_seq_len": 128,
    "max_loop_iters": 4,
    "prelude_layers": 1,
    "coda_layers": 1,
    "attn_type": "gqa",
    "n_experts": 4,
    "n_shared_experts": 1,
    "n_experts_per_tok": 2,
    "expert_dim": 128,
    "lora_rank": 8,
    "dropout": 0.0,
}


class ByteTokenizer:
    vocab_size = 256

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(token_ids: list[int]) -> str:
        data = bytes(max(0, min(255, int(t))) for t in token_ids)
        return data.decode("utf-8", errors="replace")


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = data.numel() - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in starts])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(
    model: OpenMythos,
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    n_loops: int,
    batches: int = 8,
) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, batch_size, seq_len, device)
        logits = model(x, n_loops=n_loops)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-loops", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--out", default="checkpoints/tiny_mythos.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tokenizer = ByteTokenizer()
    tokens = torch.tensor(tokenizer.encode(CORPUS.strip() + "\n") * 256)
    split = int(tokens.numel() * 0.9)
    train_data = tokens[:split]
    val_data = tokens[split:]

    cfg = MythosConfig(**CONFIG_KWARGS)
    if args.seq_len >= cfg.max_seq_len:
        raise ValueError(f"--seq-len must be < max_seq_len ({cfg.max_seq_len})")

    model = OpenMythos(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    params = sum(p.numel() for p in model.parameters())
    print(f"[setup] device={device} params={params:,} tokens={tokens.numel():,}")

    model.train()
    last_loss = 0.0
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.batch_size, args.seq_len, device)
        logits = model(x, n_loops=args.n_loops)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

        if step == 1 or step % args.log_every == 0:
            print(f"step {step:4d} | train loss {last_loss:.4f}")
        if args.eval_every > 0 and step % args.eval_every == 0:
            val_loss = evaluate(
                model, val_data, args.batch_size, args.seq_len, device, args.n_loops
            )
            print(f"          | val loss   {val_loss:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": CONFIG_KWARGS,
            "tokenizer": "byte",
            "step": args.steps,
            "loss": last_loss,
        },
        out,
    )
    print(f"[saved] {out} (step={args.steps}, loss={last_loss:.4f})")


if __name__ == "__main__":
    main()
