"""Load a tiny OpenMythos checkpoint and generate byte-level text."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from open_mythos import MythosConfig, OpenMythos
from examples.train_tiny_mythos import ByteTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/tiny_mythos.pt")
    parser.add_argument("--prompt", default="OpenMythos ")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--n-loops", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    path = Path(args.checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run examples/train_tiny_mythos.py first."
        )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = MythosConfig(**ckpt["config"])
    model = OpenMythos(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    tokenizer = ByteTokenizer()
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("--prompt must encode to at least one byte")
    max_prompt_len = max(1, cfg.max_seq_len - 1)
    prompt_ids = prompt_ids[-max_prompt_len:]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            n_loops=args.n_loops,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    text = tokenizer.decode(out[0].tolist())
    print(f"[loaded] {path} step={ckpt.get('step')} loss={ckpt.get('loss'):.4f}")
    print(text)


if __name__ == "__main__":
    main()
