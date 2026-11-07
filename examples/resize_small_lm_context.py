"""Resize a small LM checkpoint context length for generation experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from examples.train_small_lm import load_matching_model_weights
from open_mythos import MythosConfig, OpenMythos


@dataclass(frozen=True)
class ResizeResult:
    source_max_seq_len: int
    target_max_seq_len: int
    loaded: int
    skipped: list[str]


def build_resized_config(
    source_config: dict[str, Any],
    max_seq_len: int,
) -> dict[str, Any]:
    source_max_seq_len = int(source_config["max_seq_len"])
    if max_seq_len <= source_max_seq_len:
        raise ValueError(
            f"--max-seq-len must be greater than source max_seq_len="
            f"{source_max_seq_len}, got {max_seq_len}"
        )
    target_config = dict(source_config)
    target_config["max_seq_len"] = max_seq_len
    return target_config


def resize_checkpoint_context(
    source: str | Path,
    target: str | Path,
    max_seq_len: int,
) -> ResizeResult:
    source_path = Path(source)
    target_path = Path(target)
    ckpt = torch.load(source_path, map_location="cpu", weights_only=False)
    source_config = ckpt["config"]
    target_config = build_resized_config(source_config, max_seq_len)

    model = OpenMythos(MythosConfig(**target_config))
    load_result = load_matching_model_weights(model, ckpt["model"])
    resized = {
        key: value for key, value in ckpt.items() if key not in {"model", "optimizer", "scaler"}
    }
    resized["model"] = model.state_dict()
    resized["optimizer"] = {}
    resized["scaler"] = {}
    resized["config"] = target_config
    result = ResizeResult(
        source_max_seq_len=int(source_config["max_seq_len"]),
        target_max_seq_len=max_seq_len,
        loaded=load_result.loaded,
        skipped=load_result.skipped,
    )
    resized["context_resize"] = {
        "source_max_seq_len": result.source_max_seq_len,
        "target_max_seq_len": result.target_max_seq_len,
        "loaded": result.loaded,
        "skipped": result.skipped,
    }

    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(resized, target_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-seq-len", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = resize_checkpoint_context(
        args.checkpoint,
        args.out,
        max_seq_len=args.max_seq_len,
    )
    skipped = ", ".join(result.skipped) if result.skipped else "none"
    print(
        f"[resized] {args.checkpoint} -> {args.out} "
        f"max_seq_len={result.source_max_seq_len}->{result.target_max_seq_len} "
        f"loaded={result.loaded} skipped={skipped}"
    )


if __name__ == "__main__":
    main()
