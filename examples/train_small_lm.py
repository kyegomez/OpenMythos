"""Train a small OpenMythos language model with a Hugging Face tokenizer.

This is the practical next step after the tiny byte-level smoke example:
it trains OpenMythos itself on a real text dataset, saves a reusable
checkpoint, and pairs with ``examples/generate_small_lm.py`` for inference.

CPU smoke:
    python -m examples.train_small_lm --steps 20 --batch-size 4 --seq-len 64

Longer local run:
    python -m examples.train_small_lm --steps 2000 --device cuda
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from examples.small_lm_data import (
    StoryAlignedDataset,
    balance_names,
    collect_stories,
    dedup_stories,
)
from examples.small_lm_text import clean_text_artifacts
from open_mythos import MythosConfig, OpenMythos
from open_mythos.config import QuantConfig

@dataclass(frozen=True)
class WeightLoadResult:
    loaded: int
    skipped: list[str]


class NoOpScaler:
    """Keep the train step identical when AMP is disabled."""

    @staticmethod
    def scale(loss: torch.Tensor) -> torch.Tensor:
        return loss

    @staticmethod
    def step(optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    @staticmethod
    def update() -> None:
        return None

    @staticmethod
    def state_dict() -> dict[str, Any]:
        return {}

    @staticmethod
    def load_state_dict(_state: dict[str, Any]) -> None:
        return None


class PackedTextDataset(Dataset):
    """Materialize streamed text into fixed-length next-token pairs."""

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        seq_len: int,
        max_tokens: int,
        text_field: str,
        eos_token_id: int | None,
        append_eos: bool,
        clean_text: bool = True,
        name_balance_seed: int | None = None,
    ) -> None:
        if append_eos and eos_token_id is None:
            raise ValueError(
                "append_eos=True but tokenizer has no eos_token_id; "
                "pass --no-append-eos or choose a tokenizer with EOS"
            )
        buf: list[int] = []
        for sample in hf_dataset:
            text = sample[text_field]
            if isinstance(text, str) and text.strip():
                if name_balance_seed is not None:
                    text = balance_names(text, seed=name_balance_seed)
                if clean_text:
                    text = clean_text_artifacts(text)
                ids = tokenizer.encode(text, add_special_tokens=False)
                if append_eos and eos_token_id is not None:
                    ids.append(eos_token_id)
                buf.extend(ids)
            if len(buf) >= max_tokens:
                break
        if len(buf) <= seq_len:
            raise ValueError(
                f"dataset produced only {len(buf)} tokens; need > seq_len={seq_len}"
            )
        n_pairs = max(1, (len(buf) - 1) // seq_len)
        self.seq_len = seq_len
        self.data = torch.tensor(buf[: n_pairs * seq_len + 1], dtype=torch.long)

    def __len__(self) -> int:
        return (self.data.numel() - 1) // self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def load_text_dataset(name: str, config: str, split: str):
    if config:
        return load_dataset(name, config, split=split, streaming=True)
    return load_dataset(name, split=split, streaming=True)


def build_scaler(use_amp: bool):
    if not use_amp:
        return NoOpScaler()
    amp_scaler = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if amp_scaler is not None:
        return amp_scaler("cuda", enabled=True)
    torch_scaler = getattr(torch, "GradScaler", None)
    if torch_scaler is not None:
        return torch_scaler("cuda", enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def autocast_context(use_amp: bool):
    if not use_amp:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


MODEL_SIZES: dict[str, dict[str, int]] = {
    # ~15.8M params (v0-v8 lineage): embed-dominated, layers ~2.1M.
    "small": dict(dim=256, n_heads=8, n_kv_heads=2, expert_dim=256),
    # ~38M params (T3b scaling): layers grow ~4x, escape the capacity ceiling.
    "mid": dict(dim=512, n_heads=8, n_kv_heads=2, expert_dim=512),
}


def build_config(
    vocab_size: int, max_seq_len: int, model_size: str = "small"
) -> MythosConfig:
    try:
        size = MODEL_SIZES[model_size]
    except KeyError:
        raise ValueError(
            f"unknown model size {model_size!r}; "
            f"expected one of {sorted(MODEL_SIZES)}"
        ) from None
    return MythosConfig(
        vocab_size=vocab_size,
        dim=size["dim"],
        n_heads=size["n_heads"],
        n_kv_heads=size["n_kv_heads"],
        max_seq_len=max_seq_len,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=8,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=size["expert_dim"],
        lora_rank=8,
        dropout=0.0,
    )


def config_to_dict(cfg: MythosConfig) -> dict[str, int | float | str]:
    return {
        "vocab_size": cfg.vocab_size,
        "dim": cfg.dim,
        "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads,
        "max_seq_len": cfg.max_seq_len,
        "max_loop_iters": cfg.max_loop_iters,
        "prelude_layers": cfg.prelude_layers,
        "coda_layers": cfg.coda_layers,
        "attn_type": cfg.attn_type,
        "n_experts": cfg.n_experts,
        "n_shared_experts": cfg.n_shared_experts,
        "n_experts_per_tok": cfg.n_experts_per_tok,
        "expert_dim": cfg.expert_dim,
        "lora_rank": cfg.lora_rank,
        "dropout": cfg.dropout,
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    cfg: MythosConfig,
    tokenizer: str,
    step: int,
    loss: float,
    append_eos: bool,
    eos_loss_weight: float,
    clean_text: bool = True,
    packing: str = "stream",
    pair_mix: float = 0.0,
    dedup_8gram: bool = False,
    balance_names: bool = False,
    quant_mode: str = "none",
    quant_group_size: int = 128,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config_to_dict(cfg),
            "tokenizer": tokenizer,
            "step": step,
            "loss": loss,
            "append_eos": append_eos,
            "eos_loss_weight": eos_loss_weight,
            "clean_text": clean_text,
            "packing": packing,
            "pair_mix": pair_mix,
            "dedup_8gram": dedup_8gram,
            "balance_names": balance_names,
            "quant_mode": quant_mode,
            "quant_group_size": quant_group_size,
        },
        path,
    )
    print(f"[saved] {path} (step={step}, loss={loss:.4f})")


def train_stats(train_ds, packing: str) -> str:
    """Describe the train split; story datasets have no single token buffer."""
    if packing == "story":
        return f"train windows={len(train_ds):,}"
    return f"train tokens={train_ds.data.numel():,} pairs={len(train_ds):,}"


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
    class_weights: torch.Tensor | None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy with optional per-token mask (pair/pad targets excluded).

    With ``mask=None`` or all-ones this equals plain weighted
    ``F.cross_entropy`` mean reduction exactly.
    """
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    if mask is None:
        return F.cross_entropy(flat_logits, flat_targets, weight=class_weights)
    per_token = F.cross_entropy(
        flat_logits, flat_targets, weight=class_weights, reduction="none"
    )
    flat_mask = mask.reshape(-1).to(per_token.dtype)
    if class_weights is not None:
        target_weights = class_weights[flat_targets]
    else:
        target_weights = torch.ones_like(per_token)
    return (per_token * flat_mask).sum() / (flat_mask * target_weights).sum()


def build_loss_weights(
    vocab_size: int,
    eos_token_id: int | None,
    eos_loss_weight: float,
    device: torch.device,
) -> torch.Tensor | None:
    if eos_loss_weight <= 0:
        raise ValueError(f"--eos-loss-weight must be positive, got {eos_loss_weight}")
    if eos_loss_weight == 1.0:
        return None
    if eos_token_id is None:
        raise ValueError("--eos-loss-weight requires a tokenizer with eos_token_id")
    if eos_token_id < 0 or eos_token_id >= vocab_size:
        raise ValueError(
            f"eos_token_id={eos_token_id} is outside vocab_size={vocab_size}"
        )
    weights = torch.ones(vocab_size, dtype=torch.float32, device=device)
    weights[eos_token_id] = eos_loss_weight
    return weights


def validate_init_from_config(
    source_config: dict[str, Any],
    target_cfg: MythosConfig,
) -> None:
    target_config = config_to_dict(target_cfg)
    mismatches = [
        key
        for key, value in source_config.items()
        if key != "max_seq_len" and target_config.get(key) != value
    ]
    if mismatches:
        fields = ", ".join(sorted(mismatches))
        raise ValueError(f"init-from checkpoint config mismatch: {fields}")


def load_matching_model_weights(
    model: torch.nn.Module,
    source_state: dict[str, torch.Tensor],
) -> WeightLoadResult:
    target_state = model.state_dict()
    loadable = {
        key: value
        for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    skipped = sorted(key for key in source_state if key not in loadable)
    model.load_state_dict(loadable, strict=False)
    return WeightLoadResult(loaded=len(loadable), skipped=skipped)


@torch.no_grad()
def evaluate(
    model: OpenMythos,
    loader: DataLoader,
    device: torch.device,
    vocab_size: int,
    n_loops: int,
    batches: int,
) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= batches:
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x, n_loops=n_loops)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--train-tokens", type=int, default=500_000)
    parser.add_argument("--eval-tokens", type=int, default=50_000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--model-size",
        choices=sorted(MODEL_SIZES),
        default="small",
        help="model width preset (small ~15.8M, mid ~38M params)",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-loops", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--init-from",
        default="",
        help=(
            "warm-start model weights from a checkpoint without loading optimizer; "
            "allows max_seq_len changes"
        ),
    )
    parser.add_argument(
        "--no-append-eos",
        action="store_true",
        help="do not append tokenizer.eos_token_id after each text sample",
    )
    parser.add_argument(
        "--eos-loss-weight",
        type=float,
        default=1.0,
        help="multiply cross-entropy loss for EOS targets; >1 encourages natural stops",
    )
    parser.add_argument(
        "--no-clean-text",
        action="store_true",
        help="disable deterministic cleanup of mojibake punctuation before tokenizing",
    )
    parser.add_argument(
        "--packing",
        choices=["stream", "story"],
        default="stream",
        help=(
            "stream: continuous token packing (legacy); story: one story per "
            "window starting at a story boundary"
        ),
    )
    parser.add_argument(
        "--pair-mix",
        type=float,
        default=0.0,
        help=(
            "fraction of story windows whose opening targets are masked "
            "(continuation-pair training); requires --packing story"
        ),
    )
    parser.add_argument(
        "--dedup-8gram",
        action="store_true",
        help=(
            "drop template stories (exact duplicates or shared 8-token "
            "openings); requires --packing story"
        ),
    )
    parser.add_argument(
        "--balance-names",
        action="store_true",
        help=(
            "rotate dominant TinyStories names (Lily/Tom/Ben/Timmy) through a "
            "balanced pool to counter character hijacking; requires --packing "
            "story"
        ),
    )
    parser.add_argument(
        "--quant",
        choices=["none", "w158"],
        default="none",
        help=(
            "weight quantization: w158 = BitNet b1.58 ternary weights with "
            "straight-through QAT (BitLinear on attention/experts/FFN; "
            "embeddings, head, router, and LoRA stay full precision)"
        ),
    )
    parser.add_argument(
        "--quant-group-size",
        type=int,
        default=128,
        help="activation-quant group width reserved for the A8 stage",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="use CUDA autocast + GradScaler when training on cuda",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--out", default="checkpoints/small_mythos.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.resume and args.init_from:
        raise ValueError("--resume and --init-from are mutually exclusive")

    print(
        f"[setup] device={device} batch={args.batch_size} "
        f"seq_len={args.seq_len} steps={args.steps}"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = len(tokenizer)
    cfg = build_config(vocab_size, args.seq_len, args.model_size)
    cfg = replace(
        cfg, quant=QuantConfig(mode=args.quant, group_size=args.quant_group_size)
    )
    append_eos = not args.no_append_eos
    clean_text = not args.no_clean_text
    packing = args.packing
    pair_mix = args.pair_mix
    if packing != "story" and (pair_mix != 0.0 or args.dedup_8gram):
        raise ValueError("--pair-mix/--dedup-8gram require --packing story")
    if not 0.0 <= pair_mix <= 1.0:
        raise ValueError(f"--pair-mix must be within [0, 1], got {pair_mix}")
    if not append_eos and args.eos_loss_weight != 1.0:
        raise ValueError("--eos-loss-weight > 1 requires EOS labels; remove --no-append-eos")
    loss_weights = build_loss_weights(
        vocab_size,
        tokenizer.eos_token_id,
        args.eos_loss_weight,
        device,
    )

    print(f"[setup] tokenizer={args.tokenizer} vocab_size={vocab_size:,}")
    print(
        f"[setup] dataset={args.dataset} config={args.dataset_config or 'none'} "
        f"append_eos={append_eos} eos_loss_weight={args.eos_loss_weight} "
        f"clean_text={clean_text} packing={packing} pair_mix={pair_mix} "
        f"dedup_8gram={args.dedup_8gram} balance_names={args.balance_names} "
        f"quant={args.quant}"
    )
    raw_train = load_text_dataset(args.dataset, args.dataset_config, args.train_split)
    raw_eval = load_text_dataset(args.dataset, args.dataset_config, args.eval_split)
    if packing == "story":
        stories = collect_stories(
            raw_train,
            tokenizer,
            args.train_tokens,
            args.text_field,
            tokenizer.eos_token_id,
            append_eos,
            clean_text=clean_text,
            name_balance_seed=args.seed if args.balance_names else None,
        )
        stories, dropped = dedup_stories(stories, enabled=args.dedup_8gram)
        if args.dedup_8gram:
            print(f"[setup] dedup-8gram dropped {dropped} template stories")
        train_ds = StoryAlignedDataset(
            stories, args.seq_len, pair_mix, tokenizer.eos_token_id
        )
    else:
        train_ds = PackedTextDataset(
            raw_train,
            tokenizer,
            args.seq_len,
            args.train_tokens,
            args.text_field,
            tokenizer.eos_token_id,
            append_eos,
            clean_text=clean_text,
            name_balance_seed=args.seed if args.balance_names else None,
        )
    eval_ds = PackedTextDataset(
        raw_eval,
        tokenizer,
        args.seq_len,
        args.eval_tokens,
        args.text_field,
        tokenizer.eos_token_id,
        append_eos,
        clean_text=clean_text,
    )
    print(
        f"[setup] {train_stats(train_ds, packing)} | "
        f"eval tokens={eval_ds.data.numel():,} pairs={len(eval_ds):,}"
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
    data_iter = iter(train_loader)

    model = OpenMythos(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = build_scaler(use_amp)
    start_step = 0
    last_loss = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ckpt["config"] != config_to_dict(cfg):
            raise ValueError("resume checkpoint config does not match current run")
        if "append_eos" in ckpt and bool(ckpt["append_eos"]) != append_eos:
            raise ValueError("resume checkpoint append_eos does not match current run")
        if "clean_text" in ckpt and bool(ckpt["clean_text"]) != clean_text:
            raise ValueError("resume checkpoint clean_text does not match current run")
        if (
            "eos_loss_weight" in ckpt
            and float(ckpt["eos_loss_weight"]) != args.eos_loss_weight
        ):
            raise ValueError(
                "resume checkpoint eos_loss_weight does not match current run"
            )
        if "packing" in ckpt and str(ckpt["packing"]) != packing:
            raise ValueError("resume checkpoint packing does not match current run")
        if "pair_mix" in ckpt and float(ckpt["pair_mix"]) != pair_mix:
            raise ValueError("resume checkpoint pair_mix does not match current run")
        if "dedup_8gram" in ckpt and bool(ckpt["dedup_8gram"]) != args.dedup_8gram:
            raise ValueError("resume checkpoint dedup_8gram does not match current run")
        if "balance_names" in ckpt and bool(ckpt["balance_names"]) != args.balance_names:
            raise ValueError(
                "resume checkpoint balance_names does not match current run"
            )
        if "quant_mode" in ckpt and str(ckpt["quant_mode"]) != args.quant:
            raise ValueError("resume checkpoint quant_mode does not match current run")
        if (
            "quant_group_size" in ckpt
            and int(ckpt["quant_group_size"]) != args.quant_group_size
        ):
            raise ValueError(
                "resume checkpoint quant_group_size does not match current run"
            )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("step", 0))
        last_loss = float(ckpt.get("loss", 0.0))
        print(f"[resume] {args.resume} at step={start_step} loss={last_loss:.4f}")
    elif args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        validate_init_from_config(ckpt["config"], cfg)
        if "append_eos" in ckpt and bool(ckpt["append_eos"]) != append_eos:
            raise ValueError("init-from checkpoint append_eos does not match current run")
        if "clean_text" in ckpt and bool(ckpt["clean_text"]) != clean_text:
            raise ValueError("init-from checkpoint clean_text does not match current run")
        if (
            "eos_loss_weight" in ckpt
            and float(ckpt["eos_loss_weight"]) != args.eos_loss_weight
        ):
            raise ValueError(
                "init-from checkpoint eos_loss_weight does not match current run"
            )
        if "packing" in ckpt and str(ckpt["packing"]) != packing:
            raise ValueError("init-from checkpoint packing does not match current run")
        if "pair_mix" in ckpt and float(ckpt["pair_mix"]) != pair_mix:
            raise ValueError("init-from checkpoint pair_mix does not match current run")
        if "dedup_8gram" in ckpt and bool(ckpt["dedup_8gram"]) != args.dedup_8gram:
            raise ValueError(
                "init-from checkpoint dedup_8gram does not match current run"
            )
        if "balance_names" in ckpt and bool(ckpt["balance_names"]) != args.balance_names:
            raise ValueError(
                "init-from checkpoint balance_names does not match current run"
            )
        if "quant_mode" in ckpt and str(ckpt["quant_mode"]) != args.quant:
            raise ValueError(
                "init-from checkpoint quant_mode does not match current run"
            )
        if (
            "quant_group_size" in ckpt
            and int(ckpt["quant_group_size"]) != args.quant_group_size
        ):
            raise ValueError(
                "init-from checkpoint quant_group_size does not match current run"
            )
        result = load_matching_model_weights(model, ckpt["model"])
        start_step = int(ckpt.get("step", 0))
        last_loss = float(ckpt.get("loss", 0.0))
        skipped = ", ".join(result.skipped) if result.skipped else "none"
        print(
            f"[init-from] {args.init_from} at step={start_step} "
            f"loss={last_loss:.4f}; loaded={result.loaded} skipped={skipped}"
        )
    params = sum(p.numel() for p in model.parameters())
    print(f"[setup] OpenMythos params={params:,} amp={use_amp}")

    model.train()
    start_time = time.perf_counter()
    out = Path(args.out)
    final_step = start_step + args.steps
    last_saved_step = -1
    for step in range(start_step + 1, final_step + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        x, y = batch[0], batch[1]
        mask = batch[2] if len(batch) == 3 else None
        x = x.to(device)
        y = y.to(device)
        if mask is not None:
            mask = mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(use_amp):
            logits = model(x, n_loops=args.n_loops)
            loss = compute_loss(logits, y, vocab_size, loss_weights, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        last_loss = float(loss.item())

        if step == 1 or step % args.log_every == 0:
            elapsed = time.perf_counter() - start_time
            completed = step - start_step
            tok_s = (completed * x.numel()) / max(elapsed, 1e-9)
            print(f"step {step:5d} | train loss {last_loss:.4f} | tok/s {tok_s:,.0f}")
        if args.eval_every > 0 and step % args.eval_every == 0:
            val_loss = evaluate(
                model,
                eval_loader,
                device,
                vocab_size,
                args.n_loops,
                args.eval_batches,
            )
            print(f"           | val loss   {val_loss:.4f}")
        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(
                out,
                model,
                optimizer,
                scaler,
                cfg,
                args.tokenizer,
                step,
                last_loss,
                append_eos,
                args.eos_loss_weight,
                clean_text,
                packing,
                pair_mix,
                args.dedup_8gram,
                args.balance_names,
                args.quant,
                args.quant_group_size,
            )
            last_saved_step = step

    if last_saved_step != final_step:
        save_checkpoint(
            out,
            model,
            optimizer,
            scaler,
            cfg,
            args.tokenizer,
            final_step,
            last_loss,
            append_eos,
            args.eos_loss_weight,
            clean_text,
            packing,
            pair_mix,
            args.dedup_8gram,
            args.balance_names,
            args.quant,
            args.quant_group_size,
        )


if __name__ == "__main__":
    main()
