"""
LoRA Fine-tuning Script for OpenMythos.

Designed to run on free-tier GPUs (Google Colab T4, Kaggle T4/P100).
Supports QLoRA (quantized LoRA) for even lower memory usage.

Usage:
    # Standard LoRA (requires ~16GB VRAM)
    python training/lora_finetune.py --variant 1b --dataset finance

    # QLoRA (requires ~8GB VRAM, fits Colab free T4)
    python training/lora_finetune.py --variant 1b --dataset finance --qlora

    # Custom dataset
    python training/lora_finetune.py --variant 1b --dataset_path ./my_data.jsonl
"""

import argparse
import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_mythos import OpenMythos, MythosConfig
from open_mythos.variants import (
    mythos_1b,
    mythos_3b,
    mythos_10b,
)
from open_mythos.lora import (
    LoRAConfig,
    apply_lora,
    get_lora_params,
    get_lora_param_stats,
    print_lora_summary,
    save_lora_adapter,
    load_lora_adapter,
)
from open_mythos.quantization import quantize_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TextDataset(Dataset):
    """Simple text dataset for language model fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 2048,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Load data
        if data_path.endswith(".jsonl"):
            with open(data_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    text = item.get("text", item.get("content", ""))
                    self.examples.append(text)
        elif data_path.endswith(".json"):
            with open(data_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        text = item.get("text", item.get("content", ""))
                        self.examples.append(text)
        elif data_path.endswith(".txt"):
            with open(data_path, "r") as f:
                text = f.read()
                # Split into chunks
                chunks = [text[i : i + max_length * 4] for i in range(0, len(text), max_length * 4)]
                self.examples.extend(chunks)

        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": attention_mask,
        }


# ---------------------------------------------------------------------------
# Built-in Datasets (for demo / quick start)
# ---------------------------------------------------------------------------


FINANCE_DEMO_DATA = [
    {
        "text": "Analyze XAUUSD price action: Gold is trading at $2,350 with resistance at $2,380 and support at $2,320. RSI indicates overbought conditions on the 4H timeframe. Recommendation: Wait for pullback to support before entering long positions."
    },
    {
        "text": "Business Plan: E-Commerce Platform for Indonesian SMEs. Revenue Model: Transaction fee 2.5% + Premium subscriptions IDR 50K/month. Target Market: 64 million MSMEs in Indonesia. Projected Year 1 Revenue: IDR 2.4 billion."
    },
    {
        "text": "Meta Ads Campaign Optimization: CPM decreased from IDR 15,000 to IDR 8,500 after switching to Advantage+ audience. CTR improved from 1.2% to 2.8%. ROAS: 4.2x. Recommendation: Scale budget 50% and expand to Lookalike audiences."
    },
    {
        "text": "Cashflow Analysis: Monthly revenue IDR 850M, expenses IDR 720M, net profit IDR 130M. Burn rate: 3 months runway at current pace. Action items: Reduce operational costs by 15%, increase marketing ROI by focusing on organic channels."
    },
    {
        "text": "Trading Journal: Entry LONG EUR/USD at 1.0850, Stop Loss 1.0820, Take Profit 1.0920. Risk: 1% of account. Reason: Bullish engulfing on daily chart, MACD crossover, USD weakness from dovish Fed comments."
    },
]


def create_demo_dataset(output_path: str):
    """Create a demo finance dataset for testing."""
    with open(output_path, "w") as f:
        for item in FINANCE_DEMO_DATA:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Created demo dataset at {output_path}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: str,
    max_grad_norm: float = 1.0,
) -> float:
    """Single training step."""
    model.train()

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    # Forward pass
    if scaler is not None:
        with torch.cuda.amp.autocast():
            output = model(input_ids, labels=labels)
            loss = output.loss
    else:
        output = model(input_ids, labels=labels)
        loss = output.loss

    # Backward pass
    optimizer.zero_grad()

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

    return loss.item()


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    max_batches: int = 10,
) -> float:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            output = model(input_ids, labels=labels)
            total_loss += output.loss.item()
            count += 1

    return total_loss / max(count, 1)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune OpenMythos")

    # Model
    parser.add_argument(
        "--variant",
        type=str,
        default="1b",
        choices=["1b", "3b", "10b"],
        help="Model variant to fine-tune",
    )
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="Use QLoRA (quantized LoRA) for lower memory",
    )

    # LoRA
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout")

    # Training
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")

    # Data
    parser.add_argument("--dataset", type=str, default="finance", help="Built-in dataset name")
    parser.add_argument("--dataset_path", type=str, default=None, help="Custom dataset path")

    # Output
    parser.add_argument("--output_dir", type=str, default="./lora_output", help="Output directory")
    parser.add_argument("--save_steps", type=int, default=500, help="Save every N steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every N steps")

    # System
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {device}")
    logger.info(f"Variant: mythos_{args.variant}")
    logger.info(f"QLoRA: {args.qlora}")

    # Create model
    variant_fn = {"1b": mythos_1b, "3b": mythos_3b, "10b": mythos_10b}[args.variant]
    cfg = variant_fn()

    # For demo, use smaller config
    if os.environ.get("DEMO_MODE"):
        cfg.max_seq_len = 512
        cfg.max_loop_iters = 4

    logger.info("Creating model...")
    model = OpenMythos(cfg)

    # Apply QLoRA if requested
    if args.qlora:
        logger.info("Applying INT4 quantization for QLoRA...")
        model = quantize_model(model, bits=4, group_size=128)

    # Apply LoRA
    lora_config = LoRAConfig(
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = apply_lora(model, lora_config)
    print_lora_summary(model)

    # Move to device
    model = model.to(device)

    # Get trainable parameters only
    lora_params = get_lora_params(model)
    trainable_params = list(lora_params.values())

    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Setup optimizer
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Setup dataset
    if args.dataset_path:
        data_path = args.dataset_path
    else:
        # Create demo dataset
        data_path = str(output_dir / "demo_finance.jsonl")
        create_demo_dataset(data_path)

    # Simple tokenizer (for demo; use proper tokenizer in production)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = TextDataset(data_path, tokenizer, max_length=args.max_length)

    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.fp16 and device == "cuda" else None

    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    # Training loop
    logger.info("Starting training...")
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in train_loader:
            loss = train_step(
                model, batch, optimizer, scaler, device, args.max_grad_norm
            )

            epoch_loss += loss
            epoch_steps += 1
            global_step += 1

            scheduler.step()

            # Logging
            if global_step % 10 == 0:
                avg_loss = epoch_loss / epoch_steps
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch+1}/{args.epochs} | "
                    f"Step {global_step}/{total_steps} | "
                    f"Loss: {loss:.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"LR: {lr:.6f}"
                )

            # Evaluation
            if global_step % args.eval_steps == 0:
                val_loss = evaluate(model, val_loader, device)
                logger.info(f"Validation loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_lora_adapter(
                        model,
                        str(output_dir / "best_adapter.pt"),
                        config=lora_config,
                    )

            # Save checkpoint
            if global_step % args.save_steps == 0:
                save_lora_adapter(
                    model,
                    str(output_dir / f"adapter_step_{global_step}.pt"),
                    config=lora_config,
                )

        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        logger.info(
            f"Epoch {epoch+1} completed in {epoch_time:.1f}s | "
            f"Average loss: {avg_epoch_loss:.4f}"
        )

    # Save final adapter
    save_lora_adapter(
        model,
        str(output_dir / "final_adapter.pt"),
        config=lora_config,
    )

    # Save config
    config_dict = {
        "variant": args.variant,
        "lora_rank": args.rank,
        "lora_alpha": args.alpha,
        "lora_dropout": args.dropout,
        "qlora": args.qlora,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "best_val_loss": best_val_loss,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Adapters saved to {output_dir}")


if __name__ == "__main__":
    main()
