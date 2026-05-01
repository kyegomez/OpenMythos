"""
OpenMythos Data Preparation — Download, tokenize, and save as .npy token file.

Usage:
  python prepare_data.py                          # default: wikitext-2, gpt2 tokenizer
  python prepare_data.py --dataset wikitext-103   # larger dataset
  python prepare_data.py --tokenizer meta-llama/Llama-2-7b-hf --out data/tokens.npy
  python prepare_data.py --dataset codesearchnet-python --out data/code_py.npy
  # Mix FineWeb-Edu (80%) + code (20%):
  python prepare_data.py --mix data/fineweb_edu.npy data/code_py.npy --mix_ratio 0.8 --out data/mixed.npy
"""

import argparse
import numpy as np
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


DATASET_PRESETS = {
    "wikitext-2":           ("wikitext", "wikitext-2-raw-v1"),
    "wikitext-103":         ("wikitext", "wikitext-103-raw-v1"),
    "tinystories":          ("roneneldan/TinyStories", None),
    "openwebtext":          ("Skylion007/openwebtext", None),
    "fineweb-edu":          ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
    "fineweb-edu-10b":      ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
    "codesearchnet-python": ("code_search_net", "python"),
    "codesearchnet-all":    ("code_search_net", "all"),
    "starcoderdata":        ("bigcode/starcoderdata", "python"),
    # Mythos Phase 2 — verified accessible code datasets (no HF token needed)
    "hf-stack-v1":          ("smangrul/hf-stack-v1", None),           # real code, 'content' col
    "magicoder-evol":       ("ise-uiuc/Magicoder-Evol-Instruct-110K", None),  # instruction pairs
    "magicoder-oss":        ("ise-uiuc/Magicoder-OSS-Instruct-75K", None),    # OSS code instruct
    "code-feedback":        ("m-a-p/CodeFeedback-Filtered-Instruction", None), # code QA
    "evol-instruct-80k":    ("nickrosh/Evol-Instruct-Code-80k-v1", None),     # evolved code instruct
    "code-alpaca":          ("HuggingFaceH4/CodeAlpaca_20K", None),           # code alpaca
    "code-bagel":           ("Replete-AI/code_bagel", None),                  # mixed code instruct
    "code-contests":        ("deepmind/code_contests", None),                 # competitive programming
}


def prepare(args: argparse.Namespace) -> None:
    # --- Load tokenizer ---
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = tokenizer.vocab_size
    print(f"  vocab_size={vocab_size:,} | eos={tokenizer.eos_token_id}")

    # --- Load dataset ---
    preset = DATASET_PRESETS.get(args.dataset)
    if preset:
        ds_name, ds_config = preset
    else:
        ds_name, ds_config = args.dataset, args.dataset_config or None

    cfg_label = ds_config or "default"
    print(f"Loading dataset: {ds_name} ({cfg_label})")
    split_str = args.split
    if args.max_rows:
        split_str = f"{args.split}[:{args.max_rows}]"
    ds = load_dataset(ds_name, ds_config, split=split_str)

    # --- Tokenize ---
    text_col, pair_col = _find_text_column(ds)
    col_desc = f"'{text_col}' + '{pair_col}'" if pair_col else f"'{text_col}'"
    print(f"  text column: {col_desc} | rows: {len(ds):,}")

    eos = tokenizer.eos_token_id or 0
    all_tokens: list[int] = []

    def tokenize_batch(batch):
        if pair_col:
            # Instruction + response: concatenate with separator
            texts = [
                f"{q}\n\n{a}" for q, a in zip(batch[text_col], batch[pair_col])
                if q and a and q.strip() and a.strip()
            ]
        else:
            texts = [t for t in batch[text_col] if t and t.strip()]
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        return {"ids": [ids + [eos] for ids in encoded]}

    print("Tokenizing...")
    tokenized = ds.map(tokenize_batch, batched=True, batch_size=1000,
                       remove_columns=ds.column_names)

    for row in tokenized:
        all_tokens.extend(row["ids"])

    arr = np.array(all_tokens, dtype=np.int32)
    print(f"Total tokens: {len(arr):,}")

    # --- Save ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), arr)
    print(f"Saved: {out_path}  ({out_path.stat().st_size / 1024**2:.1f} MB)")
    print(f"vocab_size needed: {vocab_size}  (set --vocab_size in train.py accordingly)")


def _find_text_column(ds) -> tuple[str, str | None]:
    """Return (primary_col, secondary_col).
    For instruction datasets, returns (instruction_col, response_col).
    For plain text datasets, returns (text_col, None).
    """
    cols = set(ds.column_names)
    # Instruction + response pairs
    for q_col, a_col in [
        ("instruction", "output"), ("instruction", "response"),
        ("question", "answer"), ("query", "answer"),
        ("prompt", "completion"), ("input", "output"),
    ]:
        if q_col in cols and a_col in cols:
            return q_col, a_col
    # Plain text
    for name in ("text", "story", "content", "document",
                 "whole_func_string", "original_string", "code", "solution"):
        if name in ds.column_names:
            return name, None
    return ds.column_names[0], None


def mix_datasets(args: argparse.Namespace) -> None:
    """Interleave two pre-tokenized .npy files at a given ratio."""
    paths = [Path(p) for p in args.mix]
    if len(paths) != 2:
        raise ValueError("--mix requires exactly 2 .npy paths")
    a = np.load(str(paths[0]))
    b = np.load(str(paths[1]))
    ratio = args.mix_ratio  # fraction of tokens from paths[0]
    # Use all of FILE_A; sample FILE_B to achieve the target ratio
    n_a = len(a)
    n_b = int(n_a * (1 - ratio) / ratio)
    n_b = min(n_b, len(b))
    # interleave by sampling without replacement in proportion
    # Concatenate directly: TokenDataset samples random start positions at training time,
    # so token-level shuffling is both unnecessary and destructive to sequence continuity.
    combined = np.concatenate([a, b[:n_b]])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), combined)
    total = len(combined)
    print(f"Mixed: {n_a:,} tokens from {paths[0].name} + {n_b:,} from {paths[1].name} = {total:,} total")
    print(f"Saved: {out_path}  ({out_path.stat().st_size / 1024**2:.1f} MB)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenMythos data preparation")
    p.add_argument("--dataset",        default="wikitext-2",
                   help=f"Dataset preset or HF path. Presets: {list(DATASET_PRESETS)}")
    p.add_argument("--dataset_config", default=None, help="HF dataset config (overrides preset)")
    p.add_argument("--split",          default="train", help="Dataset split")
    p.add_argument("--tokenizer",      default="gpt2",  help="HF tokenizer name or path")
    p.add_argument("--out",            default="data/tokens.npy", help="Output .npy path")
    p.add_argument("--max_rows",       type=int, default=None,   help="Limit dataset rows (e.g. 50000 for quick test)")
    p.add_argument("--mix",            nargs=2, metavar=("FILE_A", "FILE_B"), default=None,
                   help="Mix two existing .npy files instead of downloading. Skips --dataset.")
    p.add_argument("--mix_ratio",      type=float, default=0.8,
                   help="Fraction of tokens from FILE_A (default 0.8 = 80%% A, 20%% B)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mix:
        mix_datasets(args)
    else:
        prepare(args)
