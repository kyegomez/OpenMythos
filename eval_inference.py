"""
Step C: ベストモデル推論評価
各フェーズ最良チェックポイントで3プロンプトを比較評価
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import mlx.core as mx
import numpy as np
from pathlib import Path

from open_mythos.main import OpenMythos, MythosConfig
from train import VARIANTS, load_checkpoint

try:
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    USE_TOKENIZER = True
    print("✅ GPT-2 tokenizer loaded")
except Exception as e:
    print(f"⚠️  GPT-2 tokenizer unavailable: {e}")
    USE_TOKENIZER = False

# ---------------------------------------------------------------------------
# 評価対象チェックポイント（各フェーズの最良 loss に最も近いファイル）
# ---------------------------------------------------------------------------
CHECKPOINTS = [
    ("M+  (lr=1e-5)",  "ckpt/1b-mythos/step_045000.npz",  "loss ~1.096 @ step 45500"),
    ("M++ (lr=1e-6)",  "ckpt/1b-mythos/step_050000.npz",  "loss ~1.046 @ step 50500"),
    ("M+++(lr=1e-7)",  "ckpt/1b-mythos/step_055000.npz",  "loss ~1.027 @ step 55500"),
    ("M4  (lr=1e-8)",  "ckpt/1b-mythos/step_060000.npz",  "loss ~1.023 @ step 60500"),
]

# ---------------------------------------------------------------------------
# 評価プロンプト（3種類）
# ---------------------------------------------------------------------------
PROMPTS = [
    "Once upon a time in a kingdom far away,",
    "The ancient wizard raised his staff and said,",
    "In the depths of the dungeon, the hero discovered",
]

N_LOOPS   = 4
MAX_TOKENS = 60


def encode(text: str) -> mx.array:
    if USE_TOKENIZER:
        ids = tokenizer.encode(text, return_tensors=None)
        return mx.array([ids], dtype=mx.uint32)
    # フォールバック: ASCII コードポイント
    return mx.array([[ord(c) % 50257 for c in text]], dtype=mx.uint32)


def decode(token_ids) -> str:
    ids = token_ids[0].tolist()
    if USE_TOKENIZER:
        return tokenizer.decode(ids, skip_special_tokens=True)
    return "".join(chr(min(i, 127)) for i in ids)


def run_eval():
    cfg = VARIANTS["1b"]
    print(f"\n{'='*70}")
    print(f"OpenMythos 1b-mythos — 推論評価（{len(CHECKPOINTS)} checkpoints × {len(PROMPTS)} prompts）")
    print(f"n_loops={N_LOOPS}, max_new_tokens={MAX_TOKENS}")
    print(f"{'='*70}\n")

    results = {}

    for label, ckpt_path, note in CHECKPOINTS:
        if not Path(ckpt_path).exists():
            print(f"⚠️  SKIP {label}: {ckpt_path} not found")
            continue

        print(f"\n{'─'*70}")
        print(f"📌 {label}  [{note}]")
        print(f"   チェックポイント: {ckpt_path}")
        print(f"{'─'*70}")

        # モデルをロードして評価
        model = OpenMythos(cfg)
        mx.eval(model.parameters())
        model.load_weights(ckpt_path)
        mx.eval(model.parameters())
        model.eval()

        ckpt_results = []
        for i, prompt in enumerate(PROMPTS, 1):
            tokens = encode(prompt)
            out_tokens = model.generate(tokens, max_new_tokens=MAX_TOKENS, n_loops=N_LOOPS)
            generated = decode(out_tokens)
            ckpt_results.append(generated)

            print(f"\n  [{i}] {prompt!r}")
            print(f"  → {generated!r}")

        results[label] = ckpt_results
        del model
        mx.metal.clear_cache()

    # ---------------------------------------------------------------------------
    # サマリー
    # ---------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("📊 評価サマリー")
    print(f"{'='*70}")
    print(f"{'Phase':<16} {'最良loss':>10}  {'チェックポイント'}")
    print(f"{'─'*70}")
    for label, _, note in CHECKPOINTS:
        path_short = label
        print(f"{label:<16} {note}")

    print(f"\n✅ 完了 — ベストモデル: M4 (step_060000, loss ~1.023)")
    print(f"   推奨チェックポイント: ckpt/1b-mythos/step_060000.npz\n")


if __name__ == "__main__":
    run_eval()
