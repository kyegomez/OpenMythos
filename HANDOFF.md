# OpenMythos セッション引き継ぎ（2026-05-01）

## プロジェクト概要

Apple Silicon (MLX) ネイティブな再帰深度トランスフォーマー（Recurrent-Depth Transformer）の自主訓練プロジェクト。

## 現在の状態

| 項目 | 状態 |
|------|------|
| ベストモデル | `ckpt/1b-mythos/step_060000.npz` |
| ベスト Loss | **1.0225**（全フェーズ最良） |
| 2b-mythos | 発散により停止（最良 loss 1.4069、1b に敵わず） |
| コード | upstream 構成基盤アップデート済み（commit `e1d5444`） |

## アーキテクチャ（1b-mythos）

```python
MythosConfig(
    vocab_size=50257,       # GPT-2 tokenizer
    dim=2048,
    n_heads=16,
    max_seq_len=1024,
    max_loop_iters=16,
    prelude_layers=2,
    coda_layers=2,
    n_experts=16,
    n_shared_experts=2,
    n_experts_per_tok=2,
    expert_dim=256,
)
# → ~400M params（MoE で実効 ~180M アクティブ/token）
```

## 訓練フェーズ履歴

| フェーズ | LR | ステップ範囲 | 最良 loss | 最良 step | Δ |
|---------|-----|------------|---------|---------|---|
| M+   | 1e-5 | 0 → 45,000   | 1.0960 | 45,500  | — |
| M++  | 1e-6 | 45,000 → 55,000 | 1.0462 | 50,500 | −0.050 |
| M+++ | 1e-7 | 55,000 → 60,000 | 1.0269 | 55,500 | −0.019 |
| M4   | 1e-8 | 60,000 → 65,000 | **1.0225** | 60,500 | −0.004 |

改善幅が逓減（−0.050 → −0.004）→ **1b は収束限界に達した**。

## ファイル構成

```
OpenMythos/
├── open_mythos/
│   ├── main.py          # MLX モデル定義（MythosConfig, OpenMythos, MLAttention 等）
│   ├── variants.py      # production スケール configs (1b〜1t)
│   ├── full_model.py    # DeepSeekV2Lite 推論専用モデル
│   └── mcp_server.py   # ローカル推論 MCP サーバー
├── train.py             # MLX 訓練スクリプト
├── eval_inference.py    # 推論評価スクリプト（4チェックポイント × 3プロンプト）
├── data/
│   └── mythos_train.npy # 訓練データ（369,780 chunks × 1024 tok）
└── ckpt/
    ├── 1b-mythos/       # 62個のチェックポイント（90GB）
    │   └── step_060000.npz  ← ベストモデル
    └── 2b-mythos/       # step_042000〜058000（43GB）
```

## チェックポイントのロード方法

```python
# ベストモデルのロード
from train import VARIANTS
from open_mythos.main import OpenMythos
import mlx.core as mx

model = OpenMythos(VARIANTS['1b'])
model.load_weights('ckpt/1b-mythos/step_060000.npz')
mx.eval(model.parameters())

# 推論
from open_mythos.main import ...  # GPT-2 tokenizer 別途
tokens = ...  # mx.array shape (1, T)
out = model.generate(tokens, max_new_tokens=100, n_loops=4)
```

## 訓練の再開（次フェーズ検討事項）

```bash
# もし追加訓練するなら（新データ or 別タスク）
python3 train.py \
  --variant 1b \
  --data data/new_data.npy \
  --checkpoint ckpt/1b-mythos \
  --steps 10000 \
  --batch 4 \
  --lr 1e-8 \    # M4 と同じ or さらに下げる
  --warmup_steps 1 \
  --n_loops 4 \
  --log_every 500 \
  --save_every 1000
```

## 推奨される次のアクション

1. **新データでファインチューン** — mythos データ以外の特化データで fine-tune
2. **量子化** — `step_060000.npz` を 4-bit/8-bit 量子化して推論高速化
3. **MCP サーバー更新** — `open_mythos/mcp_server.py` のモデルパスを `step_060000.npz` に向ける
4. **3b モデル訓練** — `variants.py` の `mythos_3b()` を使い新しいスケール実験

## upstream との関係

- Remote: `https://github.com/kyegomez/OpenMythos`
- ローカルは **MLX フォーク**（upstream は PyTorch + Flash Attn 2 へ移行済み）
- `open_mythos/main.py` のアーキテクチャは MLX のまま維持
- `git pull` すると PyTorch に上書きされるので **pull 禁止**

## 環境

```
Hardware: Apple M2 Ultra 64GB
Python: 3.12 / 3.14
Framework: MLX >= 0.16
Tokenizer: GPT-2 (transformers)
Training speed: ~1,200 tok/s (1b, batch=4)
```
