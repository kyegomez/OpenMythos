#!/bin/bash

# パス設定
PROJECT_PATH="/Users/ys/vault/projects/OpenMythos"
MODEL_BASE="/Users/ys/.cache/huggingface/hub/models--mlx-community--Qwen2.5.1-Coder-7B-Instruct-8bit"
MODEL_PATH="${MODEL_BASE}/snapshots/$(ls -1 ${MODEL_BASE}/snapshots/ | head -n 1)"

# 1. 既存のサーバーを終了
lsof -ti:8000 | xargs kill -9 2>/dev/null

# 2. 新しいターミナルでMLXサーバーを起動
osascript -e "tell application \"Terminal\" to do script \"cd '$PROJECT_PATH' && source .venv/bin/activate && python -m mlx_lm server --model '$MODEL_PATH' --port 8000\""

echo "Qwen2.5.1-Coder をロード中... 5秒後にAiderを起動します。"
sleep 5

# 3. Aiderを起動
cd "$PROJECT_PATH"
source .venv/bin/activate
aider --model openai/local --openai-api-base http://127.0.0 --openai-api-key dummy
