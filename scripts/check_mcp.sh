#!/bin/bash
# OpenMythos MCP サーバー ヘルスチェック
# 使い方: bash scripts/check_mcp.sh

PYTHON="/Users/ys/vault/projects/OpenMythos/.venv/bin/python"
SERVER="/Users/ys/vault/projects/OpenMythos/open_mythos/mcp_server.py"
MODEL_PATH="/Users/ys/.cache/huggingface/hub/models--mlx-community--Qwen2.5.1-Coder-7B-Instruct-8bit/snapshots/ce37efd3ed02d730900614a108d49d5006426103"

echo "=== OpenMythos MCP ヘルスチェック ==="

# 1. venv確認
if [ -f "$PYTHON" ]; then
  echo "[OK] venv: $PYTHON"
else
  echo "[FAIL] venv not found: $PYTHON"
  exit 1
fi

# 2. モデルパス確認
if [ -d "$MODEL_PATH" ]; then
  echo "[OK] モデルパス存在"
else
  echo "[FAIL] モデルパス不在: $MODEL_PATH"
  exit 1
fi

# 3. 依存ライブラリ確認
"$PYTHON" -c "from mcp.server.fastmcp import FastMCP; from mlx_lm import load" 2>/dev/null \
  && echo "[OK] mcp + mlx_lm インポート成功" \
  || { echo "[FAIL] 依存ライブラリのインポートエラー"; exit 1; }

# 4. stdioプロトコル疎通確認
RESULT=$(printf '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"healthcheck","version":"1"}}}\n' \
  | OPENMYTHOS_MODEL_PATH="$MODEL_PATH" "$PYTHON" "$SERVER" 2>/dev/null \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('OK' if 'result' in d else 'FAIL')" 2>/dev/null)

if [ "$RESULT" = "OK" ]; then
  echo "[OK] MCPサーバー stdio 疎通確認 → 正常応答"
  echo ""
  echo "✅ すべてのチェック通過。Claude Code を再起動すると openmythos ツールが利用可能になります。"
else
  echo "[FAIL] MCPサーバーが initialize に応答しませんでした"
  echo "  ログ: OPENMYTHOS_MODEL_PATH=$MODEL_PATH $PYTHON $SERVER"
  exit 1
fi
