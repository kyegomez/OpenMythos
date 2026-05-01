"""
OpenMythos MCP Server — Claude Code integration via Model Context Protocol.

Exposes OpenMythos as tools for Claude Code:
  - mythos_complete : code completion given a prefix
  - mythos_explain  : explain what a code snippet does
  - mythos_review   : review code for issues / improvements

Usage (standalone):
  python mcp_server.py --checkpoint ckpt/mythos-2b --variant 2b

Usage (via Claude Code .claude/mcp.json):
  {
    "mythos": {
      "command": "python3",
      "args": ["/Users/ys/vault/projects/OpenMythos/mcp_server.py",
               "--checkpoint", "ckpt/mythos-2b", "--variant", "2b"],
      "env": {}
    }
  }

The server speaks JSON-RPC 2.0 over stdio (MCP transport).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from transformers import AutoTokenizer

from open_mythos.main import OpenMythos, MythosConfig
from train import VARIANTS

# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

_model: OpenMythos | None = None
_tokenizer = None
_n_loops: int = 6


def load_model(checkpoint: str, variant: str, n_loops: int) -> None:
    global _model, _tokenizer, _n_loops
    cfg = VARIANTS[variant]
    _n_loops = n_loops
    _model = OpenMythos(cfg)

    ckpts = sorted(Path(checkpoint).glob("step_*.npz"))
    if not ckpts:
        _log(f"No checkpoints found in {checkpoint}", level="warning")
        return
    latest = str(ckpts[-1])
    _model.load_weights(latest)
    mx.eval(_model.parameters())
    step = int(ckpts[-1].stem.split("_")[1])
    _log(f"Loaded: {latest} (step {step}, variant={variant}, n_loops={n_loops})")

    _tokenizer = AutoTokenizer.from_pretrained("gpt2")


def _log(msg: str, level: str = "info") -> None:
    print(f"[mythos-mcp] [{level}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Text generation helper
# ---------------------------------------------------------------------------

def _generate(prompt: str, max_new_tokens: int = 256,
              temperature: float = 0.7, top_p: float = 0.9) -> str:
    if _model is None or _tokenizer is None:
        return "[Error: model not loaded]"

    input_ids = _tokenizer.encode(prompt)
    tokens = mx.array([input_ids], dtype=mx.uint32)
    eos_id = _tokenizer.eos_token_id or 50256

    for _ in range(max_new_tokens):
        logits = _model(tokens, n_loops=_n_loops)
        next_logits = logits[:, -1, :].astype(mx.float32)

        if temperature > 0:
            next_logits = next_logits / temperature
            probs = mx.softmax(next_logits, axis=-1)
            sorted_idx = mx.argsort(probs, axis=-1)[:, ::-1]
            sorted_probs = mx.take_along_axis(probs, sorted_idx, axis=-1)
            cumsum = mx.cumsum(sorted_probs, axis=-1)
            mask = (cumsum - sorted_probs) < top_p
            filtered = mx.where(mask, sorted_probs, mx.zeros_like(sorted_probs))
            normalized = filtered / (mx.sum(filtered, axis=-1, keepdims=True) + 1e-8)
            gumbel = -mx.log(-mx.log(mx.random.uniform(shape=normalized.shape) + 1e-10) + 1e-10)
            sample_idx = mx.argmax(mx.log(normalized + 1e-10) + gumbel, axis=-1, keepdims=True)
            next_token = mx.take_along_axis(sorted_idx, sample_idx, axis=-1)
        else:
            next_token = mx.argmax(next_logits, axis=-1, keepdims=True)

        tokens = mx.concatenate([tokens, next_token], axis=1)
        mx.eval(tokens)
        if int(next_token.item()) == eos_id:
            break

    return _tokenizer.decode(tokens[0].tolist())


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_complete(code_prefix: str, max_tokens: int = 256,
                   temperature: float = 0.5) -> str:
    full = _generate(code_prefix, max_new_tokens=max_tokens, temperature=temperature)
    return full[len(code_prefix):]  # return only the continuation


def _tool_explain(code: str) -> str:
    prompt = f"# Explain this code:\n{code}\n# Explanation:\n"
    full = _generate(prompt, max_new_tokens=200, temperature=0.4)
    return full[len(prompt):]


def _tool_review(code: str) -> str:
    prompt = f"# Code review for the following Python code:\n{code}\n# Issues and improvements:\n"
    full = _generate(prompt, max_new_tokens=300, temperature=0.5)
    return full[len(prompt):]


# ---------------------------------------------------------------------------
# MCP JSON-RPC 2.0 server (stdio transport)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "mythos_complete",
        "description": "Complete Python/code given a prefix. Returns the generated continuation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code_prefix": {"type": "string", "description": "Code to complete"},
                "max_tokens":  {"type": "integer", "default": 256, "description": "Max tokens to generate"},
                "temperature": {"type": "number",  "default": 0.5,  "description": "Sampling temperature"},
            },
            "required": ["code_prefix"],
        },
    },
    {
        "name": "mythos_explain",
        "description": "Explain what a code snippet does in plain language.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Code to explain"},
            },
            "required": ["code"],
        },
    },
    {
        "name": "mythos_review",
        "description": "Review code for bugs, style issues, and improvement suggestions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Code to review"},
            },
            "required": ["code"],
        },
    },
]


def _send(obj: dict) -> None:
    line = json.dumps(obj)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _handle_request(req: dict) -> dict | None:
    method = req.get("method", "")
    req_id = req.get("id")
    params = req.get("params", {})

    def ok(result: Any) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def err(code: int, msg: str) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": msg}}

    if method == "initialize":
        return ok({
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "OpenMythos", "version": "1.0"},
        })

    if method == "tools/list":
        return ok({"tools": TOOLS})

    if method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments", {})
        try:
            if name == "mythos_complete":
                result = _tool_complete(
                    args["code_prefix"],
                    max_tokens=int(args.get("max_tokens", 256)),
                    temperature=float(args.get("temperature", 0.5)),
                )
            elif name == "mythos_explain":
                result = _tool_explain(args["code"])
            elif name == "mythos_review":
                result = _tool_review(args["code"])
            else:
                return err(-32601, f"Unknown tool: {name}")

            return ok({"content": [{"type": "text", "text": result}]})
        except Exception as e:
            return err(-32603, str(e))

    if method == "notifications/initialized":
        return None  # no response for notifications

    # Unknown method
    if req_id is not None:
        return err(-32601, f"Method not found: {method}")
    return None


def run_stdio() -> None:
    _log("MCP server ready (stdio)")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            _send({"jsonrpc": "2.0", "id": None,
                   "error": {"code": -32700, "message": f"Parse error: {e}"}})
            continue

        response = _handle_request(req)
        if response is not None:
            _send(response)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenMythos MCP Server")
    p.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    p.add_argument("--variant",    default="2b", choices=list(VARIANTS))
    p.add_argument("--n_loops",    type=int, default=6, help="Recurrent loops")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_model(args.checkpoint, args.variant, args.n_loops)
    run_stdio()
