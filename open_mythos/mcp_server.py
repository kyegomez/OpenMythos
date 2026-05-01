"""
OpenMythos MCP Server — local inference gateway for Claude Code.

Default model: Qwen2.5.1-Coder-7B-Instruct-8bit (MLX, 7.6GB, coding-optimized)
Fallback:      DeepSeek-V2-Lite (custom loader, 27-layer MoE)

Architecture (A + C hybrid):
  - route_task  : decide local vs Claude API (with cost-saving compression signal)
  - local_infer : direct local inference (private, offline, free)
  - summarize   : compress file/code context → reduces Claude API token usage
  - review_code : local code review for quick feedback loop

Usage (add to .claude/settings.json):
  {
    "mcpServers": {
      "openmythos": {
        "command": "python",
        "args": ["/path/to/open_mythos/mcp_server.py"],
        "env": {"OPENMYTHOS_MODEL_PATH": "/path/to/model/snapshot"}
      }
    }
  }
"""

import os
import sys
import json
import textwrap
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

_QWEN_CODER_DEFAULT = (
    "/Users/ys/.cache/huggingface/hub"
    "/models--mlx-community--Qwen2.5.1-Coder-7B-Instruct-8bit"
    "/snapshots/ce37efd3ed02d730900614a108d49d5006426103"
)

# Lazy model loading — only initialize when first tool is called
_model = None
_tokenizer = None
_use_mlx_lm: bool = False   # True → mlx_lm.generate(); False → model.generate()
_model_path = os.environ.get("OPENMYTHOS_MODEL_PATH", _QWEN_CODER_DEFAULT)


def _detect_model_type(model_path: str) -> str:
    """Read config.json and return model_type string (e.g. 'qwen2', 'deepseek_v2')."""
    import json as _json
    for name in ("config.json",):
        p = Path(model_path) / name
        if p.exists():
            try:
                cfg = _json.loads(p.read_text())
                # Qwen3.5 wraps params under text_config
                return cfg.get("model_type") or cfg.get("text_config", {}).get("model_type", "unknown")
            except Exception:
                pass
    return "unknown"


def _ensure_model():
    global _model, _tokenizer, _use_mlx_lm
    if _model is not None:
        return

    model_type = _detect_model_type(_model_path)
    print(f"[OpenMythos] model_type={model_type!r}, path={_model_path}", file=sys.stderr)

    # DeepSeek-V2 / DeepSeek-V3 → custom loader
    if "deepseek" in model_type.lower():
        print("[OpenMythos] Loading DeepSeek-V2-Lite (27 layers, custom loader)...", file=sys.stderr)
        from mlx_lm import load as mlx_load
        from open_mythos.full_model import load_deepseek_v2_lite, MythosConfig
        cfg = MythosConfig()
        _model = load_deepseek_v2_lite(_model_path, cfg)
        _, _tokenizer = mlx_load(_model_path)
        _use_mlx_lm = False
    else:
        # Qwen2 / Qwen3.5 / Mistral / Llama etc. → standard mlx_lm loader
        print(f"[OpenMythos] Loading {model_type} via mlx_lm...", file=sys.stderr)
        from mlx_lm import load as mlx_load
        _model, _tokenizer = mlx_load(_model_path)
        _use_mlx_lm = True

    print("[OpenMythos] Model ready.", file=sys.stderr)


def _apply_chat_template(prompt: str) -> str:
    if hasattr(_tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            return _tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return f"<｜begin▁of▁sentence｜>User: {prompt}\n\nAssistant:"


def _generate(prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    _ensure_model()
    formatted = _apply_chat_template(prompt)

    if _use_mlx_lm:
        # Standard path: Qwen / Mistral / Llama etc.
        # mlx_lm >= 0.20: temperature is passed via sampler=make_sampler(temp=...)
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler
        return mlx_generate(
            _model,
            _tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
            verbose=False,
        ).strip()
    else:
        # Custom path: DeepSeek-V2-Lite
        import mlx.core as mx
        input_ids = mx.array(_tokenizer.encode(formatted))[None]
        output_ids = _model.generate(input_ids, max_new_tokens=max_tokens, temperature=temperature)
        # Decode full sequence to preserve BPE spacing, then strip the input portion
        full_text = _tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        input_text = _tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
        if full_text.startswith(input_text):
            return full_text[len(input_text):].strip()
        return full_text.strip()


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "OpenMythos",
    instructions=(
        "Local inference gateway (default: Qwen2.5.1-Coder-7B, coding-optimized). "
        "Use route_task first to decide if a task should run locally or escalate to Claude API. "
        "When escalating, always use the returned compressed_context to reduce API token usage. "
        "Use local_infer for private code that should not leave this machine."
    ),
)


@mcp.tool()
def route_task(task: str, code: str = "", offline: bool = False) -> dict:
    """
    Decide whether to handle a coding task locally or escalate to Claude API.

    Returns:
      use: "local" | "api"
      reason: why this routing was chosen
      confidence: 0-1 score
      compress_first: if True, call summarize_code before sending to Claude API
      compressed_context: pre-compressed summary when compress_first is True
    """
    from open_mythos.router import route_task as _route

    decision = _route(task, code, offline=offline)
    result = {
        "use": decision.use,
        "reason": decision.reason,
        "confidence": decision.confidence,
        "compress_first": decision.compress_first,
        "compressed_context": None,
    }

    # Pre-compress if escalating with large context
    if decision.use == "api" and decision.compress_first and code:
        result["compressed_context"] = _summarize(code, max_tokens=200)

    return result


@mcp.tool()
def local_infer(prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    """
    Run local inference (Qwen2.5.1-Coder-7B by default). Use for private code or when offline.
    The model runs entirely on this machine — no data leaves the local environment.

    Args:
      prompt: instruction + code context
      max_tokens: maximum tokens to generate (default 256)
      temperature: sampling temperature 0=greedy, 0.7=balanced (default 0.7)
    """
    return _generate(prompt, max_tokens=max_tokens, temperature=temperature)


@mcp.tool()
def summarize_code(code: str, focus: str = "purpose and key logic") -> str:
    """
    Compress code into a compact summary using local inference.
    Use this BEFORE sending large files to Claude API to reduce token usage.

    Args:
      code: source code to summarize
      focus: what aspect to emphasize (default: "purpose and key logic")
    """
    return _summarize(code, focus=focus)


@mcp.tool()
def review_code(code: str, focus: str = "bugs, style, and potential issues") -> str:
    """
    Local code review using Qwen2.5.1-Coder-7B. Fast, private, works offline.
    Best for: syntax issues, style checks, docstring quality, obvious bugs.
    For security audits or complex architectural issues, use Claude API instead.

    Args:
      code: code to review
      focus: review focus (default: "bugs, style, and potential issues")
    """
    prompt = textwrap.dedent(f"""
        Review the following code for {focus}.
        Be concise. List specific issues with line references where possible.

        ```
        {code[:3000]}
        ```

        Review:
    """).strip()
    return _generate(prompt, max_tokens=300, temperature=0.3)


@mcp.tool()
def read_and_summarize(file_path: str) -> dict:
    """
    Read a local file and return both its content and a compressed summary.
    Optimizes Claude API usage by providing summary alongside raw content.

    Returns:
      path: resolved file path
      size_chars: original file character count
      summary: local-model compressed summary (~200 tokens)
      content_preview: first 500 chars of raw content
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        return {"error": f"File not found: {path}"}
    if not path.is_file():
        return {"error": f"Not a file: {path}"}

    content = path.read_text(encoding="utf-8", errors="replace")
    summary = _summarize(content, max_tokens=200)

    return {
        "path": str(path),
        "size_chars": len(content),
        "summary": summary,
        "content_preview": content[:500],
    }


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _summarize(code: str, focus: str = "purpose and key logic", max_tokens: int = 200) -> str:
    prompt = textwrap.dedent(f"""
        Summarize the following code. Focus on: {focus}.
        Be concise (under {max_tokens // 2} words). No markdown headers.

        ```
        {code[:4000]}
        ```

        Summary:
    """).strip()
    return _generate(prompt, max_tokens=max_tokens, temperature=0.3)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
