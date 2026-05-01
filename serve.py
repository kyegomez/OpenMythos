"""
OpenMythos Inference Server — FastAPI + MLX text generation endpoint.

Usage:
  python serve.py --checkpoint ckpt/mythos-2b --variant 2b --port 8765
  python serve.py --checkpoint ckpt/1b-mixed  --variant 1b --port 8765
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    raise ImportError("pip install fastapi uvicorn pydantic")

from open_mythos.main import OpenMythos, MythosConfig
from train import VARIANTS

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_model: Optional[OpenMythos] = None
_tokenizer = None
_cfg: Optional[MythosConfig] = None
_n_loops: int = 4


def load_model(checkpoint: str, variant: str, n_loops: int) -> None:
    global _model, _tokenizer, _cfg, _n_loops
    _cfg = VARIANTS[variant]
    _n_loops = n_loops
    _model = OpenMythos(_cfg)

    ckpts = sorted(Path(checkpoint).glob("step_*.npz"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint}")
    latest = str(ckpts[-1])
    _model.load_weights(latest)
    mx.eval(_model.parameters())
    step = int(ckpts[-1].stem.split("_")[1])
    print(f"[serve] Loaded: {latest} (step {step})")

    _tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"[serve] Tokenizer: gpt2 | vocab={_tokenizer.vocab_size:,}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_text(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    n_loops: Optional[int] = None,
) -> dict:
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded")

    loops = n_loops or _n_loops
    input_ids = _tokenizer.encode(prompt)
    tokens = mx.array([input_ids], dtype=mx.uint32)
    eos_id = _tokenizer.eos_token_id or 50256

    t0 = time.time()
    generated = 0

    for _ in range(max_new_tokens):
        logits = _model(tokens, n_loops=loops)
        next_logits = logits[:, -1, :].astype(mx.float32)

        if temperature > 0:
            next_logits = next_logits / temperature
            probs = mx.softmax(next_logits, axis=-1)
            # Top-p (nucleus) sampling
            sorted_idx = mx.argsort(probs, axis=-1)[:, ::-1]
            sorted_probs = mx.take_along_axis(probs, sorted_idx, axis=-1)
            cumsum = mx.cumsum(sorted_probs, axis=-1)
            mask = (cumsum - sorted_probs) < top_p
            filtered = mx.where(mask, sorted_probs, mx.zeros_like(sorted_probs))
            filtered_sum = mx.sum(filtered, axis=-1, keepdims=True)
            normalized = filtered / (filtered_sum + 1e-8)
            gumbel = -mx.log(-mx.log(mx.random.uniform(shape=normalized.shape) + 1e-10) + 1e-10)
            sample_idx = mx.argmax(mx.log(normalized + 1e-10) + gumbel, axis=-1, keepdims=True)
            next_token = mx.take_along_axis(sorted_idx, sample_idx, axis=-1)
        else:
            next_token = mx.argmax(next_logits, axis=-1, keepdims=True)

        tokens = mx.concatenate([tokens, next_token], axis=1)
        mx.eval(tokens)
        generated += 1

        if int(next_token.item()) == eos_id:
            break

    elapsed = time.time() - t0
    text = _tokenizer.decode(tokens[0].tolist())
    tps = generated / elapsed if elapsed > 0 else 0.0

    return {
        "text": text,
        "prompt": prompt,
        "generated_tokens": generated,
        "tokens_per_second": round(tps, 1),
        "elapsed_seconds": round(elapsed, 2),
        "n_loops": loops,
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OpenMythos Inference API",
    description="Local MLX language model inference server",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    n_loops: Optional[int] = None


class GenerateResponse(BaseModel):
    text: str
    prompt: str
    generated_tokens: int
    tokens_per_second: float
    elapsed_seconds: float
    n_loops: int


@app.get("/health")
def health():
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    return {
        "status": "ok",
        "variant": next((k for k, v in VARIANTS.items() if v is _cfg), "unknown"),
        "n_loops": _n_loops,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        result = generate_text(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            n_loops=req.n_loops,
        )
        return GenerateResponse(**result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/complete")
def complete(req: GenerateRequest):
    """Code completion — returns only the generated continuation (not the prompt)."""
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    result = generate_text(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        n_loops=req.n_loops,
    )
    # Strip the prompt prefix from output
    continuation = result["text"][len(req.prompt):]
    result["text"] = continuation
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenMythos Inference Server")
    p.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    p.add_argument("--variant",    default="2b", choices=list(VARIANTS))
    p.add_argument("--n_loops",    type=int, default=6, help="Recurrent loops")
    p.add_argument("--port",       type=int, default=8765)
    p.add_argument("--host",       default="127.0.0.1")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_model(args.checkpoint, args.variant, args.n_loops)
    print(f"[serve] Starting on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
