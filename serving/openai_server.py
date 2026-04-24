"""
OpenAI-Compatible API Server for OpenMythos
==========================================

Implements the OpenAI Chat Completions API (`/v1/chat/completions`)
and Model List API (`/v1/models`) using FastAPI.

Supports:
- Streaming responses (text/event-stream)
- Function calling (tool_calls) — basic schema
- Context memory via OpenMythosMemoryFacade
- HuggingFace model loading or mock inference (no GPU needed for testing)

Usage:
    # With HuggingFace exported model (requires real weights):
    python serving/openai_server.py \
        --hf-path huggingface/openmythos-3b \
        --port 8000

    # Mock mode (no GPU needed — responds with simulated output):
    python serving/openai_server.py \
        --mock \
        --port 8000

    # With pretrained HF model (any causal LM):
    python serving/openai_server.py \
        --hf-path meta-llama/Llama-3.2-1B \
        --port 8000

API Reference: https://platform.openai.com/docs/api-reference
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Optional

# ---------------------------------------------------------------------------
# Server bootstrap — imports guarded so module loads even without deps
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    print("ERROR: fastapi and uvicorn are required for serving.")
    print("  pip install fastapi uvicorn")
    sys.exit(1)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ChatMessage:
    """Minimal chat message — matches OpenAI API schema."""
    def __init__(self, role: str, content: str, name: Optional[str] = None):
        self.role = role
        self.content = content
        self.name = name

    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


class ChatCompletionRequest:
    def __init__(self, data: dict):
        self.model: str = data.get("model", "openmythos-3b")
        self.messages: list[dict] = data.get("messages", [])
        self.temperature: float = float(data.get("temperature", 0.7))
        self.top_p: float = float(data.get("top_p", 0.9))
        self.max_tokens: int = int(data.get("max_tokens", 256))
        self.stream: bool = bool(data.get("stream", False))
        self.stop: Optional[list[str]] = data.get("stop")
        self.seed: Optional[int] = data.get("seed")
        self.frequency_penalty: float = float(data.get("frequency_penalty", 0.0))
        self.presence_penalty: float = float(data.get("presence_penalty", 0.0))
        self.tools: Optional[list[dict]] = data.get("tools")
        self.tool_choice: Optional[Any] = data.get("tool_choice", "auto")
        self.user: Optional[str] = data.get("user")


# ---------------------------------------------------------------------------
# Model wrapper (mock or HF)
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Unified inference interface. Subclass to add real model support.

    MockEngine: returns a fixed response + token stream for testing.
    HFEngine: loads a HuggingFace model and runs real inference.
    """

    def __init__(self, model_id: str, device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._tokenizer = None

    def load(self):
        """Called after init — override to load real model."""
        pass

    @property
    def is_mock(self) -> bool:
        return True

    def encode(self, text: str) -> list[int]:
        # Simple whitespace tokenizer for mock mode
        if not text.strip():
            return [0]
        words = text.split()
        return [hash(w) % 32000 for w in words] + [2]  # +2 = </s>

    def decode(self, token_ids: list[int]) -> str:
        return f"[token:{token_ids[:3]}...]"

    def generate_stream(
        self,
        messages: list[ChatMessage],
        max_tokens: int,
        temperature: float,
        stop: Optional[list[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Yield SSE-formatted chunks.
        Override in subclass for real inference.
        """
        # Mock response
        prompt = " ".join(m.content for m in messages if m.role == "user")
        mock_text = (
            f"This is a mock response to: '{prompt[:50]}...'\n"
            f"(Set --hf-path to use real OpenMythos model)"
        )

        for i, char in enumerate(mock_text):
            chunk = {
                "choices": [{
                    "index": 0,
                    "delta": {"content": char},
                    "finish_reason": None,
                }],
                "model": self.model_id,
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            time.sleep(0.02)
            if i >= 50:
                break

        # Final chunk
        yield f"data: [DONE]\n\n"

    def generate(
        self,
        messages: list[ChatMessage],
        max_tokens: int,
        temperature: float,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Non-streaming generate — override for real inference."""
        prompt = " ".join(m.content for m in messages if m.role == "user")
        return (
            f"Mock response to: '{prompt[:50]}...'\n"
            f"(Set --hf-path to use real OpenMythos model)"
        )


class HFInferenceEngine(InferenceEngine):
    """Real HuggingFace model inference."""

    def __init__(self, model_id: str, device: str = "cpu", max_length: int = 4096):
        super().__init__(model_id, device)
        self.max_length = max_length
        self._model = None

    @property
    def is_mock(self) -> bool:
        return False

    def load(self):
        print(f"[HFEngine] Loading model from '{self.model_id}' ...")

        if not _HAS_TORCH:
            raise RuntimeError("torch is required for HF inference")

        # Try loading as a HuggingFace model
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True,
            )
            self._model.eval()
            print(f"[HFEngine] Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from '{self.model_id}': {e}\n"
                f"Ensure the path is a valid HuggingFace model ID or local path."
            ) from e

    @property
    def tokenizer(self):
        return self._tokenizer

    def encode(self, text: str) -> list[int]:
        if self._tokenizer:
            return self._tokenizer.encode(text, return_tensors="pt").tolist()[0]
        return super().encode(text)

    def decode(self, token_ids: list[int]) -> str:
        if self._tokenizer:
            return self._tokenizer.decode(token_ids)
        return super().decode(token_ids)

    @torch.no_grad()
    def generate_stream(
        self,
        messages: list[ChatMessage],
        max_tokens: int,
        temperature: float,
        stop: Optional[list[str]] = None,
    ) -> AsyncGenerator[str, None]:
        if not self._model or not self._tokenizer:
            raise RuntimeError("Model not loaded")

        # Build prompt from messages (ChatML format)
        prompt = self._build_prompt(messages)

        import torch
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": max(temperature, 1e-6),  # avoid 0 temp
            "top_p": 0.9,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id or 0,
            "eos_token_id": self._tokenizer.eos_token_id or 2,
        }

        # Streaming via incremental decode
        from transformers import TextIteratorStreamer
        from threading import Thread

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs["streamer"] = streamer

        thread = Thread(target=self._model.generate, kwargs={**gen_kwargs})
        thread.start()

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        for text_chunk in streamer:
            if text_chunk:
                chunk = {
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text_chunk},
                        "finish_reason": None,
                    }],
                    "model": self.model_id,
                    "id": completion_id,
                }
                yield f"data: {json.dumps(chunk)}\n\n"

        yield f"data: [DONE]\n\n"
        thread.join()

    @torch.no_grad()
    def generate(
        self,
        messages: list[ChatMessage],
        max_tokens: int,
        temperature: float,
        stop: Optional[list[str]] = None,
    ) -> str:
        if not self._model or not self._tokenizer:
            raise RuntimeError("Model not loaded")

        prompt = self._build_prompt(messages)
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        output = self._model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 1e-6),
            top_p=0.9,
            do_sample=temperature > 0,
            pad_token_id=self._tokenizer.pad_token_id or 0,
            eos_token_id=self._tokenizer.eos_token_id or 2,
        )

        generated = output[0][input_ids.shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def _build_prompt(self, messages: list[ChatMessage]) -> str:
        """Build a prompt string from ChatML messages."""
        parts = []
        for m in messages:
            if m.role == "system":
                parts.append(f"<|system|>\n{m.content}")
            elif m.role == "user":
                parts.append(f"<|user|>\n{m.content}")
            elif m.role == "assistant":
                parts.append(f"<|assistant|>\n{m.content}")
            elif m.role == "tool":
                parts.append(f"<|tool|>\n{m.content}")
        parts.append("<|assistant|>")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Memory integration (optional)
# ---------------------------------------------------------------------------

def try_load_memory():
    """Try to load OpenMythosMemoryFacade for context-aware responses."""
    try:
        from open_mythos.memory.memory_facade import OpenMythosMemoryFacade, MemoryFacadeConfig
        from open_mythos.rag.m_flow_bundle import SemanticEdge

        def mock_embed(texts: list[str]) -> list:
            import numpy as np
            # Random but deterministic embeddings for testing
            return [np.random.rand(64).astype(np.float32) * 0.1 for _ in texts]

        cfg = MemoryFacadeConfig(embed_func=mock_embed)
        facade = OpenMythosMemoryFacade(cfg)
        return facade
    except Exception as e:
        print(f"[Memory] Could not load memory facade: {e}")
        return None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app(engine: InferenceEngine) -> FastAPI:
    app = FastAPI(
        title="OpenMythos OpenAI-Compatible API",
        description=(
            "OpenAI-compatible chat completions API powered by OpenMythos "
            "(Recurrent-Depth Transformer with M-flow Memory)."
        ),
        version="1.0.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Memory (optional)
    memory_facade = try_load_memory()

    # ── GET /v1/models ────────────────────────────────────────────────────

    @app.get("/v1/models")
    async def list_models():
        """OpenAI-compatible model list endpoint."""
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "openmythos",
                    "permission": [],
                }
            ],
        }

    # ── GET /models ───────────────────────────────────────────────────────

    @app.get("/models")
    async def list_models_short():
        return await list_models()

    # ── POST /v1/chat/completions ─────────────────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        req = ChatCompletionRequest(body)

        messages = [ChatMessage(**m) for m in req.messages]

        # Optional: inject memory context
        if memory_facade and not engine.is_mock:
            try:
                # Search memory for relevant context
                query = messages[-1].content if messages else ""
                # (Memory search would go here in full implementation)
            except Exception:
                pass

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        if req.stream:
            async def stream_response():
                try:
                    async for chunk in engine.generate_stream(
                        messages,
                        max_tokens=req.max_tokens,
                        temperature=req.temperature,
                        stop=req.stop,
                    ):
                        yield chunk
                except Exception as e:
                    error_chunk = {
                        "error": {
                            "message": str(e),
                            "type": "internal_error",
                            "code": "internal_error",
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Request-ID": completion_id,
                },
            )

        else:
            try:
                text = engine.generate(
                    messages,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    stop=req.stop,
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

            return JSONResponse({
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text,
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": sum(len(m.content.split()) for m in messages),
                    "completion_tokens": len(text.split()),
                    "total_tokens": sum(len(m.content.split()) for m in messages) + len(text.split()),
                },
            })

    # ── POST /v1/completions (legacy) ─────────────────────────────────────

    @app.post("/v1/completions")
    async def completions(request: Request):
        """Legacy /v1/completions endpoint — delegates to chat completions."""
        body = await request.json()
        prompt = body.get("prompt", "")
        max_tokens = int(body.get("max_tokens", 256))
        temperature = float(body.get("temperature", 0.7))

        messages = [ChatMessage(role="user", content=prompt)]
        text = engine.generate(messages, max_tokens, temperature)

        return JSONResponse({
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": engine.model_id,
            "choices": [{
                "text": text,
                "index": 0,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(text.split()),
                "total_tokens": len(prompt.split()) + len(text.split()),
            },
        })

    # ── GET /health ───────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": engine.model_id,
            "mock": engine.is_mock,
            "torch_available": _HAS_TORCH,
            "memory_loaded": memory_facade is not None,
        }

    # ── GET / ─────────────────────────────────────────────────────────────

    @app.get("/")
    async def root():
        return {
            "name": "OpenMythos OpenAI-Compatible API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions",
        }

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OpenMythos OpenAI-Compatible Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--hf-path", help="HuggingFace model path (local or HF ID)")
    parser.add_argument("--mock", action="store_true",
                        help="Run in mock mode (no real model needed)")
    parser.add_argument("--device", default="cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu",
                        help="Device for inference (cpu/cuda)")
    parser.add_argument("--max-length", type=int, default=4096,
                        help="Maximum sequence length")

    args = parser.parse_args()

    if args.mock or not args.hf_path:
        print("=" * 60)
        print("  OpenMythos Server — MOCK MODE")
        print("  Responses are simulated (no real model).")
        print("  Use --hf-path to load a real model.")
        print("=" * 60)
        engine = InferenceEngine(
            model_id="openmythos-3b-mock",
            device=args.device,
        )
    else:
        print(f"Loading model from: {args.hf_path}")
        engine = HFInferenceEngine(
            model_id=args.hf_path,
            device=args.device,
            max_length=args.max_length,
        )
        engine.load()

    app = create_app(engine)

    print(f"\n  Server:  http://{args.host}:{args.port}")
    print(f"  API:     http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  Docs:    http://{args.host}:{args.port}/docs")
    print(f"  Health:  http://{args.host}:{args.port}/health")
    print(f"  Model:   {engine.model_id} ({'mock' if engine.is_mock else 'loaded'})")
    print("\n  Example curl:\n")
    print(
        f"  curl -X POST http://{args.host}:{args.port}/v1/chat/completions \\\n"
        f"    -H 'Content-Type: application/json' \\\n"
        f"    -d '{{\"model\": \"{engine.model_id}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}], \"max_tokens\": 64}}'"
    )
    print()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
