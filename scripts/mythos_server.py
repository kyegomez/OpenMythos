#!/usr/bin/env python3
"""
OpenMythos local HTTP server -- one loaded checkpoint, several API dialects.

  GET  /health                  status + checkpoint info
  POST /query                   {"mode": "technique", "query": "T1003"}   (native)
  POST /v1/completions          OpenAI-style, raw prompt
  POST /v1/chat/completions     OpenAI-style, last user message (mode auto-detected)
  GET  /v1/models
  POST /api/generate            Ollama-style, raw prompt
  POST /api/chat                Ollama-style, last user message
  GET  /api/tags                Ollama-style model list

This is NOT a real Ollama model (the looped-MoE architecture cannot be converted
to GGUF). It only speaks the Ollama wire format, so point a client at it with
OLLAMA_HOST=127.0.0.1:8830.

Binds 127.0.0.1 only, no auth. Requests are serialized (one model in memory).
Output quality: a 279M model -- fluent, frequently factually wrong. Verify anything
you rely on.
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # gpt2 tokenizer is cached; skip the retry storm
sys.path.insert(0, str(Path(__file__).resolve().parent))
import infer  # noqa: E402  (sets HF_HOME via paths, imports torch)

MODEL_NAME = "openmythos:massive"
MAX_TOKENS_CAP = 512
LOCK = threading.Lock()
STATE = {}


def detect_mode(q):
    """Same auto-detection as infer.py's interactive loop."""
    u = q.upper()
    if u.startswith("T1") and "." not in q:
        return "technique", q
    if u.startswith("T1"):
        return "sub", q
    if u.startswith("CVE-"):
        return "cve", q
    if u.startswith("CAPEC-"):
        return "capec", q.split("-", 1)[1]
    return "free", q


def run(prompt, max_tokens, temp, top_k, loops):
    max_tokens = max(1, min(int(max_tokens), MAX_TOKENS_CAP))
    temp = max(0.05, float(temp))
    with LOCK:
        m = STATE["model"]
        return infer.generate(m["model"], m["tok"], m["device"], m["cfg"], prompt,
                              max_tokens=max_tokens, loops=int(loops), temp=temp, top_k=int(top_k))


def params(body):
    opts = body.get("options") or {}
    return dict(
        max_tokens=body.get("max_tokens") or opts.get("num_predict") or 120,
        temp=body.get("temperature", opts.get("temperature", 0.3)),
        top_k=body.get("top_k", opts.get("top_k", 40)),
        loops=body.get("loops", 8),
    )


def last_user(messages):
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, list):  # OpenAI content parts
                c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
            return c
    return ""


class Handler(BaseHTTPRequestHandler):
    server_version = "openmythos/1"

    def log_message(self, fmt, *a):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % a))

    def send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path == "/health":
            return self.send(200, {"ok": True, "model": MODEL_NAME, **STATE["info"]})
        if self.path == "/v1/models":
            return self.send(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}]})
        if self.path == "/api/tags":
            return self.send(200, {"models": [{"name": MODEL_NAME, "model": MODEL_NAME, "size": STATE["info"]["bytes"]}]})
        self.send(404, {"error": "not found"})

    def do_POST(self):
        try:
            b = self.body()
            t0 = time.time()
            if self.path == "/query":
                mode, q = b.get("mode", "free"), b.get("query", "")
                if mode == "auto":
                    mode, q = detect_mode(q)
                if mode not in infer.QUERY_TEMPLATES:
                    return self.send(400, {"error": "mode must be one of %s or auto" % list(infer.QUERY_TEMPLATES)})
                prompt = infer.QUERY_TEMPLATES[mode].format(query=q)
                text = run(prompt, **params(b))
                return self.send(200, {"mode": mode, "prompt": prompt, "text": text, "seconds": round(time.time() - t0, 1)})
            if self.path in ("/v1/completions", "/api/generate"):
                prompt = b.get("prompt", "")
                text = run(prompt, **params(b))
                return self.completion(b, text)
            if self.path in ("/v1/chat/completions", "/api/chat"):
                mode, q = detect_mode(last_user(b.get("messages")))
                prompt = infer.QUERY_TEMPLATES[mode].format(query=q)
                text = run(prompt, **params(b))
                return self.completion(b, text, chat=True)
            self.send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self.send(500, {"error": "%s: %s" % (type(e).__name__, e)})

    def completion(self, b, text, chat=False):
        ollama = self.path.startswith("/api/")
        if ollama:
            stream = b.get("stream", True)  # Ollama streams by default
            msg = {"role": "assistant", "content": text}
            full = {"model": MODEL_NAME, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "done": True}
            full.update({"message": msg} if chat else {"response": text})
            if not stream:
                return self.send(200, full)
            first = dict(full, done=False)
            last = {k: v for k, v in full.items() if k not in ("message", "response")}
            last.update({"message": {"role": "assistant", "content": ""}} if chat else {"response": ""})
            data = (json.dumps(first) + "\n" + json.dumps(last) + "\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)
        cid, now = "cmpl-%d" % int(time.time() * 1000), int(time.time())
        choice = {"index": 0, "finish_reason": "stop"}
        if chat:
            choice["message"] = {"role": "assistant", "content": text}
        else:
            choice["text"] = text
        if b.get("stream"):
            delta = dict(choice, finish_reason=None)
            if chat:
                delta["delta"] = delta.pop("message")
            chunk = {"id": cid, "object": "chat.completion.chunk" if chat else "text_completion", "created": now,
                     "model": MODEL_NAME, "choices": [delta]}
            data = ("data: %s\n\ndata: [DONE]\n\n" % json.dumps(chunk)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)
        self.send(200, {"id": cid, "object": "chat.completion" if chat else "text_completion", "created": now,
                        "model": MODEL_NAME, "choices": [choice],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=infer.DEFAULT_CHECKPOINT)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.environ.get("MYTHOS_PORT", "8830")))
    a = p.parse_args()
    print("Loading %s ..." % a.checkpoint, flush=True)
    model, tok, device, cfg, ckpt = infer.load_model(a.checkpoint)
    STATE["model"] = dict(model=model, tok=tok, device=device, cfg=cfg)
    STATE["info"] = {"checkpoint": a.checkpoint, "step": ckpt.get("step"), "val_loss": ckpt.get("val_loss"),
                     "device": str(device), "bytes": os.path.getsize(
                         a.checkpoint if "/" in a.checkpoint else "%s/%s" % (infer.paths.CHECKPOINT_DIR, a.checkpoint))}
    print("Serving %s on http://%s:%d" % (MODEL_NAME, a.host, a.port), flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
