#!/usr/bin/env python3
"""
OpenMythos MCP server (stdio, stdlib only). Thin client for scripts/mythos_server.py.

Start the model server first:  mythos start
Register with an agent:        mythos mcp-config
"""
import json
import os
import sys
import urllib.error
import urllib.request

URL = os.environ.get("MYTHOS_URL", "http://127.0.0.1:%s" % os.environ.get("MYTHOS_PORT", "8830"))
WARN = ("Small (279M) cybersecurity-domain model: fluent but frequently factually WRONG "
        "(e.g. wrong technique/actor attribution). Treat output as a draft and verify it.")
MODES = ["auto", "technique", "sub", "tool", "malware", "cve", "actor", "procedure", "mitigation", "capec", "free"]

TOOLS = [
    {"name": "mythos_query",
     "description": "Query the local OpenMythos model with a structured ATT&CK/CVE/tool/malware/actor prompt. " + WARN,
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "e.g. T1003, mimikatz, CVE-2021-44228, APT29"},
         "mode": {"type": "string", "enum": MODES, "default": "auto"},
         "max_tokens": {"type": "integer", "default": 120},
         "temperature": {"type": "number", "default": 0.3}}, "required": ["query"]}},
    {"name": "mythos_generate",
     "description": "Free-form completion from the local OpenMythos model. " + WARN,
     "inputSchema": {"type": "object", "properties": {
         "prompt": {"type": "string"},
         "max_tokens": {"type": "integer", "default": 120},
         "temperature": {"type": "number", "default": 0.3}}, "required": ["prompt"]}},
]


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def call(name, args):
    try:
        if name == "mythos_query":
            body = {"mode": args.get("mode", "auto"), "query": args["query"],
                    "max_tokens": args.get("max_tokens", 120), "temperature": args.get("temperature", 0.3)}
            r = post("/query", body)
            return r["prompt"] + r["text"]
        if name == "mythos_generate":
            r = post("/api/generate", {"prompt": args["prompt"], "stream": False,
                                       "max_tokens": args.get("max_tokens", 120),
                                       "temperature": args.get("temperature", 0.3)})
            return r["response"]
        raise ValueError("unknown tool %s" % name)
    except urllib.error.URLError:
        raise RuntimeError("OpenMythos server not reachable at %s. Run `mythos start` (model load takes ~30s)." % URL)


def reply(id_, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": id_}
    msg.update({"error": error} if error else {"result": result})
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            m = json.loads(line)
        except ValueError:
            continue
        id_, method = m.get("id"), m.get("method", "")
        if id_ is None:  # notification
            continue
        if method == "initialize":
            reply(id_, {"protocolVersion": (m.get("params") or {}).get("protocolVersion", "2024-11-05"),
                        "capabilities": {"tools": {}}, "serverInfo": {"name": "openmythos", "version": "1"}})
        elif method == "tools/list":
            reply(id_, {"tools": TOOLS})
        elif method == "tools/call":
            p = m.get("params") or {}
            try:
                text = call(p.get("name"), p.get("arguments") or {})
                reply(id_, {"content": [{"type": "text", "text": text}]})
            except Exception as e:  # noqa: BLE001
                reply(id_, {"content": [{"type": "text", "text": str(e)}], "isError": True})
        elif method == "ping":
            reply(id_, {})
        else:
            reply(id_, error={"code": -32601, "message": "method not found: %s" % method})


if __name__ == "__main__":
    main()
