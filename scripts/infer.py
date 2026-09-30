#!/usr/bin/env python3
"""
Consolidated OpenMythos / Sentinel Grid inference CLI.

Replaces inference.py, inference_fixed.py, inference_extended.py,
inference_final.py, inference_massive.py, inference_tools.py,
cyber_inference.py, and sentinel_query.py (now in archive/) with one tool.

Unlike the old scripts, the model architecture is rebuilt from the config
saved *inside* the checkpoint (`ckpt["config"]`) rather than re-hardcoded per
script -- so this works unmodified against any checkpoint, old or new.

Two ways to prompt:
  --mode query    structured, per-topic prompt templates (the sentinel_query.py
                  behavior -- empirically the best for factual recall, see
                  checkpoints/README.txt)
  --mode generate free-form: your --prompt text is passed straight to the model

Examples:
  python3 scripts/infer.py --mode query --query-mode technique --query T1003
  python3 scripts/infer.py --mode generate --prompt "Explain lateral movement"
  python3 scripts/infer.py --interactive
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths  # noqa: E402  (must run before transformers import, sets HF_HOME)

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from open_mythos.main import OpenMythos, MythosConfig  # noqa: E402

DEFAULT_CHECKPOINT = "massive_best.pt"

QUERY_TEMPLATES = {
    "technique": "MITRE ATT&CK Technique {query}:",
    "tool": "Offensive Security Tool: {query}\n",
    "malware": "Malware Family: {query}\n",
    "cve": "Known Exploited Vulnerability: {query}\n",
    "actor": "Threat Actor Group: {query}\n",
    "procedure": "Procedure Example: {query}\n",
    "sub": "MITRE ATT&CK Sub-technique {query}:",
    "mitigation": "MITRE ATT&CK Mitigation: {query}\n",
    "capec": "CAPEC-{query}:",
    "free": "{query}",
}

QUERY_MODE_HELP = {
    "technique": "ATT&CK technique by ID    e.g. T1003, T1566.001",
    "tool": "Offensive tool by name    e.g. mimikatz, cobalt strike",
    "malware": "Malware family by name    e.g. emotet, lockbit",
    "cve": "CVE or KEV entry          e.g. CVE-2021-44228",
    "actor": "Threat actor group        e.g. APT29, Lazarus Group",
    "procedure": "Procedure example         e.g. 'APT29 T1566'",
    "sub": "ATT&CK sub-technique      e.g. T1059.001",
    "mitigation": "ATT&CK mitigation         e.g. 'Multi-factor Authentication'",
    "capec": "CAPEC attack pattern      e.g. 98",
    "free": "Free-form prompt          e.g. any text",
}


def load_model(checkpoint_name: str):
    """Load a model + tokenizer from a checkpoint, rebuilding the architecture
    from the config stored inside the checkpoint itself."""
    checkpoint_path = checkpoint_name
    if "/" not in checkpoint_name:
        checkpoint_path = f"{paths.CHECKPOINT_DIR}/{checkpoint_name}"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = MythosConfig(**ckpt["config"])
    model = OpenMythos(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer, device, cfg, ckpt


def generate(model, tokenizer, device, cfg, prompt,
             max_tokens=120, loops=8, temp=0.7, top_k=40):
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_tokens,
            n_loops=loops, temperature=temp, top_k=top_k,
        )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def query(model, tokenizer, device, cfg, query_mode, q, **gen_kwargs):
    template = QUERY_TEMPLATES.get(query_mode, QUERY_TEMPLATES["free"])
    prompt = template.format(query=q)
    completion = generate(model, tokenizer, device, cfg, prompt, **gen_kwargs)
    return prompt, completion


def print_result(prompt, completion):
    print(f"\n{'-'*70}")
    print(f"{prompt}{completion}")
    print(f"{'-'*70}\n")


def interactive_loop(model, tokenizer, device, cfg):
    print("\nSENTINEL GRID -- interactive query")
    for m, h in QUERY_MODE_HELP.items():
        print(f"  {m:<12} {h}")
    print("  Commands: '<mode> <query>' | 'loops N' | 'tokens N' | 'temp F' | 'quit'\n")

    max_tokens, loops, temp = 120, 8, 0.7
    modes = list(QUERY_TEMPLATES.keys())

    while True:
        try:
            raw = input("sentinel> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            break
        if raw.lower().startswith("loops "):
            loops = int(raw.split()[1]); continue
        if raw.lower().startswith("tokens "):
            max_tokens = int(raw.split()[1]); continue
        if raw.lower().startswith("temp "):
            temp = float(raw.split()[1]); continue

        parts = raw.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() in modes:
            mode, q = parts[0].lower(), parts[1]
        else:
            q, mode = raw, "free"
            if q.upper().startswith("T1") and "." not in q:
                mode = "technique"
            elif q.upper().startswith("T1"):
                mode = "sub"
            elif q.upper().startswith("CVE-"):
                mode = "cve"
            elif q.upper().startswith("CAPEC-"):
                mode, q = "capec", q.split("-", 1)[1]

        prompt, completion = query(model, tokenizer, device, cfg, mode, q,
                                    max_tokens=max_tokens, loops=loops, temp=temp)
        print_result(prompt, completion)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                    help="Checkpoint filename (resolved inside CHECKPOINT_DIR) or full path")
    p.add_argument("--mode", choices=["query", "generate"], default="query")
    p.add_argument("--query-mode", choices=list(QUERY_TEMPLATES.keys()), default="free")
    p.add_argument("--query", type=str, default="", help="Query text for --mode query")
    p.add_argument("--prompt", type=str, default="", help="Raw prompt for --mode generate")
    p.add_argument("--loops", type=int, default=8)
    p.add_argument("--tokens", type=int, default=120)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--interactive", action="store_true")
    args = p.parse_args()

    print(f"Loading {args.checkpoint} ...", end=" ", flush=True)
    model, tokenizer, device, cfg, ckpt = load_model(args.checkpoint)
    print(f"OK (step {ckpt['step']}, val_loss {ckpt.get('val_loss', 0):.4f})\n")

    if args.interactive or (not args.query and not args.prompt):
        interactive_loop(model, tokenizer, device, cfg)
        return

    gen_kwargs = dict(max_tokens=args.tokens, loops=args.loops, temp=args.temp, top_k=args.top_k)
    if args.mode == "generate":
        completion = generate(model, tokenizer, device, cfg, args.prompt, **gen_kwargs)
        print_result(args.prompt, completion)
    else:
        prompt, completion = query(model, tokenizer, device, cfg, args.query_mode, args.query, **gen_kwargs)
        print_result(prompt, completion)


if __name__ == "__main__":
    main()
