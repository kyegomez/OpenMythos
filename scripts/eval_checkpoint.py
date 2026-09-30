#!/usr/bin/env python3
"""
Factual eval harness for OpenMythos / Sentinel Grid checkpoints.

Replaces compare_ckpts.py's side-by-side eyeballing with an actual score.
This exists because checkpoints/README.txt documents a real finding: lower
validation loss does NOT mean better factual recall on this model
(extended_best.pt has lower val loss than massive_best.pt but worse facts).
Val loss alone can't be trusted to pick a checkpoint -- this can.

Each eval item is a (query-mode, query, expected keywords) triple verified
against the actual training corpus (cyber_corpus/massive_corpus.jsonl /
extended_corpus.jsonl) -- e.g. querying T1003 should mention "LSASS" or
"credential dumping" because that's what the corpus says about it. A
checkpoint scores a point per item where at least one expected keyword shows
up (case-insensitive) in its generated completion.

Examples:
  python3 scripts/eval_checkpoint.py massive_best.pt
  python3 scripts/eval_checkpoint.py massive_best.pt extended_best.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infer import load_model, query  # noqa: E402

# (query_mode, query, [expected keywords, any one match scores the item])
EVAL_SET = [
    ("technique", "T1003.001", ["LSASS", "credential"]),
    ("technique", "T1003.006", ["DCSync", "credential"]),
    ("sub", "T1566.002", ["Spearphishing", "phishing", "link"]),
    ("tool", "Mimikatz", ["credential", "dumping", "password"]),
    ("actor", "APT29", ["SolarWinds", "SAML", "Cloud"]),
    ("actor", "Lazarus Group", ["RawDisk", "Blockbuster", "Novetta"]),
    ("procedure", "BlackByte T1021.001", ["RDP", "Remote Desktop"]),
    ("mitigation", "Data Obfuscation", ["intrusion detection", "network signature", "signature"]),
    ("cve", "CVE-2021-44228", ["Log4", "exploit", "vulnerab"]),
    ("technique", "T1082", ["System Information", "computer name", "OS version"]),
]


def score_checkpoint(checkpoint_name, max_tokens=100, loops=8, temp=0.3, top_k=20):
    print(f"Loading {checkpoint_name} ...", end=" ", flush=True)
    model, tokenizer, device, cfg, ckpt = load_model(checkpoint_name)
    print(f"OK (step {ckpt['step']}, val_loss {ckpt.get('val_loss', 0):.4f})")

    hits = 0
    rows = []
    for mode, q, expected in EVAL_SET:
        prompt, completion = query(
            model, tokenizer, device, cfg, mode, q,
            max_tokens=max_tokens, loops=loops, temp=temp, top_k=top_k,
        )
        hit = any(kw.lower() in completion.lower() for kw in expected)
        hits += hit
        rows.append((mode, q, hit, completion.strip().replace("\n", " ")[:100]))

    score = hits / len(EVAL_SET)
    return score, ckpt, rows


def print_report(name, score, ckpt, rows):
    print(f"\n{'='*78}")
    print(f"{name}  ->  factual score: {score:.0%}  ({sum(r[2] for r in rows)}/{len(rows)})"
          f"   step {ckpt['step']}  val_loss {ckpt.get('val_loss', 0):.4f}")
    print(f"{'='*78}")
    for mode, q, hit, snippet in rows:
        mark = "PASS" if hit else "FAIL"
        print(f"  [{mark}] {mode:<10} {q:<20} {snippet}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+", help="One or more checkpoint filenames/paths to score and compare")
    p.add_argument("--tokens", type=int, default=100)
    p.add_argument("--loops", type=int, default=8)
    p.add_argument("--temp", type=float, default=0.3, help="Low temperature for reproducible factual scoring")
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()

    results = []
    for ckpt_name in args.checkpoints:
        score, ckpt, rows = score_checkpoint(
            ckpt_name, max_tokens=args.tokens, loops=args.loops, temp=args.temp, top_k=args.top_k
        )
        print_report(ckpt_name, score, ckpt, rows)
        results.append((ckpt_name, score, ckpt.get("val_loss", 0)))

    if len(results) > 1:
        print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
        for name, score, val_loss in sorted(results, key=lambda r: -r[1]):
            print(f"  {name:<30} factual: {score:.0%}   val_loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
