"""
OpenMythos loop-scaling evaluation.

Given a trained checkpoint, compute held-out perplexity on FineWeb-Edu's
validation stream while varying the number of inference-time recurrent
loops. This is the central test of the "more loops = deeper reasoning"
claim: a vanilla transformer cannot do this, a looped transformer should
show monotonically decreasing PPL that plateaus.

Also emits generation samples at different loop counts for qualitative
comparison.

Usage:
    python evaluate.py --ckpt /workspace/runs/looped_8/ckpt_30000.pt
    python evaluate.py --ckpt /workspace/runs/baseline_1/ckpt_30000.pt \\
        --loop_grid 1
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from open_mythos.main import OpenMythos, MythosConfig

from data import build_loader, get_tokenizer


@torch.no_grad()
def compute_ppl(model, loader_iter, n_loops: int, num_batches: int, vocab_size: int):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for _ in range(num_batches):
        x, y = next(loader_iter)
        x = x.to("cuda", non_blocking=True)
        y = y.to("cuda", non_blocking=True)
        logits = model(x, n_loops=n_loops)
        loss = F.cross_entropy(
            logits.float().view(-1, vocab_size),
            y.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += y.numel()
    return math.exp(total_loss / total_tokens), total_loss / total_tokens


@torch.no_grad()
def generate_sample(model, tokenizer, prompt: str, n_loops: int, max_new_tokens: int = 64):
    model.eval()
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    out = model.generate(ids, max_new_tokens=max_new_tokens, n_loops=n_loops, temperature=0.8, top_k=50)
    return tokenizer.decode(out[0].tolist(), skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--loop_grid", type=int, nargs="+", default=[1, 2, 4, 6, 8, 12, 16])
    ap.add_argument("--num_batches", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--tokenizer", type=str, default="gpt2")
    ap.add_argument("--output_json", type=str, default=None)
    ap.add_argument("--sample_prompts", type=str, nargs="+", default=[
        "The main function of mitochondria is to",
        "In physics, the second law of thermodynamics states that",
        "A short guide to writing clear English:",
    ])
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    print(f"==> loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    mcfg_dict = ckpt["mcfg"]
    mcfg = MythosConfig(**mcfg_dict)
    print(f"==> model config: dim={mcfg.dim} n_experts={mcfg.n_experts} "
          f"max_loop_iters={mcfg.max_loop_iters} attn_type={mcfg.attn_type}")

    model = OpenMythos(mcfg).to("cuda", torch.bfloat16)
    model.load_state_dict(ckpt["model"])
    print(f"==> loaded step={ckpt['step']} tokens={ckpt['tokens_seen']/1e6:.1f}M")

    tok = get_tokenizer(args.tokenizer)

    # Build a validation loader that skips past the training window so
    # evaluation never sees documents the model already trained on.
    # With 500M tokens trained and ~500 tokens/doc average, train consumed
    # ~1M docs; we skip 2M to be safe.
    val_loader = build_loader(
        tokenizer=tok,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="sample-10BT",
        num_workers=1,
        skip_docs=2_000_000,
    )

    # Cache a fixed evaluation set so each n_loops sees the same batches.
    print(f"==> caching {args.num_batches} eval batches to memory...")
    cached = []
    it = iter(val_loader)
    for i in range(args.num_batches):
        cached.append(next(it))
        if (i + 1) % 10 == 0:
            print(f"   cached {i+1}/{args.num_batches}")

    def cached_iter():
        for batch in cached:
            yield batch

    print(f"\n==> loop grid sweep: {args.loop_grid}")
    results = []
    for n_loops in args.loop_grid:
        ppl, nll = compute_ppl(
            model, cached_iter(), n_loops=n_loops,
            num_batches=args.num_batches, vocab_size=mcfg.vocab_size,
        )
        rho_A = model.recurrent.injection.get_A().max().item()
        print(f"n_loops={n_loops:2d}  ppl={ppl:7.3f}  nll={nll:.4f}  rho_A={rho_A:.4f}")
        results.append({"n_loops": n_loops, "ppl": ppl, "nll": nll, "rho_A": rho_A})

    print("\n==> generation samples (n_loops=trained / doubled)")
    samples = {}
    for n_loops in [mcfg.max_loop_iters, mcfg.max_loop_iters * 2]:
        samples[n_loops] = {}
        for prompt in args.sample_prompts:
            gen = generate_sample(model, tok, prompt, n_loops=n_loops, max_new_tokens=48)
            samples[n_loops][prompt] = gen
            print(f"\n[n_loops={n_loops}] {prompt!r}\n  -> {gen}")

    out_json = args.output_json or str(ckpt_path.parent / f"eval_{ckpt_path.stem}.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "ckpt": str(ckpt_path),
                "step": ckpt["step"],
                "tokens_seen": ckpt["tokens_seen"],
                "trained_with_max_loop_iters": mcfg.max_loop_iters,
                "loop_sweep": results,
                "samples": samples,
            },
            f,
            indent=2,
        )
    print(f"\n==> wrote {out_json}")


if __name__ == "__main__":
    main()
