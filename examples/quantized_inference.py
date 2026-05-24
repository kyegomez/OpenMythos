"""
Quantized Inference Example for OpenMythos.

Demonstrates running mythos_1b with INT4 quantization and expert offloading
on consumer hardware (RTX 3060 12GB).

Usage:
    python examples/quantized_inference.py
"""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_mythos import OpenMythos, mythos_1b
from open_mythos.quantization import quantize_model, print_quantization_summary
from open_mythos.expert_offloader import ExpertOffloader


def main():
    print("=" * 60)
    print("OpenMythos Quantized Inference Demo")
    print("=" * 60)

    # 1. Create model
    print("\n[1/5] Creating mythos_1b model...")
    cfg = mythos_1b()
    model = OpenMythos(cfg)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Quantize to INT4
    print("\n[2/5] Quantizing to INT4 (expert FFN layers only)...")
    model = quantize_model(model, bits=4, group_size=128)
    print_quantization_summary(model)

    # 3. Setup expert offloading
    print("\n[3/5] Setting up expert offloading...")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    if torch.cuda.is_available():
        offloader = ExpertOffloader(
            model,
            gpu_experts=4,    # Keep 4 experts on GPU
            cache_experts=16, # Keep 16 in CPU RAM
        )
        offloader.prepare()
        print(f"  GPU experts: 4 | CPU cache: 16 | Disk: rest")
    else:
        print("  Running on CPU (no offloading needed)")

    # 4. Generate text
    print("\n[4/5] Generating text...")
    input_ids = torch.randint(0, cfg.vocab_size, (1, 32))
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    # Warmup
    _ = model.generate(input_ids, max_new_tokens=4, n_loops=2)

    # Benchmark
    start = time.time()
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=64, n_loops=4)
    elapsed = time.time() - start

    tokens_generated = output.shape[1] - input_ids.shape[1]
    tokens_per_sec = tokens_generated / elapsed

    print(f"  Generated {tokens_generated} tokens in {elapsed:.2f}s")
    print(f"  Speed: {tokens_per_sec:.1f} tokens/sec")

    # 5. Memory usage
    print("\n[5/5] Memory usage:")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        print(f"  GPU allocated: {allocated:.1f} MB")
        print(f"  GPU reserved:  {reserved:.1f} MB")

    if torch.cuda.is_available():
        print(f"\nOffloader stats:")
        offloader.print_stats()

    print("\n" + "=" * 60)
    print("Done! Model runs successfully with INT4 quantization.")
    print("=" * 60)


if __name__ == "__main__":
    main()
