# Getting Started with OpenMythos

Get OpenMythos running in under 5 minutes on any machine, including CPU-only
laptops and Intel Macs.

## 1. Install

```bash
pip install open-mythos
```

**Note on flash-attn:** `flash-attn` is optional and only available on CUDA
hardware. The model falls back to standard scaled-dot-product attention
automatically when it is absent — no code change needed.

For CUDA machines that want the ~30% throughput speedup:

```bash
pip install open-mythos[flash]
```

## 2. Instant smoke test with TINY_CONFIG

`TINY_CONFIG` is a pre-built preset (~1-3M parameters, CPU-safe) included for
quick verification that your install is working.

```python
import torch
from open_mythos.main import OpenMythos
from open_mythos.variants import TINY_CONFIG

# 1. Instantiate the model
model = OpenMythos(TINY_CONFIG)
model.eval()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# 2. Create a random input batch  (batch=1, sequence_length=16)
ids = torch.randint(0, TINY_CONFIG.vocab_size, (1, 16))

# 3. Run a forward pass (2 recurrent loop iterations)
with torch.no_grad():
    logits = model(ids, n_loops=TINY_CONFIG.max_loop_iters)

# 4. Verify output shape: (batch, seq_len, vocab_size)
print(f"Output shape: {list(logits.shape)}")
assert logits.shape == (1, 16, TINY_CONFIG.vocab_size)

# 5. Verify LTI stability: spectral radius of A must be < 1
A_diag = model.recurrent.injection.get_A()
rho = A_diag.abs().max().item()
print(f"Spectral radius rho(A) = {rho:.6f}  (must be < 1)")
assert rho < 1.0

# 6. Quick backward pass check
model.train()
logits2 = model(ids, n_loops=TINY_CONFIG.max_loop_iters)
logits2.mean().backward()
grad_norm = sum(
    p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
) ** 0.5
print(f"Gradient norm: {grad_norm:.4f}  (should be > 0)")
assert grad_norm > 0

print("All checks passed.")
```

Expected output (exact numbers will vary):

```
Parameters: 1,234,567
Output shape: [1, 16, 1000]
Spectral radius rho(A) = 0.412345  (must be < 1)
Gradient norm: 3.2109  (should be > 0)
All checks passed.
```

## 3. Use a production-scale config

Switch from `TINY_CONFIG` to one of the named presets for real experiments:

```python
from open_mythos.variants import mythos_1b
from open_mythos.main import OpenMythos

cfg = mythos_1b()          # 1B-parameter research config
model = OpenMythos(cfg)
```

See [variants.py](../open_mythos/variants.py) for the full list:
`mythos_1b`, `mythos_3b`, `mythos_10b`, `mythos_50b`, `mythos_100b`,
`mythos_500b`, `mythos_1t`.

## 4. Generate text

```python
import torch
from open_mythos.main import OpenMythos
from open_mythos.variants import TINY_CONFIG

model = OpenMythos(TINY_CONFIG)
model.eval()

prompt = torch.randint(0, TINY_CONFIG.vocab_size, (1, 8))
with torch.no_grad():
    output = model.generate(prompt, max_new_tokens=16, n_loops=4)

print(f"Generated shape: {list(output.shape)}")  # [1, 24]
```

## 5. Run the test suite

```bash
pip install pytest
pytest tests/ -v
```

The tests use `TINY_CONFIG` so they complete in seconds on CPU.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pip install` fails with `torch==2.11.0 not found` | Upgrade to `open-mythos>=0.5.1`; the pin is fixed |
| `ImportError: flash_attn` | Not needed — the model falls back automatically. Only install it on CUDA. |
| `RuntimeError: eigvals expects 2-D input` | Use `get_A().abs().max()` as shown above, not `torch.linalg.eigvals(get_A())` |
| Tests fail with OOM | Tests use `TINY_CONFIG` and should fit in < 200 MB RAM |
