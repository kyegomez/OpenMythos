# Copilot Instructions

## Build, test, and lint commands

- Install dependencies for local work with `poetry install --with dev`. The repo also keeps a minimal `requirements.txt` for `pip install -r requirements.txt`.
- Build the package with `poetry build`.
- Run the main unit suite with `poetry run pytest tests/test_main.py`.
- Run a single test with `poetry run pytest tests/test_main.py::TestOpenMythosGQA::test_generate_shape`.
- Run tokenizer tests with `poetry run pytest tests/test_tokenizer.py`. These load the default Hugging Face tokenizer (`openai/gpt-oss-20b`), so they are heavier than `tests/test_main.py`.
- Run the RoPE debug script with `poetry run python tests/test_rope_debug.py`.
- Lint with `poetry run ruff check .`.
- Check formatting with `poetry run black --check .`.
- The documented training entrypoints are `poetry run python training/3b_fine_web_edu.py` and `poetry run torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") training/3b_fine_web_edu.py`.
- Benchmark utilities are executable scripts under `tests/`, not pytest files: `poetry run python tests/small_benchmark.py` and `poetry run python tests/bench_vs_transformer.py`.

## High-level architecture

- `open_mythos/main.py` is the primary implementation. It defines `MythosConfig`, both attention backends, the MoE FFN, the recurrent block, and the top-level `OpenMythos` model.
- The model pipeline is fixed: token embedding -> Prelude (`cfg.prelude_layers` dense `TransformerBlock`s run once) -> Recurrent Block (one shared `TransformerBlock` looped `n_loops` times with LoRA, ACT halting, and LTI-stable input injection) -> Coda (`cfg.coda_layers` dense `TransformerBlock`s run once) -> `RMSNorm` -> tied LM head.
- The key architectural invariant is that the encoded input `e` is frozen after the Prelude and injected on every recurrent iteration. The recurrent update path is `loop_index_embedding -> TransformerBlock(use_moe=True) -> LoRAAdapter -> LTIInjection -> ACTHalting`.
- `cfg.attn_type` switches the whole attention/cache path. `GQAttention` stores full K/V by layer, while `MLAttention` stores compressed `c_kv` plus `k_rope` and reconstructs K/V on demand. `OpenMythos.forward()` also switches RoPE buffers based on `attn_type`.
- `open_mythos/variants.py` contains the named model scales (`mythos_1b` through `mythos_1t`) as `MythosConfig` factories. `open_mythos/tokenizer.py` is a small wrapper around `transformers.AutoTokenizer` with `openai/gpt-oss-20b` as the default tokenizer.
- `training/3b_fine_web_edu.py` is a standalone FSDP training script for the 3B variant. It streams FineWeb-Edu shards, uses `MythosTokenizer`, and keeps checkpointing/training logic out of the library module.
- `docs/open_mythos.md` is the detailed API and architecture reference. `docs/datasets.md` holds the training dataset recommendations and token-budget guidance referenced by the training script/README.
- `open_mythos/moda.py` is a separate experimental MoDA + MoE implementation, not part of the main `OpenMythos` export surface.

## Key conventions

- For tests, smoke runs, and examples that should execute quickly, use the tiny helper configs from `tests/test_main.py` (`gqa_cfg()` / `mla_cfg()`) or similarly small custom configs. `MythosConfig()` defaults to a much larger research model.
- Preserve decode-position semantics when touching inference code: `start_pos` chooses the RoPE slice for incremental decoding, and each cache entry uses a deterministic key (`prelude_{i}`, `recurrent_loop_{t}`, `coda_{i}`).
- Do not short-circuit recurrent execution when a KV cache is active. `RecurrentBlock` only exits early on ACT convergence when `kv_cache is None`; cached decoding relies on every loop depth writing its cache entry on every step.
- Keep the embedding/LM head weight tying intact (`self.head.weight = self.embed.weight`), and keep `router_bias` as a registered buffer rather than a trainable parameter.
- Prelude and Coda are always dense FFN blocks (`use_moe=False`); the recurrent block is the only place that uses the MoE FFN (`use_moe=True`).
- Named variants and default configs are MLA-first. If you switch code or tests to GQA, make sure the config still provides valid MLA fields because shared config helpers are reused across both modes.
- Use the variant helpers that actually exist in `open_mythos/variants.py`: `mythos_1b`, `mythos_3b`, `mythos_10b`, `mythos_50b`, `mythos_100b`, `mythos_500b`, and `mythos_1t`. The README currently shows a stale `mythos_7b()` example.
- The benchmark scripts live under `tests/` and are intended to be run directly as scripts. Their docstrings still reference a `benchmarks/` path that does not exist in this repository.
