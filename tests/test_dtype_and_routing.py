"""
Regression tests for defects found by the first GPU test run:

- casting the whole model to a half-precision dtype crashed (the ACT weights
  promoted the residual stream back to float32) and silently destroyed the
  complex RoPE tables;
- the loop-index signal swamped freshly initialized hidden states, so every
  token looked alike inside the recurrent block and MoE routing collapsed;
- the flash-attn path was taken for CPU tensors whenever flash-attn was installed.
"""

import pytest
import torch
import torch.nn.functional as F

import open_mythos.main as main_mod
from open_mythos.main import MythosConfig, OpenMythos

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_cfg(attn_type: str = "gqa", **overrides) -> MythosConfig:
    kw = dict(
        vocab_size=500,
        dim=128,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=96,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        attn_type=attn_type,
        n_experts=8,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=32,
        lora_rank=8,
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
    )
    kw.update(overrides)
    return MythosConfig(**kw)


def _bf16_supported() -> bool:
    """bf16 matmul/softmax exist on CUDA and on recent CPU builds; probe once."""
    try:
        x = torch.randn(2, 4, 8, device=DEVICE, dtype=torch.bfloat16)
        F.softmax(x @ x.transpose(-1, -2), dim=-1)
        return True
    except (RuntimeError, TypeError):
        return False


needs_bf16 = pytest.mark.skipif(
    not _bf16_supported(), reason=f"bfloat16 ops unavailable on {DEVICE}"
)


class TestDtypeCast:
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float64])
    def test_rope_tables_stay_complex(self, dtype):
        model = OpenMythos(make_cfg()).to(dtype=dtype)
        assert model.freqs_cis.dtype == torch.complex64
        assert model.freqs_cis_mla.dtype == torch.complex64
        # unit phasors: an imaginary part that was dropped would break this
        assert torch.allclose(model.freqs_cis.abs(), torch.ones(()), atol=1e-6)
        assert model.freqs_cis.imag.abs().sum() > 0

    def test_rope_tables_follow_device(self):
        model = OpenMythos(make_cfg()).to(device=DEVICE, dtype=torch.float32)
        assert model.freqs_cis.device.type == DEVICE
        assert model.freqs_cis.is_complex()

    @needs_bf16
    @pytest.mark.parametrize("attn_type", ["gqa", "mla"])
    def test_bf16_model_runs_and_matches_fp32(self, attn_type):
        torch.manual_seed(0)
        model = OpenMythos(make_cfg(attn_type)).to(DEVICE).eval()
        ids = torch.randint(0, 500, (2, 24), device=DEVICE)
        with torch.no_grad():
            ref = model(ids, n_loops=4)
            half = model.to(dtype=torch.bfloat16)(ids, n_loops=4)
        assert half.dtype == torch.bfloat16
        assert not torch.isnan(half).any()
        # Same weights at lower precision: logits must point the same way.
        # A model whose RoPE had been reduced to its real part fails this.
        cos = F.cosine_similarity(ref.flatten(1), half.float().flatten(1), dim=-1)
        assert cos.min().item() > 0.98

    @needs_bf16
    def test_bf16_generate_and_cache(self):
        torch.manual_seed(0)
        model = OpenMythos(make_cfg()).to(device=DEVICE, dtype=torch.bfloat16).eval()
        ids = torch.randint(0, 500, (2, 8), device=DEVICE)
        out = model.generate(ids, max_new_tokens=4, n_loops=4)
        assert out.shape == (2, 12)

    @needs_bf16
    def test_recurrent_output_keeps_activation_dtype(self):
        cfg = make_cfg()
        model = OpenMythos(cfg).to(device=DEVICE, dtype=torch.bfloat16)
        h = torch.randn(2, 8, cfg.dim, device=DEVICE, dtype=torch.bfloat16)
        out = model.recurrent(h, h, model.freqs_cis[:8], n_loops=3)
        assert out.dtype == torch.bfloat16


class TestRecurrentInputScale:
    def _router_inputs(self, model, ids):
        seen = []

        def hook(_mod, inputs):
            seen.append(inputs[0].detach().flatten(0, 1))

        handle = model.recurrent.block.ffn.register_forward_pre_hook(hook)
        model(ids)
        handle.remove()
        return seen[0]

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_tokens_stay_distinguishable_at_init(self, seed):
        torch.manual_seed(seed)
        model = OpenMythos(make_cfg()).train()
        ids = torch.randint(0, 500, (4, 32))
        v = F.normalize(self._router_inputs(model, ids), dim=-1)
        cos = v @ v.T
        off_diag = cos[~torch.eye(len(v), dtype=torch.bool)]
        # was ~0.96 when the loop-index signal was added before the norm
        assert off_diag.mean().item() < 0.5

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_routing_not_collapsed_at_init(self, seed):
        torch.manual_seed(seed)
        model = OpenMythos(make_cfg()).train()
        model(torch.randint(0, 500, (4, 32)))
        load = model.recurrent.block.ffn.expert_load
        share = load / load.sum()
        assert (load > 0).all(), f"unused experts at init: {load.tolist()}"
        # two experts used to take ~50% each; uniform would be 12.5%
        assert share.max().item() < 0.35

    def test_loop_index_still_changes_the_block_input(self):
        """The loop signal must survive the reordering: different depth, different output."""
        torch.manual_seed(0)
        cfg = make_cfg()
        model = OpenMythos(cfg).eval()
        h = torch.randn(1, 4, cfg.dim)
        with torch.no_grad():
            one = model.recurrent(h, h, model.freqs_cis[:4], n_loops=1)
            three = model.recurrent(h, h, model.freqs_cis[:4], n_loops=3)
        assert not torch.allclose(one, three)


class TestFlashAttnGuard:
    def test_cpu_tensors_never_reach_flash_attn(self, monkeypatch):
        """flash-attn is CUDA-only; having it installed must not break CPU models."""

        def boom(*_a, **_k):
            raise AssertionError("flash_attn_func called with CPU tensors")

        monkeypatch.setattr(main_mod, "_HAS_FLASH_ATTN", True)
        monkeypatch.setattr(main_mod, "flash_attn_func", boom, raising=False)
        model = OpenMythos(make_cfg("gqa")).eval()
        ids = torch.randint(0, 500, (2, 8))
        with torch.no_grad():
            logits = model(ids)
        assert logits.shape == (2, 8, 500)
