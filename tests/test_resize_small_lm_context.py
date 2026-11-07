from pathlib import Path

import torch

from examples.resize_small_lm_context import build_resized_config, resize_checkpoint_context
from examples.train_small_lm import NoOpScaler, config_to_dict, save_checkpoint
from open_mythos import MythosConfig, OpenMythos


def tiny_config(max_seq_len: int) -> MythosConfig:
    return MythosConfig(
        vocab_size=64,
        dim=32,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=max_seq_len,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=2,
        n_shared_experts=1,
        n_experts_per_tok=1,
        expert_dim=32,
        lora_rank=4,
        dropout=0.0,
    )


def write_checkpoint(path: Path, cfg: MythosConfig) -> None:
    model = OpenMythos(cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    save_checkpoint(
        path,
        model,
        optimizer,
        NoOpScaler(),
        cfg,
        tokenizer="gpt2",
        step=123,
        loss=1.23,
        append_eos=True,
        eos_loss_weight=4.0,
        clean_text=True,
    )


def test_resize_checkpoint_context_expands_max_seq_len(tmp_path):
    source = tmp_path / "source.pt"
    target = tmp_path / "target.pt"
    write_checkpoint(source, tiny_config(max_seq_len=8))

    result = resize_checkpoint_context(source, target, max_seq_len=12)

    ckpt = torch.load(target, map_location="cpu", weights_only=False)
    assert result.source_max_seq_len == 8
    assert result.target_max_seq_len == 12
    assert ckpt["config"]["max_seq_len"] == 12
    assert ckpt["step"] == 123
    assert ckpt["append_eos"] is True
    assert ckpt["eos_loss_weight"] == 4.0
    assert ckpt["clean_text"] is True
    assert ckpt["context_resize"] == {
        "source_max_seq_len": 8,
        "target_max_seq_len": 12,
        "loaded": result.loaded,
        "skipped": result.skipped,
    }


def test_resize_checkpoint_context_rejects_non_expansion(tmp_path):
    source = tmp_path / "source.pt"
    target = tmp_path / "target.pt"
    write_checkpoint(source, tiny_config(max_seq_len=8))

    try:
        resize_checkpoint_context(source, target, max_seq_len=8)
    except ValueError as exc:
        assert "greater than source max_seq_len=8" in str(exc)
    else:
        raise AssertionError("expected non-expansion to fail")

    assert not target.exists()


def test_build_resized_config_preserves_all_other_fields():
    source_cfg = config_to_dict(tiny_config(max_seq_len=8))

    target_cfg = build_resized_config(source_cfg, max_seq_len=12)

    assert target_cfg["max_seq_len"] == 12
    assert {
        key: value for key, value in target_cfg.items() if key != "max_seq_len"
    } == {key: value for key, value in source_cfg.items() if key != "max_seq_len"}
