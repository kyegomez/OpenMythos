#!/usr/bin/env python3
"""OpenMythos 3B pretraining entry (Stage 4.5). Thin Trainer + eval orchestration."""
from __future__ import annotations
import os, random
from contextlib import nullcontext
from dataclasses import replace
import torch, torch.distributed as dist
from loguru import logger
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from open_mythos import OpenMythos
from open_mythos.main import RecurrentBlock, TransformerBlock
from open_mythos.tokenizer import MythosTokenizer
from open_mythos.variants import mythos_3b
from training.data import _validate_vocab_compatibility, build_dataloader
from training.eval import PeriodicEvalHook, build_validation_loader
from training.trainer import Trainer, _list_ckpts, apply_grad_checkpoint_if_enabled, load_checkpoint

SEED, SEQ_LEN, MB, TARGET_TOKENS, WARMUP, LR, WD = 42, 2048, 4, 30_000_000_000, 2000, 3e-4, 0.1
LOG_EVERY, CKPT_EVERY, EVAL_EVERY, CKPT_DIR, SUBSET = 10, 1000, 1000, "checkpoints", "sample-10BT"


def main() -> None:
    """Entry: `python -m training.run_3b` · `torchrun --nproc_per_node=N -m training.run_3b`."""
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank, li, ws = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
        device = f"cuda:{li}"; torch.cuda.set_device(device)
    else:
        rank, li, ws, device = 0, 0, 1, "cuda" if torch.cuda.is_available() else "cpu"
    master = rank == 0
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); random.seed(SEED)
    enc = MythosTokenizer()
    cfg = replace(mythos_3b(), vocab_size=enc.vocab_size, max_seq_len=SEQ_LEN)
    _validate_vocab_compatibility(enc, cfg)
    dt = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = OpenMythos(cfg); apply_grad_checkpoint_if_enabled(model)  # no-op when USE_GRAD_CHECKPOINT=False
    if ddp:
        model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD,
                     mixed_precision=MixedPrecision(param_dtype=dt, reduce_dtype=dt, buffer_dtype=dt),
                     auto_wrap_policy=ModuleWrapPolicy({TransformerBlock, RecurrentBlock}), device_id=li)
        amp = nullcontext()
    else:
        model = model.to(device)
        amp = torch.amp.autocast(device_type="cuda", dtype=dt) if "cuda" in device else nullcontext()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95), fused=True)
    ga = max(1, 256 // (ws * MB)); gbt = ws * MB * ga * SEQ_LEN; total = TARGET_TOKENS // gbt
    ckpts = _list_ckpts(CKPT_DIR)
    start = load_checkpoint(model, opt, ckpts[-1], ddp, cfg=cfg) if ckpts else 0
    if master and start: logger.success(f"Resumed at step {start}")
    hook = PeriodicEvalHook(EVAL_EVERY, build_validation_loader(enc, rank, ws, SEQ_LEN, SUBSET, MB), device, cfg.vocab_size, logger_obj=logger)
    Trainer(cfg=cfg, model=model, optimizer=opt, dataloader=build_dataloader(enc, SEQ_LEN, SUBSET, rank, ws, MB),
            device=device, local_rank=li, rank=rank, world_size=ws, ddp=ddp, master=master, amp_ctx=amp,
            vocab_size=cfg.vocab_size, ckpt_dir=CKPT_DIR, total_steps=total, log_every=LOG_EVERY, ckpt_every=CKPT_EVERY,
            grad_accum_steps=ga, warmup_steps=WARMUP, max_lr=LR, global_batch_tok=gbt, eval_hook=hook).run(start_step=start)


if __name__ == "__main__":
    main()
