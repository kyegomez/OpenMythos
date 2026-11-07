# OpenMythos 优化方案 — 分阶段执行路线

**配套:** [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md)
**原则:** 先堵静默失败 → 再加回归守护 → 再性能融合 → 最后可扩展性

所有提议改动均**可独立提交**,每项附:优先级 / 位置 / 预期收益 / 改动规模。

---

## 阶段 0 — 紧急修复(1 天内,阻塞其他工作)

### 0.1 [P0] 修复 `__init__.py` 幽灵导出
**位置:** `open_mythos/__init__.py:52-53`
**当前:** `__all__` 引用不存在的 `load_tokenizer` / `get_vocab_size`
**修复:** 从 `__all__` 删除这两个名字
**规模:** 2 行
**验证:** `python -c "from open_mythos import *"` 不再报错

### 0.2 [P0] 训练环无 NaN 守护
**位置:** `training/3b_fine_web_edu.py:499-506`
**修复:**
```python
loss_value = loss.item()
if not math.isfinite(loss_value):
    raise RuntimeError(f"Non-finite loss at step {step}: {loss_value}")
loss_accum += loss_value
```
同时在 checkpoint 保存前对 `p.isfinite().all()` 扫描所有参数。
**规模:** ~10 行
**影响:** 避免 NaN checkpoint 污染整个训练

### 0.3 [P0] MoE 除零保护
**位置:** `open_mythos/main.py:488`
**修复:** `topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp(min=1e-9)`
**规模:** 1 行 diff
**影响:** fp16/bf16 下 softmax 下溢不再产生 NaN

### 0.4 [P0] attn_type 字面量化
**位置:** `main.py:53, 621, 983`
**修复:**
```python
from typing import Literal
attn_type: Literal["gqa", "mla"] = "mla"

# __post_init__:
if self.attn_type not in ("gqa", "mla"):
    raise ValueError(f"attn_type must be 'gqa' or 'mla', got {self.attn_type!r}")
```
**规模:** ~6 行
**影响:** 错拼不再静默 fallback

### 0.5 [P0] 训练复现性
**位置:** `training/3b_fine_web_edu.py:main()` 开头
**修复:**
```python
seed = cfg.seed if hasattr(cfg, "seed") else 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed); numpy.random.seed(seed)
# DataLoader worker_init_fn 也要设 seed
```
**规模:** ~8 行
**影响:** 所有 bug 可复现,调试/A-B 测试可能

---

## 阶段 1 — 契约硬化(1 周内)

### 1.1 [P1] `MythosConfig` 冻结 + `__post_init__` 校验
**位置:** `main.py:9-74`
**修复:**
```python
@dataclass(frozen=True)
class MythosConfig:
    ...
    def __post_init__(self):
        assert self.dim % self.n_heads == 0, f"dim({self.dim}) must divide n_heads({self.n_heads})"
        if self.attn_type == "gqa":
            assert self.n_heads % self.n_kv_heads == 0
        assert self.n_experts_per_tok <= self.n_experts
        assert 0.0 < self.act_threshold <= 1.0
        assert self.attn_type in ("gqa", "mla")
        assert self.lora_rank > 0 and self.lora_rank <= self.dim
```
**调用点改动:** 训练脚本不能再 mutate `cfg.vocab_size`,改用 `dataclasses.replace(cfg, vocab_size=..., max_seq_len=...)`
**规模:** ~30 行(含调用点修复)

### 1.2 [P1] ACT halted 位置真正跳过计算
**位置:** `main.py:829-861` `RecurrentBlock.forward`
**当前:** halted 位置仍做 Transformer + LoRA + LTI 全部 flops,只在 weight 处门控
**修复方案 A (推荐):** 在 Transformer 前构造 `still_running` mask → 用 `mask.any(dim=-1)` 在 batch 维度筛选可继续的 token,传入子集
**修复方案 B (保守):** 至少把 `LoRAAdapter(trans_out)` 和 `LTIInjection` 的结果用 `still_running.unsqueeze(-1)` 广播截断,避免 halted 位置的 hidden state 继续演化(目前它们还在演化,影响后续 KV cache 写入)
**规模:** ~20 行
**影响:** 修复 A5 Finding #7 的正确性隐患

### 1.3 [P1] LoRA Extrapolation 显式 warn
**位置:** `main.py:588`
**修复:** 首次 `loop_t > max_t` 时 `warnings.warn_once(...)`,告知用户正在 depth extrapolation;不然深度 > max_loop_iters 时模型质量静默下降
**规模:** ~5 行

### 1.4 [P1] Tokenizer vs Model vocab_size 启动校验
**位置:** `training/3b_fine_web_edu.py:365-399`
**修复:** 模型构造后显式 `assert tokenizer.vocab_size == cfg.vocab_size`;load_checkpoint 读取 `ckpt["vocab_size"]` 与当前比较
**规模:** ~10 行

### 1.5 [P1] `generate()` 越界保护
**位置:** `main.py:1031-1047`
**修复:**
```python
if start_pos + 1 > self.cfg.max_seq_len:
    break  # 或 raise
```
**规模:** 3 行

---

## 阶段 2 — 回归测试补齐(1 周内)

**目标:** 最近 5 个修复的 bug 全部拥有专属回归测试。

### 2.1 ACT halting 权重总和测试
```python
def test_act_weights_sum_to_one():
    # 构造足够大的 n_loops,断言 Σ weight[t] ≈ 1.0 per position
    ...

def test_act_forced_halt_iter1_matches_iter1_hidden():
    # act_threshold = 0.01 → 所有 token 在 iter 1 就 halt
    # h_out 应该 ≈ h_after_iter1
    ...
```

### 2.2 KV cache loop-key 完整性测试
```python
def test_kv_cache_populates_all_loop_keys():
    cache = {}
    model(ids, kv_cache=cache, n_loops=4)
    for i in range(4):
        assert f"recurrent_loop_{i}" in cache
```

### 2.3 LoRA depth extrapolation bounds
```python
def test_lora_clamps_beyond_max_loops():
    lora = LoRAAdapter(dim=64, rank=4, max_loops=10)
    out_clamp = lora(x, loop_t=9)
    out_extrap = lora(x, loop_t=100)
    torch.testing.assert_close(out_clamp, out_extrap)  # 都应使用 index 9
```

### 2.4 RoPE decode 逐步 start_pos 推进
```python
def test_rope_generate_uses_increasing_start_pos():
    # 手动跑 2 步 decode,捕获两次 apply_rope 的 freqs slice,断言不同
    ...
```

### 2.5 moda.py 烟雾测试
```python
def test_moda_forward_shapes():
    cfg = MoDAConfig(vocab_size=100, d_model=64, ...)
    model = MoDAModel(cfg)
    logits, loss = model(ids, labels=ids)
    assert logits.shape == (B, T, 100)
    assert loss.dim() == 0
```

### 2.6 `variants.py` 所有 preset 可构造
```python
@pytest.mark.parametrize("variant", [mythos_1b, mythos_3b, mythos_10b, ...])
def test_variant_constructible(variant):
    cfg = variant()
    assert isinstance(cfg, MythosConfig)
    # __post_init__ 已经校验了所有不变量
```

### 2.7 训练最小端到端
```python
def test_overfit_single_batch():
    # 一个 batch,10 步 AdamW,断言 loss 单调下降
    ...

def test_checkpoint_roundtrip():
    save → load → 所有参数 torch.equal
    ...
```

### 2.8 修复 `tests/test_rope_debug.py`
把 195 行 `print` 脚本转成 `def test_*` 函数,替换 `print` 为 `assert` / `torch.testing.assert_close`。

**阶段 2 总规模:** 新增 ~300 行测试,从 ~40% 行覆盖提至 ~70%+,回归防护从 1/5 提至 5/5。

---

## 阶段 3 — 性能融合(2 周内,按收益排序)

### 3.1 [最高收益] FlashAttention via SDPA
**位置:** `main.py:242-246`(GQA)和 `main.py:384-388`(MLA)
**修复:**
```python
# GQAttention.forward
out = F.scaled_dot_product_attention(
    q, k, v,
    is_causal=(mask is not None and T > 1),
    dropout_p=self.attn_drop.p if self.training else 0.0,
    enable_gqa=True,  # PyTorch 2.5+,自动处理 GQA
)
```
对 MLA 类似,但要注意 q/k 的组合头维度。
**风险:** 需验证 `enable_gqa=True` 行为等价于 `repeat_interleave`(PyTorch ≥ 2.5)
**预期:** 注意力时间 −40~60%,峰值显存 −30~50%(长序列)

### 3.2 [次高收益] 向量化 MoE 分派
**位置:** `main.py:469-505`
**修复方案:** 采用 DeepSeek-V3 / Megablocks grouped-GEMM 模式:
```python
# 1. topk indices 展开
expert_idx = topk_idx.flatten()     # (B*T*topk,)
token_idx = arange(B*T).repeat_interleave(topk)
weights = topk_scores.flatten()

# 2. 按 expert 排序
sort_idx = expert_idx.argsort()
sorted_tokens = flat[token_idx[sort_idx]]
counts = torch.bincount(expert_idx, minlength=n_experts)

# 3. 按 expert 切片做 batched matmul(无 Python for-eid)
offsets = counts.cumsum(0)
# 用 torch.split + batched forward,或 torch._grouped_mm (PyTorch 2.4+)
```
**规模:** ~50 行,测试充分
**预期:** 3–5× MoE 时间,解锁 `torch.compile`

### 3.3 静态预分配 KV Cache
**位置:** `main.py:229-231, 365-367`
**修复:** 将 cache 从 dict-of-growing-tensors 改成:
```python
# 在 OpenMythos.__init__ 时
self.kv_cache_buf = {
    "k": torch.empty(max_bsz, max_seq_len, n_kv_heads, head_dim),
    "v": torch.empty(...),
}
# forward 中按 start_pos 切片写入,读取 [:start_pos+T]
```
**规模:** ~60 行(需要调整 cache 生命周期管理)

### 3.4 RoPE 去除 `.float()`
**位置:** `main.py:145-165 apply_rope`
**修复:** 参照 `moda.py:253-256` 的 rotate-half 实数实现,或保留 complex 路径但只在必要处 cast
**预期:** RoPE 带宽减半,bf16 连续

### 3.5 消除 ACT `.all()` CPU 同步
**位置:** `main.py:860`
**修复:**
- 训练:固定运行 `max_loop_iters` 步,权重门控已足够(去掉 early exit)
- 生成:改为每 `K` 步(K=4)检测一次,并用 `halted.all()` 的输出给 **下一轮** 的决策(延迟一步避免 sync)
**规模:** ~10 行

### 3.6 LoRA scale index 预分配
**位置:** `main.py:589`
**修复:**
```python
# __init__
self.register_buffer("_loop_indices", torch.arange(max_loops))

# forward
s = self.scale.weight[min(loop_t, self.scale.num_embeddings-1)]  # 直接索引,无 tensor 构造
```
**规模:** 5 行

### 3.7 `loop_index_embedding` 预计算
**位置:** `main.py:522-548`
**修复:** 在 `OpenMythos.__init__` 时预计算 `(max_loops, loop_dim)` buffer,`forward` 中索引
**规模:** ~15 行

### 3.8 训练脚本优化
- `optimizer.zero_grad(set_to_none=True)` 至 `training:480`
- 启用 gradient checkpointing: `torch.utils.checkpoint.checkpoint(recurrent_block.forward, ...)`
- `buf` 改用 `collections.deque` 或索引偏移避免 O(n) 切片(`training:104`)

---

## 阶段 4 — 架构重构(2~4 周)

### 4.1 拆分 `main.py` 为模块
```
open_mythos/
  __init__.py       (只暴露公共 API: MythosConfig, OpenMythos, MythosTokenizer, variants)
  config.py         MythosConfig + 校验
  primitives/
    norm.py         RMSNorm
    rope.py         precompute_rope_freqs, apply_rope, loop_index_embedding
  attention/
    base.py         Protocol
    gqa.py
    mla.py
  moe/
    expert.py
    router.py       MoEFFN(向量化后)
  recurrence/
    lti.py          LTIInjection
    act.py          ACTHalting
    lora.py         LoRAAdapter
    block.py        RecurrentBlock
  blocks.py         TransformerBlock
  model.py          OpenMythos
  generate.py       KV cache + generate loop
```
**每个文件 ≤ 400 行**,符合项目的 200-400 行典型 / 800 上限规则。

### 4.2 消除 `main.py` vs `moda.py` 重复
- 提取共享 `RMSNorm`/`Expert` 到 `primitives/`
- `moda.py` 的 `DeepSeekMoE` 比 `main.py` 的 `MoEFFN` 更完整(grouped routing + sigmoid + balance loss) — **反向迁移到主路径**
- 将 `moda.py` 移至 `open_mythos/experimental/moda.py`,显式标记实验性

### 4.3 定义 Protocol 扩展点
```python
class Attention(Protocol):
    def forward(self, x, freqs_cis, mask, kv_cache, cache_key) -> Tensor: ...

class Router(Protocol):
    def forward(self, x) -> tuple[Tensor, LoadBalanceMetrics]: ...

class Injection(Protocol):
    def forward(self, h, e, transformer_out) -> Tensor: ...
```
并通过 `build_attention(cfg) / build_router(cfg) / build_injection(cfg)` 工厂函数消除 `if cfg.attn_type` 分支。

### 4.4 Config 拆分嵌套
```python
@dataclass(frozen=True)
class AttentionConfig: ...
class MoEConfig: ...
class RecurrenceConfig: ...   # ACT + LoRA + LTI + max_loop_iters
@dataclass(frozen=True)
class MythosConfig:
    core: CoreConfig
    attn: AttentionConfig
    moe: MoEConfig
    recurrence: RecurrenceConfig
    def __post_init__(self): ...
```
保留扁平视图的 `MythosConfig.legacy()` 工厂,平滑过渡。

### 4.5 训练脚本模块化
```
training/
  __init__.py
  data.py          FineWebEduDataset, DataLoader factory
  trainer.py       FSDP setup, loop, eval, checkpoint
  eval.py          periodic validation, generation samples
  schedule.py      warmup + cosine
  run_3b.py        入口 — 只包含 cfg + Trainer(cfg).run()
```
**训练脚本从 551 行降至 ~60 行入口 + 4 个 ≤200 行模块。**

---

## 阶段 5 — 大规模训练能力(长期)

### 5.1 Tensor Parallelism hooks
- `nn.Linear` 替换为 `ColumnParallelLinear` / `RowParallelLinear` 抽象
- 与 FSDP 2D 并行集成

### 5.2 Expert Parallelism
- MoE experts 跨 rank 分片,`all_to_all` 分派
- 可复用 DeepSpeed-MoE / Megablocks 基础设施

### 5.3 Activation Checkpointing
- 默认对 `RecurrentBlock` 启用,16 次循环的激活显存减 16×

### 5.4 Paged KV Cache
- 参考 vLLM / Hugging Face 的 PagedAttention
- `n_loops × max_seq_len` 的 cache 规模必须分页

### 5.5 长上下文支持
- `_causal_mask` 改为 SDPA `is_causal=True` 自动处理,不再分配 `(T, T)` 矩阵
- 移除 `variants.py` 中 `max_seq_len=1_000_000` 的声明,降级为现实可支持的值

---

## 执行优先级总表

| 阶段 | 工作量 | 解锁能力 | 风险 |
|---|---|---|---|
| **阶段 0** | 1 天 | 消除静默失败 + checkpoint 污染 | 极低 |
| **阶段 1** | 3~5 天 | 契约硬化,配置错误立即失败 | 低(需改调用点) |
| **阶段 2** | 3~5 天 | 回归防护从 20% 提至 100% | 极低 |
| **阶段 3** | 1~2 周 | 训练吞吐 ×2~3,显存 −40% | 中(内核变更需仔细 benchmark) |
| **阶段 4** | 2~3 周 | 模块化 + moda.py 消债 | 中(大改,依赖测试) |
| **阶段 5** | 1~3 月 | 50B+ 训练能力 | 高(分布式调试复杂) |

**推荐执行顺序:** 阶段 0 → 阶段 2(并行) → 阶段 1 → 阶段 3 → 阶段 4 → 阶段 5。

测试先行(阶段 2 早于后续所有重构),确保每次重构都有 safety net。
