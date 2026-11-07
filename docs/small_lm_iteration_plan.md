# Small LM 训练成果盘点与迭代计划

**日期:** 2026-10-24 · **基线:** v6.1(step 105000, seq384, clean text)
**范围:** TinyStories 15M 本机(MPS)训练轨道 — 与核心库阶段 5(GPU)无关

---

## 1. 当前训练成果

### 1.1 Loss 轨迹(105k steps,15.8M params,dim=256)

| 里程碑 | step | seq | val loss | 关键变化 |
|---|---|---|---|---|
| 早期 | 5k | 128 | 2.5594 | TinyStories 起步 |
| — | 20k | 128 | 2.1564 | — |
| — | 30k | 128 | 1.9065 | — |
| v0 | 40k | 128 | 1.8761 | 首个归档基线 |
| v1 | 60k | 128 | 1.8518 | 收益收窄 |
| v2 | 70k | 128 | 1.8282 | EOS 追加数据 |
| v3 | 80k | 128 | 1.8320 | weighted EOS(4.0) |
| v4 | 85k | 128 | 1.8189 | 低 loss 对照 |
| v5 | 90k | 256 | 1.7513 | seq256 warm-start,自然停止 60/60 |
| v6 | 100k | 256 | 1.6520 | clean text |
| **v6.1** | **105k** | **384** | **1.6227** | seq384,当前最佳 |

**结论:** 105k 步内 val loss 2.56 → 1.62;v5 之后每 5-10k 步收益 0.05-0.10,**仍在下降但边际递减**。自然停止问题已在 v5 解决并保持至今。

### 1.2 生成质量(v6.1 + coherence-v2,60 样本:20 prompts × 3 presets)

| 指标 | 数值 | 判定 |
|---|---|---|
| 自然停止(EOS) | **60/60** | ✅ 硬门槛保持 |
| 平均词数 / 句数 | 85.9 / 8.8 | — |
| prompt overlap | 0.44 | ⚠️ 中 |
| story endings | 33/60(55%) | ⚠️ 主瓶颈 |
| character drift | 8/60(13%) | ⚠️ |
| repeated trigram | 14/60(23%) | ⚠️ |
| too_short / unclosed_quote | 4 / 1 | ✅ 可接受 |
| budget_stop / long_generation | 0 / 0 | ✅ |

preset 对比已收敛:`coherence-v2` 为自然 EOS-bounded 评估推荐,`compact-v61` 为短 demo 推荐。

### 1.3 能力边界(现状代码)

- 模型尺寸**硬编码**(dim=256 / n_heads=8 / n_kv_heads=2 / expert_dim=256 / n_loops=4)——扩容需加 CLI
- 训练数据为**连续 token 流打包**(story 边界只有 EOS 标记,窗口不与故事对齐)
- 评估工具链齐全:quality/coherence 启发式、preset 集、report 对比(CSV/markdown)、context resize、runbook

---

## 2. 瓶颈分析(按 flag 数排序)

| 排序 | 瓶颈 | 证据 | 归因假设 |
|---|---|---|---|
| 1 | prompt adherence | weak 29/60(48%) | 训练分布无"给定开头→续写"结构;模型学到的是整篇故事流,不是条件续写 |
| 2 | 结尾完整度 | missing_ending 27/60 | 故事平均长于 384 token,训练窗口很少同时看到 开头→结尾 的完整弧;结尾信号在窗口内欠采样 |
| 3 | 重复 | repeated_trigram 14/60 | TinyStories 模板化语料 + 15M 容量下倾向复读安全片段 |
| 4 | 角色漂移 | drift 8/60 | 容量 + 长生成(85 词)下名字/指代保持能力有限 |

**核心判断:** 停止问题已终结,当前是**条件对齐与故事结构**问题;纯加步数收益递减,应转向训练数据组织与容量。

---

## 3. 迭代路线

> **实施状态(2026-10-24 更新):** T1/T2/T4 代码交付(18 项回归测试全绿)。首轮训练两轮均**未过验收 gate**:
> v7(pair0.5+dedup)与 v7.1(纯 story 对齐)在 EOS 红线与重复指标上回退,adherence/ending 仅小幅改善
> (29→22/25、27→29/25)。详细数据与教训见
> [runbooks/tinystories_v7_story_aligned.md §5.1](runbooks/tinystories_v7_story_aligned.md)。
> **决策规则触发后 T3b 已启动:v9(mid 37.7M)从头长跑运行中**;Q 量化轨道同期定案 15.8M 不达标
> (见 [bitnet_quantization_plan.md](bitnet_quantization_plan.md)),两条轨道共同指向容量是主约束。

### T1 — 续写对齐训练(prompt adherence + endings)— 优先

**假设:** 把训练样本从"连续流"改为"故事对齐 + 续写对"混合,可同时改善 prompt adherence 与结尾完整度。

**改动:**
- `examples/train_small_lm.py` 新增故事对齐打包模式:每个训练窗口以故事边界开头(整篇故事放入窗口,超出部分截断到 EOS),消除跨故事半截窗口
- 新增续写对采样:取故事前 1-2 句为"prompt 段",其余为续写目标,与完整故事混合(混合比可调,默认 50/50)
- TDD:`tests/test_train_small_lm.py` 增加对齐打包与续写对切分回归(先红后绿)

**验收(eval 60 样本,同 prompt 集 / 同 seeds / coherence-v2 preset):**
- weak_prompt_adherence ≤ 15/60(当前 29)
- missing_ending ≤ 15/60(当前 27)
- EOS 60/60 不回退(硬门槛)
- val loss 回退 ≤ 0.02(数据分布变化容差;超出则调低混合比)

**成本:** 代码约 1 天 + 5k-10k steps 续训(约 1-2 小时 MPS @ 5k tok/s)

### T2 — 重复抑制(数据侧)— 次优先

**改动:** 训练流做 8-gram 级去重(模板句去重),可选 token-level repetition-aware loss 权重。
**验收:** repeated_trigram ≤ 5/60(当前 14),其余指标不回退。
**成本:** 数据预处理 0.5 天 + 续训;可与 T1 共用一次训练。

### T3 — 容量 / 上下文扩展 — 视 T1/T2 结果决定

- **T3a seq512 warm-start:** 工具已具备(`examples/resize_small_lm_context.py`),目标让完整故事弧进窗,直接攻击 endings。验收:val loss < 1.60,endings ≥ 40/60
- **T3b 模型扩容 15.8M → ~35M:** `train_small_lm.py` 增加模型尺寸预设/CLI(dim 256→384)。验收:val loss < 1.55,endings ≥ 40/60。成本:训练时间 ×2-3
  > **实施状态(2026-10-27 · 轨道收束):** v9-T5(75k)为**最终交付基线**:val 1.4569 · EOS 60/60 严格-256 ·
  > repeated 6 · missing 24 · endings 36/60。weak 判据经 60 样本人审校准(49% 误报,代词续写),
  > 校准后与基线平手。T6 名字平衡化(90k)实测:drift 11→9(未达 ≤8)但 endings/missing 各付
  > −6/+6,**交换不划算,轨道按预登记规则收束**。全过程数据与教训见
  > [runbooks/tinystories_v9_mid_scaling.md §6-9](runbooks/tinystories_v9_mid_scaling.md)。
  >
  > **T6 进行中(2026-10-27 01:00 快照):** 名字平衡化(替换增强 `--balance-names`,Lily 150% →
  > 10 名平衡池)续训 75k→90k,输出到 `tinystories_mps_mid_mythos_seq384_v9_t6.pt`(不覆盖 T5)。
  > 当前 step 78.2k,val 1.6497@78k(76k 1.6387 / 77k 1.6563)——相对 T5 的 1.4569 有抬升,
  > 预期为训练分布切换(平衡名)vs 规范评估语料的瞬态,观察 80k 后是否回落;tok/s ~3.9k,
  > 预计 ~04:20 跑完。验收(校准判据):**drift ≤ 8** 且 val/EOS/endings/repeated/missing 不回退;
  > 达标则名字平衡化进正式数据配方,未达则收束 small LM 轨道。

### T4 — 评估验收工具化(与 T1 训练并行)

- `evaluate_small_lm_generations.py` 增加 gate 参数:`--require-eos 60/60`、`--fail-on repeated_trigram>5` 类断言,一条命令输出 pass/fail
- 用 compare CLI 的 `--csv` 建立迭代趋势表(每轮归档一行)
- 可选:LLM judge 作为 prefilter(启发式指标保留为快速 gate,不替代)

---

## 4. 建议执行顺序

```
T1(数据组织 + 续训)──┬──> T4(评估 gate,训练期间并行)
                      │
                      └──> T2(数据去重,可与 T1 同一次续训合并验证)
                                │
                                └──> T3a(seq512)──> T3b(扩容,按需)
```

每轮迭代必须满足的**回归红线:**
1. EOS 60/60 不回退(自然停止是 v5 以来的硬成果)
2. val loss 回退 ≤ 0.02(除非该轮目标就是分布切换)
3. 同 seeds 可复现(`--seed` 固定,报告含 preset set)

## 5. 风险与回退

| 风险 | 缓解 |
|---|---|
| 续写对混合伤 val loss | 混合比可调;超容差即回退纯对齐打包 |
| 15.8M 容量是天花板,T1/T2 收益有限 | 直接跳 T3b,避免在小模型上过度调数据 |
| 语料去重降低多样性 | 去重阈值保守(只删完全重复模板),监控 val loss |
| 迭代周期内 checkpoint 管理混乱 | 沿用归档命名 + 模型卡惯例,每轮新文件不覆盖 |
