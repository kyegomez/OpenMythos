# TinyStories v9 · T3b 扩容训练(small 15.8M → mid 37.7M)

**目标:** 迭代计划 T3b([docs/small_lm_iteration_plan.md](../small_lm_iteration_plan.md))
**变量隔离:** 数据配方与 v6.1 谱系完全一致(stream packing / clean text / EOS weight 4.0),
唯一变量是容量——验收 val < 1.55、endings ≥ 40/60 只归因于扩容。
**关键约束:** dim 变化后 `--init-from` 正确拒绝跨维 warm-start → **从头训练**。

---

## 0. Preflight

```bash
cd /Users/macos/Desktop/Claude/OpenMythos
.venv-arm64/bin/python -c "import torch; print(torch.backends.mps.is_available())"   # True
.venv-arm64/bin/python -m pytest tests/test_train_small_lm.py -q   # 全绿
```

## 1. 冒烟(CPU 1 步 + MPS 50 步,已执行 ✓)

- CPU:`--model-size mid --steps 1 ...` → `params=37,675,042`
- MPS 50 步(batch 8 / seq 384):loss 10.9 → 5.3 正常下降,吞吐 **~3,000 tok/s**

## 2. 全量 v9 从头训练(30k steps ≈ 8.5h,过夜)

```bash
nohup env HF_HUB_DISABLE_XET=1 .venv-arm64/bin/python -u -m examples.train_small_lm \
  --dataset roneneldan/TinyStories --dataset-config "" \
  --steps 30000 --batch-size 8 --seq-len 384 --device mps \
  --lr 3e-4 --eos-loss-weight 4.0 \
  --model-size mid \
  --train-tokens 20000000 --eval-tokens 200000 \
  --eval-every 1000 --eval-batches 20 --log-every 100 --save-every 1000 \
  --seed 7 --out checkpoints/tinystories_mps_mid_mythos_seq384_v9.pt \
  > runs/train_v9_mid_scaling.log 2>&1 &
```

- 监控:`tail -f runs/train_v9_mid_scaling.log` / `grep "val loss" runs/train_v9_mid_scaling.log`
- 中断续跑:同命令改 `--resume checkpoints/tinystories_mps_mid_mythos_seq384_v9.pt`(去掉隐式 from-scratch 语义)
- 到 30k 后如需继续冲验收:同命令 `--resume` + `--steps` 加大增量
- 如 MPS OOM:`--batch-size 4`(吞吐相应减半)
- **数据预算红线(T3b 轮教训):20M tokens ≈ 9 万窗口,30k 步 × batch 8 ≈ 1.6 epochs,安全区;不得下调 `--train-tokens`**

## 3. 验收(对照迭代计划 T3b)

| 指标 | v6.1(small)基线 | v9(mid)目标 |
|---|---|---|
| val loss | 1.6227 | **< 1.55** |
| story endings | 33/60 | **≥ 40/60** |
| EOS | 60/60 | **60/60 红线** |
| weak_prompt_adherence | 29 | 不回退 |

```bash
.venv-arm64/bin/python -m examples.evaluate_small_lm_generations \
  --checkpoint checkpoints/tinystories_mps_mid_mythos_seq384_v9.pt \
  --device mps --prompt-file eval_prompts/tinystories_natural_stop_20.txt \
  --preset-set coherence-v2 --max-new-tokens 256 --stop-at-eos \
  --require-eos 60/60 --fail-on weak_prompt_adherence=29 \
  --fail-on missing_ending=27 --fail-on repeated_trigram=14 --out-dir runs
```

- 注:coherence-v2 preset 参数是按 small 模型调的;mid 首评可另跑一版 `--max-new-tokens 376`(全 seq384 预算)对照
- 趋势对比:`examples.compare_small_lm_eval_reports --glob 'runs/generation_eval_*.md' --latest 3`

## 4. 里程碑参考(small 谱系,供中途判读)

| steps | small val | mid 预期 |
|---|---|---|
| 5k | 2.56 | 显著更低(容量×4) |
| 20k | 2.16 | ~1.9 或更低 |
| 30k | 1.91 | 待实测 |
| 105k | 1.62 | 验收目标 < 1.55 需续跑至更多步 |

## 5. 归档 checklist

- [x] checkpoint 复制 `checkpoints/openmythos_tinystories_mid_38m_step60000_seq384_v9.pt` + SHA256(`43a88b98…`)
- [x] 模型卡 `docs/model_cards/openmythos_tinystories_mid_38m_step60000_seq384_v9.md`
- [x] AGENTS.md 执行日志 + 滚动快照更新

## 6. 执行结果(2026-10-24 · 60k 完成 + mid-v1 preset)

| 阶段 | val | EOS(256/376) | endings | repeated | 备注 |
|---|---|---|---|---|---|
| 30k | 1.5515 | 58/60 → 60/60 | 25-26/60 | 17-18 | EOS 有预算长尾 |
| 45k | 1.5142 ✓ | **60/60 双预算** | 30/60 | 14 | 生成素养追平期 |
| 60k(coherence-v2) | 1.5130 ✓ | 58-59/60 ❌ | 30-31/60(平台) | **22-23 ❌** | ~9.2 epochs,记忆化前兆 |
| 60k(**mid-v1**) | 同 | **60/60 严格-256** | **35/60** | 17 | 推荐配置 |

**判决:** val ✓ · EOS 红线 ✓(需 mid-v1 preset)· endings 35/60 vs ≥40 ❌。45k→60k 生成指标平台化且
repeated 恶化 → **停止堆步数**;endings/weak 的下一杠杆在数据侧(20M 预算 9.2 epochs 是结构性原因)。

**教训(已入日志):** `--resume` 续跑前必须先归档当前 checkpoint——45k 的最佳生成权重被 60k 覆盖且不可恢复。

## 7. T5 · 新鲜数据续跑(2026-10-24 启动)

T3b 终判后 weak(35 vs 29)/repeated(17 vs 14)的结构性根因是 20M 预算 × 9.2 epochs。
T5 以 **60M 预算(3 倍)resume 续跑至 75k**:数据集按流序确定性收集,旧 52k 窗口之外新增
~104k 新鲜窗口(67% 未见过),15k 步 ≈ 0.77 epochs,远离记忆化区间。

```bash
nohup env HF_HUB_DISABLE_XET=1 .venv-arm64/bin/python -u -m examples.train_small_lm \
  --dataset roneneldan/TinyStories --dataset-config "" \
  --resume checkpoints/tinystories_mps_mid_mythos_seq384_v9.pt \
  --steps 15000 --batch-size 8 --seq-len 384 --device mps \
  --lr 3e-4 --eos-loss-weight 4.0 --model-size mid \
  --train-tokens 60000000 --eval-tokens 200000 \
  --eval-every 1000 --eval-batches 20 --log-every 100 --save-every 1000 \
  --seed 7 --out checkpoints/tinystories_mps_mid_mythos_seq384_v9.pt \
  > runs/train_v9_t5_freshdata.log 2>&1 &
```

- **续跑前归档检查**:60k 已归档(`openmythos_tinystories_mid_38m_step60000_seq384_v9.pt`)✓
- 验收(mid-v1 preset,256 严格):endings ≥ 40/60 · weak ≤ 29 · repeated ≤ 14 · EOS 60/60 · val 回退 ≤ 0.01
- 判读:endings 35→38+ 则继续加数据量;平台化则 T5 判负,结论收敛为"TinyStories 语料质量上限"

### T5 结果(2026-10-24 · 75k 完成)

| 指标 | 60k | **75k(T5,60M 新鲜数据)** | 验收线 | 判定 |
|---|---|---|---|---|
| val | 1.5130 | **1.4569** | <1.55 | ✅ 破平台 −0.056 |
| EOS(256 严格) | 60/60 | **60/60** | 红线 | ✅ |
| repeated_trigram | 17 | **6** | ≤14 | ✅ 基线的 43% |
| missing_ending | 25 | **24** | ≤27 | ✅ 超基线 |
| story endings | 35/60 | **36/60** | ≥40 | ❌ 斜率 +1/15k |
| weak_prompt_adherence | 35 | **35** | ≤29 | ❌ 两轮未动 |

**终判:** T5 证实"新鲜数据重开改进通道"(val −0.056、repeated 减半、故事更紧凑 75.8 词),
但 weak 顽固不动 + endings 斜率衰减表明**继续堆训练低性价比**。weak 差距的合理解释是启发式判据
对高多样性写作系统性偏严,剩余杠杆:语料规模/多样性、weak 判据人工校准——均为新会话级工作。
归档:`openmythos_tinystories_mid_38m_step75000_seq384_v9_t5.pt`(SHA256 `24033aa0…`)+ 模型卡。

## 8. weak 判据人工校准(2026-10-24)

**审计:** 60 样本(v9-T5 mid-v1 256 严格)逐条人读 vs 判据标记 — **17/35(49%)为误报**:
自然续写以代词(She/It)展开 prompt 场景,被判据的"逐字关键词重复率 <40%"误伤;同时
18 条真阳性确认模型确有"Lily 劫持"失败模式(prompt 主角被换成高频训练名)。

**校准(TDD):** `weak_prompt_adherence` 触发条件追加回指检查 — 续写开头两句内出现
prompt 关键词回指或句首主语代词即视为"已锚定",不再标记;漂浮式跑题(无任何回指)仍报警。
劫持类失败由 `character_drift` 继续负责(具名 prompt)。+2 测试,1 处期望按新语义更新。

**校准后 apples-to-apples(同样本仅换判据,确定性复现):**

| 校准后 | v6.1 | v9-T5 75k |
|---|---|---|
| weak | 3 | 5(噪声区间内,3 vs 5/60 不显著) |
| character_drift | 8 | **11 ← 真实差距所在** |

**结论:** 旧 weak 35 是判据噪声为主; adherence 校准后两代模型基本平手。真正该投入的是
**drift(劫持)**——数据侧杠杆 = TinyStories 语料 Lily 频率(名字平衡化/重加权)。
后续验收线应使用校准定义并在同定义下对比基线。

## 9. T6 · 名字平衡化(2026-10-24 启动)

**根因量化:** 50k 故事抽样,`Lily` 出现于 **150%** 的故事(第二名 Tom 53%,3 倍断层);
"卡壳就喊 Lily"是语料分布的直接镜像(校准后 drift 11 vs 基线 8)。

**方案:** 替换增强 — `--balance-names`:把高频名(Lily/Tom/Ben/Timmy)按确定性哈希
(text+seed)轮换到 10 名平衡池(Nora/Ella/Ruby/Sofia/Maya/Leo/Owen/Jack/Ivy/Zoe),
逐故事一致、逐名不碰撞;纯文本层变换,与 packing/quant 正交,eval 侧保持规范语料。

**执行(教训已应用:输出到新文件,不覆盖 T5):**

```bash
nohup env HF_HUB_DISABLE_XET=1 .venv-arm64/bin/python -u -m examples.train_small_lm \
  --dataset roneneldan/TinyStories --dataset-config "" \
  --resume checkpoints/tinystories_mps_mid_mythos_seq384_v9.pt \
  --steps 15000 --batch-size 8 --seq-len 384 --device mps \
  --lr 3e-4 --eos-loss-weight 4.0 --model-size mid --balance-names \
  --train-tokens 60000000 --eval-tokens 200000 \
  --eval-every 1000 --eval-batches 20 --log-every 100 --save-every 1000 \
  --seed 7 --out checkpoints/tinystories_mps_mid_mythos_seq384_v9_t6.pt \
  > runs/train_v9_t6_namebalance.log 2>&1 &
```

- 验收(校准判据,mid-v1 preset 256 严格):**drift ≤ 8** 且 val/EOS/endings/repeated/missing 不回退
- 判读:drift 达标 → T6 成立,"名字平衡化"进正式数据配方;未达 → 劫持非纯频率问题,记录后收束 small LM 轨道
- metadata:`balance_names` 字段 + resume/init-from 守卫(同 packing 模式)

**运行状态(2026-10-27 01:00):** PID 60242(00:09 启动,MPS)运行中,step 78.2k/90k,
tok/s ~3.9k,val 1.6387@76k / 1.6563@77k / 1.6497@78k。相对 T5 终值 1.4569 抬升约 0.19,
预期为名字平衡化分布切换 vs 规范评估语料的瞬态(评估仍用原始 Lily 偏置语料),
观察 80k 后是否回落;checkpoint 已随 save-every=1000 覆盖写至 t6 文件,T5 最佳不受影响。
预计 ~04:20 完成,完成后跑 60 样本 eval(mid-v1 preset 256 严格 + 校准判据)出终判。

### T6 结果(2026-10-27 · 90k 完成,未过 gate,轨道收束)

| 指标 | T5 75k(交付基线) | T6 90k(名字平衡化) | 验收线 | 判定 |
|---|---|---|---|---|
| drift | 11 | **9** | ≤ 8 | ❌ 差 1(改善但不足) |
| endings | **36/60** | 29-30/60 | 不回退 | ❌ −6 |
| missing | **24** | 30-31 | 不回退 | ❌ |
| repeated | **6** | 9 | 不回退 | ✅ 仍优于基线 14 |
| EOS(256/376) | 60/60 | 59/60 / 60/60 | 红线 | ❌ 边缘长尾 |
| val(canonical) | 1.4569 | 1.6738 | 参考 | 分布迁移成本 |

**终判:交换不划算。** drift 11→9 证实劫持部分源于名字频率,但硬替换到稀有名(Ivy/Zoe/Maya
等)拆掉了故事模板"名字槽"的统计连续性,endings/missing 各付 −6/+6。**按预登记规则收束
small LM 轨道:T5(75k)为最终交付基线。** checkpoint 归档为
`openmythos_tinystories_mid_38m_step90000_seq384_v9_t6_failed_gate.pt`(对照保留)。

若未来重启:候选修正 = 平衡池扩到高频名带(Tom/Ben/Timmy 级)而非稀有名,或平衡/规范 50/50
混合以保留模板统计——均为新会话工作,当前不再投入训练时间。
