# BitNet b1.58 量化轨道计划(Q 轨道)

**日期:** 2026-10-24 · **理论依据:** "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" ([arXiv:2402.17764](https://arxiv.org/abs/2402.17764))
**定位:** 将 b1.58 三值权重理论接入 OpenMythos(QAT 路线);与 P0-P5 主线、small LM 轨道并行

---

## 0. 适配性总判断

| b1.58 理论 | 对 OpenMythos 的可用性 |
|---|---|
| BitLinear(absmean γ + 三值化) | ✅ 直接可用,作为 `nn.Linear` drop-in |
| QAT + STE | ✅ 本地双阶梯:先 v6.1 微调(快),再从头训练(论文正统) |
| 激活 int8(W1.58A8) | ⚠️ 二阶引入;先做 W1.58A16(仅权重) |
| 推理加速(加法代替乘法) | ❌ PyTorch 仿真拿不到(需专用 kernel);**本地可兑现的是显存/体积 ~16x**,预期写死在验收 |
| "匹配 FP16 基线" | ⚠️ 论文 parity 从 3B 起;15.8M 预期有差距,小模型轨道定位为方法论验证 |

**规模账:** 15.8M 小模型 embedding+head 占 ~80% 且按惯例保持 FP → 只量化层参数的整体收益 ~17%;**b1.58 真正的收益在 3B+ preset**(层参数主导)。

## 1. OpenMythos 特有技术决策

1. **LoRA 深度适配器保持 FP(决策修正)**:原设想"合并 `W + Δ_t` 联合量化",但 OpenMythos 的 LoRA 是**输出残差**(`trans_out + lora(trans_out, t)`),不挂在具体 Linear 上,无可合并权重——修正为 LoRA 模块整体保持 FP
2. **ACT halting 新 gate**:量化噪声会移动 halt 分布;验收要求 halt 步分布偏移 ≤ 0.1 步
3. **MoE 分界**:expert FFN(参数大头)量化;router/gate 保持 FP(离散决策敏感)
4. **循环深度红利**:RecurrentBlock 权重被循环复用,量化收益按迭代次数放大

## 2. 分阶段计划

### Q1 · BitLinear 原语 — ✅ 完成(2026-10-24)

- `open_mythos/primitives/quant.py`:`absmean_quantize`(γ 显式返回,**必须落盘**——taiji-neural-core 的致命教训写成 API 契约)、`absmax_act_quant`(分组 absmax,A8 预留)、`BitLinear`(STE 前向,`enabled=False` 为精确线性路径)
- `QuantConfig`(`mode: none|w158` + `group_size`,`__post_init__` 校验),flat kwargs `quant_mode`/`quant_group_size` 兼容路由,顶层导出
- 验收(全部达成):值域 ∈{-1,0,1} ✓;disabled 与 `nn.Linear` 逐位相等 ✓;同上游梯度下 STE 参数梯度与 Linear 一致 ✓;零权重 γ clamp 不产生 NaN ✓;`test_bitlinear_mlp_trains_loss_decreases`(QAT 端到端训练通路)✓
- 测试 264 → **282**(+18);mypy 0 errors / 35 files;ruff 干净;覆盖率 97% 保持(quant.py 91%)

### Q2 · 架构接入 — ✅ 完成(2026-10-24)

- `BitLinear` 改为 **`nn.Linear` 子类**:state-dict 键与参数形状与 Linear 完全一致 → FP checkpoint 可直接 warm-start 进 W1.58 模型(Q3 阶梯 a 的前提);`OpenMythos._init_weights` 的 isinstance 扫描自动覆盖
- `build_linear(cfg, in, out)` 工厂接入:GQA 4 处 / MLA 6 处 / Expert 3 处(gate/up/down,经可选 cfg 参数)/ MoE routed+shared experts
- **保持 FP**:embedding、lm_head(权重绑定)、MoE router、LoRA、LTI、RMSNorm(模块类型映射测试固化)
- checkpoint metadata:`quant_mode`/`quant_group_size` 独立字段(不动 config dict,旧 checkpoint 的 resume 等值校验零破坏);resume/init-from 一致性守卫;`train_small_lm.py` 新增 `--quant` / `--quant-group-size`;`generate_small_lm.config_from_checkpoint` 共享 helper——生成/评估侧按 metadata 重建量化模型
- 验收(全部达成):289 tests(282+7 新增)✓;7 个 variants 预设构造通过 ✓;`quant=none` 工厂返回原生 Linear(数值路径零改动)✓;同 seed FP/W1.58 双胞胎模型 logits 相异(量化确实生效)且 W1.58 前向有限 ✓
- 冒烟:CPU 2 步 w158 训练 → metadata `w158/64` 落盘;按 metadata 重建出 45 个 BitLinear 模块;v6.1 旧 checkpoint 兼容加载(quant=none)

### Q3 · QAT 训练双阶梯 — 阶梯 a 完成,未过 gate(2026-10-24);阶梯 b 按决策规则移交 3B+

**阶梯 a(v6.1 warm-start + W1.58 微调 5k steps,stream 数据配置,单变量隔离):**

| 验收项 | 门槛 | 实测 | 判定 |
|---|---|---|---|
| val loss(stream,同协议) | Δ ≤ +0.10 vs v6.1(1.6227) | **1.8914(Δ +0.27)** | ❌ |
| EOS(coherence-v2) | 60/60 红线 | 58/60 | ❌ |
| weak_prompt_adherence | ≤ 29(基线不回退) | 39 | ❌ |
| missing_ending | ≤ 27 | 30 | ❌ |
| repeated_trigram | ≤ 14 | 15 | ❌ |
| ACT halt 偏移 | ≤ 0.1 步 | 0.0(双方均饱和于末循环,区分度有限) | ✅* |
| 稀疏度实测 | 报告值 | **33.9%**(理论 31-34% 吻合) | ✅ |

报告:`runs/generation_eval_1792933993.md` · checkpoint 归档 `checkpoints/..._q_w158_v8_failed_gate.pt`
量化损伤轨迹:初始 +0.65(纯 PTQ 损伤)→ 5k 步恢复至 +0.27(恢复曲线明显减速平台)。

**结论:** 三值量化的真实代价在 15.8M 上远超可接受门槛。量化层仅 2.98M/15.86M(18.8%,embedding+head 主导且保持 FP),小模型无可让渡的冗余;与 b1.58 论文 parity 从 3B 起成立的结论一致。

**阶梯 b(从头 QAT 20k steps)决策:不在 15.8M 上执行。** 理由:① 阶梯 a 的恢复平台(Δ+0.27)与容量分析均指向从头训练也无法收敛到 Δ≤0.10;② 3 GPU 小时换一个近似确定的负结果,违背 T1/T2 轮"避免在小模型上过度验证"的决策规则;③ Q 轨道的兑现点本来就是 3B+(Stage 5 GPU 到位后从头 QAT 训练)。管线(Q1/Q2)保持就绪状态。

**新增工具:** `examples/measure_act_halt.py`(ACT halt 步测量,支持双 checkpoint 差分做 gate)+ 2 项单测

### Q4 · 存储与推理兑现 — 未开始

- 导出器:量化层 4-trit/byte 打包 + **γ 逐层落盘**;加载按层反量化 → checkpoint 体积 ~16x(量化层)
- benchmark 诚实基线:PyTorch FP16 线性层;声明 kernel 缺失时速度持平、收益为显存/体积;int8/addsub kernel 列为后续(bitnet.cpp/llama.cpp 不支持 OpenMythos 自定义循环架构,不可直接复用)

### Q5 · 3B 前瞻 — 未开始

- Stage 5 GPU 到位后 3B 直接以 QAT 训练(一次训练同时产出小体积模型);3B W1.58 权重 ≈ 0.9GB → 无 GPU 本地部署 3B 成为可能

## 3. 反面清单(taiji-neural-core 教训 → 硬性验收)

1. scale 因子必须存、必须有测试(Q1 API 契约已固化)
2. benchmark 基线必须含 PyTorch FP16 线性层;不得只对比自写弱基线(Q4)
3. 精度验收用真实任务(TinyStories gates),禁止"合成权重零误差"式同义反复
4. 稀疏度报实测值;报阈值-稀疏度-精度权衡曲线,不报单点

## 4. 风险与回退

| 风险 | 缓解 |
|---|---|
| 15.8M QAT 精度差距超门槛 | 定位为方法论验证;门槛按 Q3 数值,超了如实记录,不降红线改判据 |
| ACT halt 分布漂移 | Q2 接入时加 halt 分布对账测试;超偏移即缩小量化范围(先只量化 FFN) |
| 循环 + 量化数值不稳 | RecurrentBlock 最后接入;先在 prelude/coda 层验证 |
| 训练发散(小模型 QAT 已知风险) | lr 降半起步;复用 v7 事故教训(数据预算显式、val 趋势监控) |
