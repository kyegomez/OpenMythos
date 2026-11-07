# 商用优化路线(基于全部实验证据)

**日期:** 2026-10-28 · **性质:** 路线级规划——每一段都有准入门槛与 kill criteria,禁止跳段投入

---

## 0. 证据底座(路线只建在被证明的东西上)

| 实验证据 | 对商用的含义 |
|---|---|
| 深度实验:训练期循环深度 = 真实串行算力(add 1→4 循环 1.7%→98.7%;parity 4 循环不可学、16 循环可学) | 架构"可训练深度收益"**成立** |
| 深度实验:推理期外推不存在(三个模型全部锁定训练深度) | "同一模型按请求调算力"卖点**不成立**(首要嫌疑 loop-index 钳制,未验证) |
| 量化:15.8M Δ+0.27 | 三值量化兑现点在 3B+(GPU),管线就绪 |
| small LM:T1-T6 全程,质量天花板 TinyStories 级 | 通用文本生成产品化**死路**(免费基线碾压 100 倍) |
| 工程资产:310 tests / 97% cov / gate 工具链 / MPS 训练方法论 | **确定性的资产**,任何方向都复用 |

## 1. 三段门控路线

### Phase 0 · 决定性小实验(本周 · 本地 · 零现金)

**Gate 0a — 深度外推实验(已完成 2026-10-27):判负**
- 四模型终局:add@4 自深度 98.7% 但外推持平;parity 连 16 循环训练都学不会(½ln2 平台);外推零收益且 add@16 伤优化(93.7% vs 98.7%)
- **"推理可调算力"叙事永久关闭**;详见 docs/depth_extrapolation_experiment.md §6

**Gate 0b — 垂直领域选择:✅ 通过(2026-10-27,用户资产盘点)**
- 用户已有:FactorMiner 因子挖掘系统(116 因子/5 条 IC 筛选/V1→V3 迭代记录,最强 IC 0.3556,BTC_USDT 1h)+ freqtrade 实盘机器人 + 量化领域知识库(51 笔记/31k 词,Obsidian ClaudeBrain)
- **护城河 = 已验证的因子研究基础设施 + 实盘执行渠道**,非公开 K 线(K 线预测本身维持排除)
- **Phase 1 产品形态聚焦:FactorMiner 研究自动化**(LLM 因子假设生成 → 候选代码 → 经用户 IC gate 自动验证 → 自动归档知识库)+ 知识库 RAG 助手 + 交易事件守卫
- 体量事实:知识库 31k 词 = RAG 规模,非训练语料;不做从零训练,文本理解任务用微调预训练小模型

### Phase 1 · FactorMiner 研究自动化 MVP(2-4 周 · 本地 · 零现金,Gate 0b 已通过)

- **M1 知识库 RAG 助手 — ✅ 完成(2026-10-28)**:`~/Desktop/Claude/QuantKB`(独立仓库,246 行,纯标准库)
  — BM25 + CJK bigram 分词,双语料 1360 passages,三组中英查询实测精准命中(IC 筛选/freqtrade/go-live);
  `python -m quantkb.kb "query" -k N` 即用。M2 接入要点:FactorMiner 自带 `mcp_server/mining_server.py`
  (FactorMiningAPI),研究自动化环走 MCP 即可,无需新管道
- **M2 研究自动化环 — ✅ MVP 完成(2026-10-28)**:`QuantKB/quantkb/research_loop.py` — LLM 假设因子 →
  IC gate(MVP 版 V3 近似)→ vault 格式自动归档;真实 BTC_USDT 1h(736 bars)首跑 2/3 过 gate
  (range_position IC −0.13 / momentum_divergence IC +0.12 / volume_thrust 被诚实淘汰);
  M2 收尾(2026-10-28):**完整 V3 五条筛选 + walk-forward 样本外验证已接入**——range_position 在
  三个时间分段 IC 全部同号(−0.317/−0.174/−0.255,样本外稳定),effectiveness 0.791 / win_rate 52%
  全过 V3 条件;与 V3 人工榜(top12 波动率族)互补、无重复。待办:FactorMiner 重构稳定后接
  `mcp_server/mining_server.py` 对齐其精确打分,并入 freqtrade 信号层回测
- **M3 事件守卫 — ✅ MVP 完成(2026-10-28)**:`QuantKB/quantkb/event_guard.py` — 词典 v0
  (PAUSE/REDUCE 两档 18 条规则)+ 真实数据冲击校准(BTC 24h |ret| p95 = **2.88%**,36/712 窗口)+
  freqtrade / hyperliquid-hft 双接入示例;升级路径:新闻采集(CryptoPanic token 待用户开通)→
  微调小模型替换词典,`assess()` 合同不变
- 验收:M2 产出的因子必须过用户现有 IC 筛选(≥60% 一致性等 5 条),且研究迭代周期(假设→验证)实测缩短;LLM 生成因子与人工因子同 gate 同标准
- **M4 因子→PnL 回测 — ✅ 完成(2026-10-28,诚实负读数)**:向量化回测(防未来函数/含费/
  29 tests)机制全过;但 30 天窗口 naive 信号→仓位不产生超额(**费率主导**:550-880bp 换手成本
  vs IC≈0.13 边际)。结论:IC≠PnL,下一杠杆在降换手(hysteresis)与更长历史;freqtrade 策略
  工件已产出(`QuantKB/freqtrade_strategies/`)供用户环境完整回测
- **M4b 降换手 — ⚠️ 结果撤回**:hysteresis 分支存在未来函数(位置未滞后),修正后 M4b 的
  "转正"消失(30 天窗 −0.47%/+0.63%);回归测试已固化(`test_hysteresis_positions_lag_factor_no_lookahead`)
- **M4c 4 年全假设类 OOS 终判 — ✅ 完成(2026-10-28):全面证伪**。36,177 bars(4.1 年)上
  五个策略(方向动量/均值回归/波动率族 ×3)、IS 参数冻结:**OOS 全部跑输 buy&hold**
  (−15% 至 −50% vs −8.02%),零样本外 alpha。LLM 因子生成对本垂直无增量;商用价值 =
  自有 bot 策略 + 本套已三次实战证明的防伪基础设施。详见 docs/m4_factor_backtest_design.md §8
- **M4c 4 年样本外验证 — ✅ 完成(2026-10-28):信号证伪**。30m→1h 重采样 36,177 bars(4.1 年零缺口):
  两因子分年 IC 变号、量级塌缩 10 倍(+0.13→+0.01);IS −62%/−9.9%,OOS −31%/−26% 均跑输 B&H。
  30 天正收益 = 区间过拟合假信号;预注册 OOS 在投入真金前杀死它——**方法论按设计目的工作**。
  修正:方向性动量/均值回归假设类关闭,研究环假设类转向波动率族(V3 自身发现 7/12 主导)
- 变现形态:先增强自有交易系统(PnL 即回报),研究自动化成熟后可产品化为量化研究工具

### Phase 2 · 架构规模化验证 — 已关闭(Gate 0a 判负)

- 深度外推实验负结果移除了规模化验证的前提;除非未来出现新的架构证据(索引编码修复 + 更深循环块的论文结果),不重启

### Phase 3 · 资产化(零成本并行)

- 工具链与负结果文档开源维护;换取声誉/机会流,不做现金流预期

## 2. 不做清单(证据禁止项)

- ❌ 继续投入 TinyStories 质量迭代(天花板已证,T6 负交换)
- ❌ 无 GPU 碰 3B+ 量化(15.8M 负结果 + bitnet.cpp 不支持自定义架构)
- ❌ Gate 0a 之前对"推理可调算力"做任何产品承诺
- ❌ 通用 LM 方向的一切投入

## 3. 决策树速览

```
Gate 0a(2h 本地)──翻盘──┐
                          ├──> Phase 2(GPU 赌注,$5k-30k)
Gate 0b(用户回答)─有数据─┴──> Phase 1(垂直 MVP,零现金)──打得过基线──> 变现
        │                        │
        └─无数据                  └─打不过─> Phase 3 资产化,商用预期归零
```
