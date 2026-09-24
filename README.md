# Small Agent Architecture Notes

面向约 1B 参数小型 Agent 模型的公开研究记录。

当前目标聚焦于：

- 工具调用与短程任务规划；
- 简单 Python、Shell 和代码修复；
- 基础数学与工具辅助计算；
- 固定状态预算下的长任务状态保持；
- 从零训练文本专用 GDN＋GQA 混合注意力 Base；
- 小模型上的训练效率、推理延迟和状态缓存；
- 新架构与公开小模型的可复现横向比较。

## 当前方向（2026-09-24）

从随机初始化训练约 1B 的**文本专用混合注意力 Base**：线性层选 **Gated DeltaNet（GDN）**，间隔插入全局 GQA 注意力层。优先研究基础语言、代码、数学与后续 Agent 能力；暂不接入事件记忆、SetEncoder、反事实效用教师或动态规划。3:1 层比例是首个候选而非已验证的最优值；层数、宽度、词表和训练预算仍待参数统计及 L40 短跑后冻结。此方向借鉴 Qwen3.8 同系文本主干，但不复制其 27B 多模态配置或权重。

## 历史研究状态（当前暂停）

此前研究过[效用巩固记忆 v2](docs/utility-consolidation-memory-v2.zh.md)：$S$ 限制独立事件条数，$C$ 限制活动 latent 向量总数，模型预测每条记录的可变表示长度 $r_i$ 并在硬预算内联合分配。它的数学规范保留作历史研究，**不属于当前 GDN Base 的训练结构**。v1 是固定 $r_i=1$ 的历史特例；[整体架构旧草案](docs/architecture-current.zh.md) 记录旧研究背景。以下功能分工和实验结果均指暂停的旧方案：

1. 长期记忆采用不可变事件记录；
2. 每次只有一个候选进入竞争，容量超限时只淘汰一条；
3. 训练期用决策之后的反事实未来损失监督效用预测器；
4. 已实现的 B2 原型用共享字节编码器和集合编码器联合预测各记录效用，再执行单次淘汰；
5. v2 推理期不访问未来，而是预测每条记录的容量效用曲线，并在 $S,C$ 硬约束下执行确定性联合分配；满槽时仍只淘汰一条；
6. 原生主干通过层内记忆注意力读取保留表示；主干规模、具体读取层及数值配置仍待从零训练的受控实验决定。

第一轮符号实验中，Oracle−FIFO 成功率差距在同分布测试为 $0.895\pm0.012$，长度外推测试为 $0.948\pm0.014$；预测器分别恢复 93.6% 和 92.3% 的差距。这个结果只证明结构化符号环境中的可行性和可学习性，不代表自然语言 Reader 已经验证通过。

Phase 0B2 首个 4.97M 参数文本实验在同模板测试恢复 99.1% 的 Oracle−FIFO 差距，但在未见释义和“未见释义 + 长度外推”上只恢复 23.8% 和 31.0%。后续 B2.1 的完整集合交互＋排序监督通过了受控合成任务的多种子确认，但关系链仍退化。

Phase 0B2.1 第一阶段已经在六张 L40 上完成 A–F 六个严格对照。预注册主候选 E（完整集合 + Regression + Ranking + Eviction）虽然把组合成功率提高 5.17 个百分点，但平均淘汰 regret 反而增加 35.87%，且只有 3/6 类任务不下降，因此正式决定为 `stop_and_diagnose`。诊断性消融中 C（完整集合 + Regression + Ranking）最好：组合成功率提高 9.67 个百分点，平均 regret 降低 41.93%。由于 C 是事后选出的候选，项目随后为它单独冻结并执行了多种子确认计划。

C 随后完成了独立冻结的 A/C Seed 0、1、2 确认。C 的三种子组合成功率为 $0.860\pm0.020$，A 为 $0.605\pm0.182$；C 平均提高 25.5 个百分点，并将四测试集平均淘汰 regret 降低 63.8%。但在 B2.4 的高干扰关系链与 B2.7 的编码器接入冻结 Reader 测试中，现有选择器**没有稳定保留关键旧记录**。这否定的是“当前 checkpoint 已可部署”，不是反事实效用定义。

Phase 1 的冻结 Qwen＋末端外挂 Reader 在单记录与多格式、多记录组合之间出现明显失败。该试验**没有**从零训练原生主干，不能作为完整架构的验证或反证。最终方案仍需从零 A/B；预训练来源、tokenizer 和预算尚在准备。v2 第一版不依赖显式年龄，但 B2 实验性编码器曾使用年龄 embedding，不能混称为已冻结设计。

## 历史方案：v2 数学核心（当前暂停）

$$
\mathcal R_t=M_t\cup\lbrace c_t\rbrace
$$

每条记录由事件编码器产生有序潜在向量；$Z_i(r_i)$ 表示前 $r_i$ 个活动向量，$r_i=0$ 表示整条淘汰。训练期以**决策之后**的未来损失构造容量效用曲线 $U_{t,i}(k)$，推理期由当前可见信息预测 $\hat U_{t,i}(k)$。完整教师定义见[v2 数学规范](docs/utility-consolidation-memory-v2.zh.md)。

$$
a_t^*=\arg\max_{a\in\mathcal A_t}
\left[\sum_i\hat U_{t,i}(r_i)-\lambda\sum_i r_i\right],
\qquad \sum_i r_i\le C,
\quad |\lbrace i:r_i\gt0\rbrace|\le S.
$$

其中满槽时约束还要求恰好淘汰一条；未满槽时全部保留。$R_{\max}=1,C=S,\lambda=0$ 时严格退化为 v1 的单条最低预测效用淘汰。动态规划精确优化的是**可加预测代理目标**，不是实际未来损失的全局最优。

## 对比模型

- TinyLlama-1.1B：标准小型 GQA 架构参考；
- OLMo-1B：开放训练过程和中间 checkpoint 参考；
- RWKV-7：固定连续状态和广义 Delta Rule 参考；
- Kimi Linear/KDA：高效线性注意力和连续矩阵状态参考；
- Hymba-1.5B：公开小型混合架构参考；
- Qwen2.5-Coder-1.5B：代码能力参考；
- SmolLM2-1.7B：小模型数据工程、数学和指令能力参考。

## 文档

- [效用巩固记忆 v2：已暂停的历史数学规范](docs/utility-consolidation-memory-v2.zh.md)
- [整体架构旧草案：v2 前的编码器与主干背景](docs/architecture-current.zh.md)
- [架构与横向比较方案](docs/architecture-comparison-plan.md)
- [基础预训练数据规划 v0.2](docs/base-pretraining-data-plan.md)
- [P0：5000 万 token 自动预训练数据管线计划](docs/p0-50m-data-pipeline-plan.md)
- [效用巩固记忆 v1：固定容量历史特例](docs/utility-consolidation-memory-v1.md)
- [Phase 0A/0B：符号验证方案](docs/phase0-ab-validation.md)
- [Phase 0B2：从文本预测记忆效用](docs/phase0-b2-text.md)
- [Phase 0B2.1 第一阶段计划：完整竞争集合联合排序](docs/phase0-b2.1-stage1-plan.md)
- [Phase 0B2.1 数据生成与审计](docs/phase0-b2.1-data.md)
- [Phase 0B2.1 第一阶段执行说明](docs/phase0-b2.1-execution.md)
- [Phase 0B2.1 第一阶段实验结果](results/phase0_b21_stage1/RESULTS.md)
- [Phase 0B2.1 A/C 多种子确认计划](docs/phase0-b2.1-ac-seed-plan.md)
- [Phase 0B2.1 A/C 多种子确认结果](results/phase0_b21_ac_seeds/RESULTS.md)
- [实验可视化：TensorBoard 与记忆淘汰查看器](docs/visualization.md)
- [后续任务清单](TODO.md)
- [早期持久情景记忆推导（已被简化版取代）](docs/candidate-persistent-episodic-memory.md)
- [决策记录](docs/decisions.md)

## 公式书写约定

- 行内公式用 `$...$`，独立公式用单独的 `$$` 行包围；`\[...\]` 和 `\(...\)` 在 GitHub 上不渲染；
- 独立公式内不要出现只有 `=` 或 `-` 的行，否则会被 Markdown 当成标题解析；
- 公式内的大小比较写 `\lt`、`\gt`，避免 `<`、`>` 被转义；
- 公式里的花括号写 `\lbrace`、`\rbrace`，因为 `\{`、`\}` 的反斜杠会被 Markdown 吃掉；
- 算子名写 `\mathrm{...}`，不要用 `\operatorname{...}`：GitHub 的宏白名单不含它，页面会显示红框 `The following macros are not allowed: operatorname`。

## Phase 0A/0B 复现

该实验不训练语言模型，用可审计的符号 Reader 分离验证“选择性保留是否存在收益”和“效用是否能从决策时信息预测”：

```bash
python3 -m pip install -r requirements-phase0.txt
python3 -m unittest discover -s tests -v
python3 experiments/phase0_ab.py --output results/phase0_ab
```

`Future Oracle` 只作为上界和标签生成器；`Predicted Utility` 的特征不包含未来记录或最终答案。

已记录的首次运行结果见 [`results/phase0_ab/RESULTS.md`](results/phase0_ab/RESULTS.md)。

纯文本 B2 实验：

```bash
python3 experiments/phase0_b2_text.py --output results/phase0_b2_text_seed0
```

首个 Seed 0 结果见 [`results/phase0_b2_text_seed0/RESULTS.md`](results/phase0_b2_text_seed0/RESULTS.md)。

## 原则

- 最终不重复训练完整传统 1B 基线；
- 公开模型用于最终横向比较；
- 架构归因从无语言模型、20M–50M、100M–150M 到 300M–400M 分阶段完成；
- 不把不同 Tokenizer 的 perplexity 直接横向比较；
- 同时报告质量、Agent 成功率、记忆选择 regret、代码/数学能力和效率；
- 状态预算必须同时计算局部 KV、短期缓冲、长期记录和元数据；
- Future Oracle 不超过同预算 FIFO 时停止该方向；
- 架构结论必须经过参数、数据和训练 token 匹配的实验验证。

## 仓库结构

中文文档是当前架构讨论的主文档。既有英文和 LaTeX 文件保留作历史研究稿，**目前不保证与最新中文架构总览同步**，不可单独据此认定最终设计：

```text
README.md                 中文原文
README.en.md              历史英文原稿
docs/*.md                 中文原文分册
docs/en/*.md              历史英文原稿分册
paper/main.tex            历史 LaTeX 原稿（英文，tectonic 编译）
paper/preamble.tex        LaTeX 导言区与宏定义
paper/sections/*.tex      LaTeX 原稿分节
paper/check-section.sh    单独编译某一节，便于早期发现错误
```

构建 PDF：

```bash
cd paper
tectonic -X compile main.tex
```

核心架构文档在三份文本中同步维护；当前实验运行记录只写中文，结论冻结后再决定是否进入英文与 LaTeX 手稿。Markdown 使用 GitHub 能渲染的写法（`$$`、`\mathrm`、`\lbrace`），LaTeX 原稿使用规范 LaTeX（`\[...\]`、`\operatorname`、`\{...\}`）。

## License

Apache-2.0
