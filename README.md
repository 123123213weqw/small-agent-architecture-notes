# Small Agent Architecture Notes

面向约 1B 参数小型 Agent 模型的公开研究记录。

当前目标聚焦于：

- 工具调用与短程任务规划；
- 简单 Python、Shell 和代码修复；
- 基础数学与工具辅助计算；
- 固定状态预算下的长任务状态保持；
- 通过决策后反事实未来损失学习记忆淘汰策略；
- 小模型上的训练效率、推理延迟和状态缓存；
- 新架构与公开小模型的可复现横向比较。

## 当前状态

项目处于符号原型验证阶段。已经冻结效用巩固记忆 v1 的核心公式，并完成 Phase 0A/0B、纯文本 B2 和 B2.1 第一阶段，但尚未锁定最终 1B 主干。当前状态：

1. 长期记忆采用不可变事件记录；
2. 每次只有一个候选进入竞争，容量超限时只淘汰一条；
3. 训练期用决策之后的反事实未来损失监督效用预测器；
4. 推理期不访问未来，只执行效用预测、单次淘汰和标准 Cross-Attention 读取；
5. Transformer、线性注意力或混合主干仍由小规模实验决定。

第一轮符号实验中，Oracle−FIFO 成功率差距在同分布测试为 $0.895\pm0.012$，长度外推测试为 $0.948\pm0.014$；预测器分别恢复 93.6% 和 92.3% 的差距。这个结果只证明结构化符号环境中的可行性和可学习性，不代表自然语言 Reader 已经验证通过。

Phase 0B2 已开始验证纯文本效用预测。首个 4.97M 参数实验在同模板测试恢复 99.1% 的 Oracle−FIFO 差距，但在未见释义和“未见释义 + 长度外推”上只恢复 23.8% 和 31.0%，因此 B2 当前未通过，暂不进入 20M–50M Reader。

Phase 0B2.1 第一阶段已经在六张 L40 上完成 A–F 六个严格对照。预注册主候选 E（完整集合 + Regression + Ranking + Eviction）虽然把组合成功率提高 5.17 个百分点，但平均淘汰 regret 反而增加 35.87%，且只有 3/6 类任务不下降，因此正式决定为 `stop_and_diagnose`。诊断性消融中 C（完整集合 + Regression + Ranking）最好：组合成功率提高 9.67 个百分点，平均 regret 降低 41.93%。由于 C 是事后选出的候选，项目随后为它单独冻结并执行了多种子确认计划。

C 随后完成了独立冻结的 A/C Seed 0、1、2 确认。C 的三种子组合成功率为 $0.860\pm0.020$，A 为 $0.605\pm0.182$；C 平均提高 25.5 个百分点，并将四测试集平均淘汰 regret 降低 63.8%。全部预注册门槛均通过，因此 B2.1 在当前合成任务范围内确认“完整集合交互 + 排序监督”有效。`relation_chain` 仍明显退化，且该结果不代表语言表示或真实 Reader 已通过。

第一版明确不加入显式衰减、年龄、occupancy、新奇度、冲突门、关联图、soft write 或记忆合并。

## 冻结的核心

$$
\mathcal R_t=M_t\cup\lbrace c_t\rbrace
$$

$$
u_{t,i} =
\mathcal L_{\bar\theta,F}
\left(Y_t\mid B_t,\mathrm{mask}_i(\mathcal R_t)\right) -
\mathcal L_{\bar\theta,F}
\left(Y_t\mid B_t,\mathcal R_t\right)
$$

$$
\hat u_{t,i}=f_\phi(m_i,g_t,B_t,\mathcal R_t)
$$

$$
M_{t+1} =
\mathcal R_t\setminus
\left\lbrace \arg\min_i\hat u_{t,i}\right\rbrace
$$

最后一式只在 $|\mathcal R_t|\gt S$ 时执行。

## 对比模型

- TinyLlama-1.1B：标准小型 GQA 架构参考；
- OLMo-1B：开放训练过程和中间 checkpoint 参考；
- RWKV-7：固定连续状态和广义 Delta Rule 参考；
- Kimi Linear/KDA：高效线性注意力和连续矩阵状态参考；
- Hymba-1.5B：公开小型混合架构参考；
- Qwen2.5-Coder-1.5B：代码能力参考；
- SmolLM2-1.7B：小模型数据工程、数学和指令能力参考。

## 文档

- [架构与横向比较方案](docs/architecture-comparison-plan.md)
- [基础预训练数据规划 v0.2](docs/base-pretraining-data-plan.md)
- [效用巩固记忆 v1：冻结的最小原型](docs/utility-consolidation-memory-v1.md)
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

中文原文是主文档，英文原稿和 LaTeX 原稿与它同步维护：

```text
README.md                 中文原文
README.en.md              英文原稿
docs/*.md                 中文原文分册
docs/en/*.md              英文原稿分册
paper/main.tex            LaTeX 原稿（英文，tectonic 编译）
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
