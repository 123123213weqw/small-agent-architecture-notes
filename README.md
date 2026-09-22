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

项目处于最小原型定义阶段。已经冻结效用巩固记忆 v1 的核心公式，但尚未锁定最终 1B 主干。当前状态：

1. 长期记忆采用不可变事件记录；
2. 每次只有一个候选进入竞争，容量超限时只淘汰一条；
3. 训练期用决策之后的反事实未来损失监督效用预测器；
4. 推理期不访问未来，只执行效用预测、单次淘汰和标准 Cross-Attention 读取；
5. Transformer、线性注意力或混合主干仍由小规模实验决定。

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
- [效用巩固记忆 v1：冻结的最小原型](docs/utility-consolidation-memory-v1.md)
- [早期持久情景记忆推导（已被简化版取代）](docs/candidate-persistent-episodic-memory.md)
- [决策记录](docs/decisions.md)

## 公式书写约定

- 行内公式用 `$...$`，独立公式用单独的 `$$` 行包围；`\[...\]` 和 `\(...\)` 在 GitHub 上不渲染；
- 独立公式内不要出现只有 `=` 或 `-` 的行，否则会被 Markdown 当成标题解析；
- 公式内的大小比较写 `\lt`、`\gt`，避免 `<`、`>` 被转义；
- 公式里的花括号写 `\lbrace`、`\rbrace`，因为 `\{`、`\}` 的反斜杠会被 Markdown 吃掉；
- 算子名写 `\mathrm{...}`，不要用 `\operatorname{...}`：GitHub 的宏白名单不含它，页面会显示红框 `The following macros are not allowed: operatorname`。

## 原则

- 最终不重复训练完整传统 1B 基线；
- 公开模型用于最终横向比较；
- 架构归因从无语言模型、20M–50M、100M–150M 到 300M–400M 分阶段完成；
- 不把不同 Tokenizer 的 perplexity 直接横向比较；
- 同时报告质量、Agent 成功率、记忆选择 regret、代码/数学能力和效率；
- 状态预算必须同时计算局部 KV、短期缓冲、长期记录和元数据；
- Future Oracle 不超过同预算 FIFO 时停止该方向；
- 架构结论必须经过参数、数据和训练 token 匹配的实验验证。

## License

Apache-2.0
