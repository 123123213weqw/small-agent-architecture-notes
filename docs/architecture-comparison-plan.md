# 1B Agent 模型架构与横向比较方案

> 本文保留旧记忆架构的 v0.3 横向比较计划，当前暂停；现在的从零训练方向见[决策记录 ADR-012](decisions.md)。旧研究的 v2 数学见[效用巩固记忆 v2](utility-consolidation-memory-v2.zh.md)。

> 版本：v0.3
>
> 状态：历史计划，不用于当前 GDN＋GQA Base。
>
> 目标：从头训练约 1B 参数、面向工具调用、简单代码、基础数学和长任务状态保持的小型 Agent 模型。

## 1. 项目目标

模型定位为小型 Agent 控制器，而不是覆盖所有知识的通用聊天模型。核心能力包括：

1. 理解中英文任务；
2. 生成结构正确的工具调用；
3. 编写和修复简单代码；
4. 完成基础数学推理；
5. 解释工具结果并从失败中有限纠错；
6. 在固定状态预算下维持长任务中的目标、约束、状态更新和关键工具结果；
7. 输出简洁、可验证的最终答案。

精确计算和程序执行交给外部工具。模型重点学习任务分解、工具选择、参数构造、结果解释、停止条件，以及有限容量下的记忆取舍。

### 1.1 第一版任务范围

- 基础算术、代数、比例、单位换算和简单应用题；
- Python 小程序和 Bash/Shell 基础任务；
- 阅读小型代码文件并根据单元测试失败修复代码；
- Calculator、Python、Shell、文件和测试工具调用；
- 8–64 步的受控 Agent 或状态跟踪轨迹；
- 多次覆盖变量、约束、文件状态和工具结果后的最终状态查询；
- 关键事件超出局部窗口后的选择性召回。

### 1.2 第一版不追求

- 大型仓库级软件工程；
- 高难竞赛数学；
- 跨用户、跨会话的永久个性化记忆；
- 多 Agent 协作、GUI 自动化和多模态；
- 无损保存任意长度历史；
- 在短上下文或无限算力条件下全面超过 Full Attention。

## 2. 当前架构状态

### 2.1 已冻结：效用巩固记忆 v1

每次只有一个候选进入长期记忆竞争：

$$
\mathcal R_t=M_t\cup\lbrace c_t\rbrace.
$$

训练期使用决策之后的反事实未来损失生成效用标签：

$$
u_{t,i} =
\mathcal L_{\bar\theta,F}
\left(Y_t\mid B_t,\mathrm{mask}_i(\mathcal R_t)\right) -
\mathcal L_{\bar\theta,F}
\left(Y_t\mid B_t,\mathcal R_t\right).
$$

预测器只能使用决策时已经观察到的信息：

$$
\hat u_{t,i}=f_\phi(m_i,g_t,B_t,\mathcal R_t).
$$

容量超限时只淘汰预测效用最低的一条：

$$
M_{t+1} =
\mathcal R_t\setminus
\left\lbrace \arg\min_i\hat u_{t,i}\right\rbrace.
$$

完整定义见 [效用巩固记忆 v1](utility-consolidation-memory-v1.md)。

### 2.2 v1 明确排除

- 显式年龄、occupancy 或时间衰减；
- 新奇度、surprise 或冲突门；
- 事件关联图；
- soft write；
- 记忆内容插值、改写或合并；
- Gumbel-Softmax 或可微 Top-K；
- 推理时反向传播或未来信息访问。

### 2.3 仍未冻结

```text
参数规模：约 1.0B–1.3B
模型类型：自回归文本生成模型
训练精度：BF16
主要语言：中文和英文
主要领域：工具调用、简单代码、基础数学、长任务状态保持
部署目标：单张 16GB–48GB GPU 可推理
```

仍需受控实验决定：

- 主干使用 Full/Local Attention、线性注意力或混合结构；
- 层数、hidden size、FFN 宽度和记忆读取层位置；
- 局部窗口 $W$；
- 长期容量 $S$、等待期 $D$ 和监督区间 $F$；
- 事件记录粒度、编码维度和原文保留方式；
- Utility Predictor 的具体实现；
- 位置编码、Tokenizer、数据混合和训练 token。

冻结记忆公式不等于已经决定最终 1B 主干。

## 3. 公开模型和架构参考

### 3.1 TinyLlama-1.1B：标准小型 Transformer

| 项目 | 配置 |
|---|---:|
| 参数量 | 约 1.1B |
| 层数 | 22 |
| Hidden size | 2048 |
| FFN size | 5632 |
| Query/KV heads | 32/4 |
| Attention | GQA |
| 激活/归一化 | SwiGLU/RMSNorm |
| 位置编码 | RoPE |
| 词表/上下文 | 32K/2048 |

参考：[TinyLlama 官方仓库](https://github.com/jzhang38/tinyllama)、[预训练说明](https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md)。

### 3.2 OLMo-1B：开放训练参考

用于比较训练曲线、数据效率和相近 token 数下的语言模型能力。参考：[OLMo 官方仓库](https://github.com/allenai/OLMo)、[OLMo-1B 模型卡](https://huggingface.co/allenai/OLMo-1B)。

### 3.3 RWKV-7：固定递归状态参考

RWKV-7 使用广义 Delta Rule 和向量门控维护固定递归状态，是连续状态对照。它用于检验离散、可寻址事件记录是否真的优于持续压缩历史的状态矩阵。

参考：[RWKV-7](https://arxiv.org/abs/2503.14456)、[官方仓库](https://github.com/BlinkDL/RWKV-LM)。

### 3.4 Kimi Linear / KDA：线性注意力参考

KDA 的核心状态更新可概括为：

$$
S_t=(D_t-a_tb_t^\top)S_{t-1}+k_tv_t^\top.
$$

它用于比较连续矩阵状态、混合 Full Attention 和离散事件保留之间的质量与效率差异。参考：[Kimi Linear](https://arxiv.org/abs/2510.26692)。

Kimi K3 是 KDA、Gated MLA、Attention Residuals 和 Stable LatentMoE 的大规模组合，不作为 1B 严格能力基线。

### 3.5 其他能力与混合架构参考

- [Hymba-1.5B](https://arxiv.org/abs/2411.13676)：小模型混合架构、缓存和长上下文效率；
- [Qwen2.5-Coder-1.5B](https://arxiv.org/abs/2409.12186)：简单代码和 Code Agent 能力；
- [SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/blob/main/README.md)：小模型数据工程、数学、指令和 Function calling。

### 3.6 相邻记忆方法

- [MEM1](https://arxiv.org/abs/2506.15841)：固定容量的推理—记忆联合状态；
- [Memory-R2](https://arxiv.org/abs/2605.21768)：长程记忆操作的局部与全局信用分配；
- [Titans](https://arxiv.org/abs/2501.00663)：Attention 与测试时神经长期记忆；
- [Infini-attention](https://arxiv.org/abs/2404.07143)：局部注意力与有界压缩记忆；
- [Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881)：跨片段递归 memory token。

## 4. 对比职责

| 对象 | 主要比较内容 | 严格架构对照 |
|---|---|---:|
| TinyLlama-1.1B | 标准小型 GQA、参数效率 | 否 |
| OLMo-1B | 开放训练过程、相近 token 预算 | 否 |
| RWKV-7 | 固定连续状态、吞吐、状态跟踪 | 受控实现中是 |
| KDA | 线性注意力、连续矩阵状态 | 受控实现中是 |
| Hymba/Qwen/SmolLM2 | 混合、代码、数学和 Agent 能力 | 否 |
| FIFO/随机淘汰 | 记忆策略下界 | 是 |
| Future Oracle 淘汰 | Reader 与记录表示的离线上界 | 是 |
| 预测效用淘汰 | 本项目可部署策略 | 是 |

## 5. 核心研究假设

本项目不主张全面超过 Transformer 或线性注意力。待验证主张为：

> 在相同常驻状态预算下，对于关键事件稀疏、任务很长且状态会被多次覆盖的轨迹，使用决策后反事实未来损失训练的单次淘汰策略，能够比按时间淘汰和连续压缩状态更可靠地保留未来有用的信息。

### 5.1 相对 Full/Local Transformer

全上下文 Transformer 的 KV 状态随总长度增长：

$$
O(Td).
$$

本项目使用局部窗口 $W$ 和长期容量 $S$：

$$
O((W+S)d).
$$

相对滑动窗口的差异不是状态是否有界，而是旧记录按预测未来价值而不是纯时间顺序淘汰。

### 5.2 相对线性或递归状态模型

线性注意力和 RWKV/KDA 的优势是固定状态和高效逐 token 更新。本项目的假设优势是：

- 事件记录保持独立身份；
- 单条记录可寻址、遮蔽和删除；
- 历史不必持续混入同一个矩阵状态；
- 记忆取舍具有直接的反事实未来效用监督；
- 延迟决策能利用事件发生后已经观察到的证据。

在线性模型的纯吞吐、稠密序列建模和未来用途高度不可预测的任务上，本项目可能处于劣势。

## 6. 公平比较规则

### 6.1 公开横向比较与受控实验分开

公开模型用于回答最终能力水平和部署效率；受控小模型用于架构归因。不同 Tokenizer、数据和训练 token 的公开模型不能用于证明效用记忆有效。

### 6.2 参数与状态预算

受控模型匹配总参数、非 Embedding 参数或训练 FLOPs，并完整报告其他项。状态预算必须包括：

$$
\text{local KV}
+\text{short-term buffer}
+\text{long-term records}
+\text{metadata}.
$$

如果短期缓冲完全包含在局部窗口中，应明确说明，避免重复计算。

### 6.3 跨 Tokenizer 指标

不直接比较 perplexity，统一使用 Bits Per Byte：

$$
\mathrm{BPB} =
\frac{\mathrm{NLL}}
{N_{\mathrm{bytes}}\ln2}.
$$

分别报告中文、英文、代码、数学和 API/JSON BPB。

## 7. 评价指标

### 7.1 通用、数学、代码和 Agent

- 中英文 held-out BPB；
- 基础数学与工具辅助数学准确率；
- Python pass@1、语法正确率和单元测试通过率；
- 根据错误日志修复代码的成功率；
- Tool-call JSON 合法率；
- 工具选择和参数构造准确率；
- 多步任务成功率、失败恢复率和最终答案正确率。

### 7.2 长程记忆

- 跨局部窗口事实召回；
- 多次状态覆盖后的最新值正确率；
- 旧状态误用率；
- 无关工具输出增加时的性能下降；
- 不同依赖距离 $D,F$ 下的召回；
- 记录淘汰准确率；
- Utility Predictor 与 Oracle 的排序相关性；
- 预测策略相对 Oracle 的选择 regret；
- 完整留出领域上的记忆策略迁移。

### 7.3 效率

- Prefill 和 Decode tokens/s；
- 首 token、单 token 和 consolidation 延迟；
- 峰值显存和全部状态大小；
- 每千 token consolidation 次数；
- 反事实标签生成 FLOPs；
- 总训练 tokens/s、单步时间和 MFU。

推理阶段不得把离线反事实标签成本算入在线延迟；训练成本必须完整报告。

## 8. 实验阶段

### 8.1 阶段 0：无语言模型的选择器验证

在合成状态机和 key-value 覆盖任务上验证：

1. 单次淘汰公式和 mask 实现正确；
2. Future Oracle 是否显著超过 FIFO；
3. 记录冗余或互补时，单次淘汰解释是否成立；
4. 候选顺序对贪心策略的影响。

Oracle 不超过 FIFO 时，不进入更大模型实验。

### 8.2 阶段 1：20M–50M Reader 与 Utility Predictor

1. 先训练 Reader 使用外部记录；
2. 冻结或 EMA Reader/事件编码器；
3. 生成少量精确反事实标签；
4. 训练 Utility Predictor；
5. 比较 FIFO、随机、预测效用和 Oracle。

### 8.3 阶段 2：100M–150M 受控架构实验

匹配 Tokenizer、数据顺序、参数/FLOPs、batch、优化器、训练 token 和随机种子。比较：

- Local Transformer + FIFO；
- Local Transformer + 效用巩固记忆；
- 一个 RWKV/KDA 类连续状态基线；
- 必要时 Full Attention 上界。

### 8.4 阶段 3：300M–400M 扩展验证

只扩展同时满足以下条件的方案：

- Oracle 有明显上界；
- 预测策略显著接近 Oracle；
- 同状态预算下超过 FIFO 和连续状态基线；
- 标签成本仍可接受。

### 8.5 阶段 4：最终约 1B 模型

最终只训练一个约 1B 候选。Transformer、线性或混合主干由阶段 2–3 的质量—效率曲线决定。

## 9. Agent 工具协议草案

第一版预计支持：

```text
calculator
python
shell
read_file
apply_patch
run_tests
```

初始限制：

```yaml
max_agent_steps: 64
max_retries_per_step: 2
python_timeout_seconds: 10
shell_timeout_seconds: 10
max_tool_output_chars: 12000
```

短能力评测保留 8 步子集；记忆评测逐步扩展到 16、32 和 64 步。

## 10. 结果记录模板

### 10.1 记忆选择

| 策略 | 总状态预算 | 最终成功率 | 旧值误用率 | 选择 Regret | Consolidation 延迟 |
|---|---:|---:|---:|---:|---:|
| FIFO | | | | | |
| Random | | | | | |
| Predicted Utility | | | | | |
| Future Oracle | | | | | |

### 10.2 架构质量与效率

| 模型 | 参数 | 训练 Token | BPB | Agent 成功率 | Decode tok/s | 状态大小 |
|---|---:|---:|---:|---:|---:|---:|
| Local Transformer | | | | | | |
| Linear/Recurrent | | | | | | |
| Utility Memory | | | | | | |
| Full Attention Upper Bound | | | | | | |

## 11. 当前决策与停止条件

已确定：

- 约 1B 小型 Agent 定位；
- 工具调用、简单代码、基础数学和长任务状态保持；
- 不重复训练完整传统 1B 基线；
- 公开模型承担最终能力参照；
- 小规模受控实验承担架构归因；
- 效用巩固记忆 v1 的核心公式；
- 单候选进入、单记录淘汰和不可变事件记录；
- 第一版不增加其他记忆门控或关联模块。

仍未确定：

- 完整主干序列混合结构；
- 深度、宽度和记忆读取层位置；
- $W,S,D,F$；
- 事件记录编码；
- 标签采样率和教师更新日程；
- Tokenizer、数据混合和最终训练 token。

停止条件：

1. Future Oracle 不超过同预算 FIFO；
2. 预测效用长期无法明显接近 Oracle；
3. 相同状态预算下不超过滑动窗口或连续状态基线；
4. 标签成本无法通过采样、缓存或蒸馏降到可接受范围。
