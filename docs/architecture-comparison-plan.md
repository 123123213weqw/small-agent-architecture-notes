# 1B Agent 模型架构与横向比较方案

> 版本：v0.2  
> 状态：架构尚未确定  
> 目标：从头训练约 1B 参数、面向工具调用、简单代码和基础数学的小型 Agent 模型。

## 1. 项目目标

模型定位为小型 Agent 控制器，而不是覆盖所有知识的通用聊天模型。核心能力包括：

1. 理解中英文任务；
2. 判断是否需要调用工具；
3. 生成结构正确的工具调用；
4. 编写和修复简单代码；
5. 阅读执行结果和错误日志；
6. 完成基础数学推理；
7. 在工具调用失败后进行有限纠错；
8. 输出简洁、可验证的最终答案。

精确计算和程序执行交给外部工具，模型重点学习任务分解、工具选择、参数构造、结果解释和停止条件。

### 1.1 第一版任务范围

- 基础算术、代数、比例、单位换算和简单应用题；
- Python 小程序；
- Bash/Shell 基础任务；
- 阅读小型代码文件；
- 根据单元测试失败修改代码；
- Calculator、Python、Shell、文件和测试工具调用；
- 最多约 8 步的短程 Agent 轨迹。

### 1.2 第一版不追求

- 大型仓库级软件工程；
- 高难竞赛数学；
- 长期记忆；
- 多 Agent 协作；
- GUI 自动化；
- 多模态；
- 超长上下文。

## 2. 当前架构状态

最终架构尚未确定。目前只冻结以下外部约束：

```text
参数规模：约 1.0B–1.3B
模型类型：自回归文本生成模型
目标上下文：至少 4K，期望支持 8K
主要语言：中文和英文
主要领域：工具调用、简单代码、基础数学
训练精度：BF16
部署目标：单张 16GB–48GB GPU 可推理
```

以下内容暂不冻结：

- 层数与隐藏维度；
- Attention 或其他序列混合模块；
- 残差连接结构；
- FFN 形式与宽度；
- 状态缓存形式；
- 位置编码；
- 词表大小；
- 是否采用层间或层内混合结构。

任何候选结构都应单独形成设计提案，写清数学定义、参数量、计算复杂度、缓存复杂度、kernel 可用性和预期收益，再进入实验。

## 3. 公开对比模型

### 3.1 TinyLlama-1.1B：主要架构参考

TinyLlama 是标准小型 Decoder Transformer 的主要参照。

| 项目 | 配置 |
|---|---:|
| 参数量 | 约 1.1B |
| 层数 | 22 |
| Hidden size | 2048 |
| FFN size | 5632 |
| Query heads | 32 |
| KV heads | 4 |
| Attention | GQA |
| 激活函数 | SwiGLU |
| 归一化 | RMSNorm |
| 位置编码 | RoPE |
| 词表 | 32K |
| 上下文 | 2048 |

TinyLlama 提供多个中间 checkpoint。接近本项目训练预算的中间版本可用于观察同量级模型在不同训练 token 下的能力变化。

参考：

- [TinyLlama 官方仓库](https://github.com/jzhang38/tinyllama)
- [TinyLlama 预训练说明](https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md)

### 3.2 OLMo-1B：开放训练参考

OLMo-1B 用于比较训练曲线、数据效率和相近 token 数下的语言模型能力。

| 项目 | 配置 |
|---|---:|
| 参数量 | 约 1B |
| 层数 | 16 |
| Hidden size | 2048 |
| Attention heads | 16 |
| Attention | Full Attention |
| 上下文 | 2048 |

OLMo 开放模型代码、训练配置、数据说明和大量中间 checkpoint，适合作为训练过程参考。

参考：

- [OLMo-1B 模型卡](https://huggingface.co/allenai/OLMo-1B)
- [OLMo 官方训练代码](https://github.com/allenai/OLMo)
- [OLMo-1B 官方配置](https://github.com/allenai/OLMo/blob/main/configs/official-0724/OLMo-1B.yaml)

### 3.3 Hymba-1.5B：公开混合架构参考

Hymba 用于比较小模型混合架构、长上下文召回、状态缓存和推理吞吐。它不作为严格受控基线，因为模型规模、训练数据和配方与本项目不同。

参考：

- [Hymba 论文](https://arxiv.org/abs/2411.13676)
- [Hymba 官方代码](https://github.com/NVlabs/hymba)

### 3.4 Qwen2.5-Coder-1.5B：代码能力参考

用于比较简单代码生成、代码补全、错误修复和 Code Agent 基础能力。它不用于直接证明架构优势。

参考：[Qwen2.5-Coder 技术报告](https://arxiv.org/abs/2409.12186)

### 3.5 SmolLM2-1.7B：小模型能力和数据工程参考

用于比较通用能力、数学、指令遵循和 Function calling。其训练 token 和数据工程投入显著高于本项目，因此主要作为能力上界与数据配方参考。

参考：[SmolLM2-1.7B 模型卡](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/blob/main/README.md)

## 4. 对比模型职责

| 模型 | 主要比较内容 | 严格架构对照 |
|---|---|---:|
| TinyLlama-1.1B | 标准小型 GQA、参数效率 | 否 |
| OLMo-1B | 开放训练过程、相近 token 预算 | 否 |
| Hymba-1.5B | 混合架构、缓存和长上下文效率 | 否 |
| Qwen2.5-Coder-1.5B | 代码能力 | 否 |
| SmolLM2-1.7B | 数学、指令、Function calling | 否 |
| 本地小规模受控模型 | 同数据、同配方下的架构差异 | 是 |

## 5. 比较方法

### 5.1 两类比较必须分开

#### 公开模型横向比较

用于回答：

- 最终模型处于什么能力水平；
- 推理延迟和显存是否有优势；
- Agent、代码和数学能力与现有小模型相比如何。

#### 小规模受控比较

用于回答：

- 新架构本身是否改善模型；
- 改善是否来自数据或训练配方；
- 新模块是否稳定、可扩展且具备实际硬件效率。

公开模型不能替代受控架构实验，因为它们的 Tokenizer、数据和训练 token 不同。但受控实验不需要训练完整 1B 传统模型，可以在较小规模完成。

### 5.2 不跨 Tokenizer 直接比较 perplexity

跨模型统一使用 Bits Per Byte：

\[
\operatorname{BPB}=
\frac{\operatorname{NLL}}
{N_{bytes}\ln 2}
\]

分别报告：

- 中文 BPB；
- 英文 BPB；
- 代码 BPB；
- 数学 BPB；
- API/JSON BPB。

## 6. 能力指标

### 6.1 通用能力

- 中文与英文 held-out BPB；
- 基础常识和阅读理解；
- 自建无污染测试集。

### 6.2 数学能力

- 基础算术；
- 比例和单位换算；
- 简单代数；
- 简单应用题；
- 工具辅助数学准确率。

### 6.3 代码能力

- 简单 Python pass@1；
- 语法正确率；
- 单元测试通过率；
- 根据错误日志修复代码的成功率；
- 代码和变量名精确复制。

### 6.4 Agent 能力

- Tool-call JSON 合法率；
- 工具选择准确率；
- 参数构造准确率；
- 不必要工具调用率；
- 多步任务成功率；
- 工具失败恢复率；
- 最大步数超限率；
- 最终答案正确率。

### 6.5 长程召回

- Passkey retrieval；
- Needle-in-a-haystack；
- 文件名、JSON 字段和数值精确复制；
- 不同上下文距离下的召回准确率。

## 7. 效率指标

所有模型在相同硬件、精度和运行时下测试。至少记录：

- Prefill tokens/s；
- Decode tokens/s；
- UTF-8 bytes/s；
- 首 token 延迟；
- 单 token 延迟；
- 峰值显存；
- KV Cache 或状态缓存大小；
- 训练 tokens/s；
- 单步时间；
- MFU；
- 近似训练 FLOPs。

推荐覆盖：

```text
Batch：1、8
Prompt：1K、4K、8K
Generation：128、512
Temperature：0
```

## 8. 实验阶段

### 8.1 架构纸面设计

每个候选结构首先提交一份设计说明，至少包含：

1. 模块数学定义；
2. 参数量公式；
3. 训练计算复杂度；
4. Prefill 和 Decode 复杂度；
5. 缓存或状态大小；
6. 初始化方法；
7. 数值稳定性风险；
8. 所需 kernel 与框架支持；
9. 与公开模型的差异；
10. 可证伪的收益假设。

### 8.2 小规模受控实验

先使用约 100M–150M 参数模型完成快速比较。所有候选模型必须匹配：

- Tokenizer；
- 数据和数据顺序；
- 参数量或训练 FLOPs；
- global batch；
- 优化器和学习率；
- 随机种子；
- 训练 token；
- 评测集。

### 8.3 中等规模验证

小规模胜出方案扩展至约 300M–400M 参数，训练更长 token，用于检查架构优势是否能够随规模扩大。

### 8.4 最终模型

最终只训练一个约 1B 的候选模型。是否训练 60B、100B 或更多 token，根据前期 scaling curve、数据准备情况和计算预算决定。

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

工具调用使用严格 JSON Schema。运行时限制最大步数、执行时间、重试次数和输出长度。

建议初始限制：

```yaml
max_agent_steps: 8
max_retries: 2
python_timeout_seconds: 10
shell_timeout_seconds: 10
max_tool_output_chars: 12000
```

具体特殊 token 和消息格式在 Tokenizer 方案确定后冻结。

## 10. 结果记录模板

### 10.1 能力

| 模型 | 参数 | 训练 Token | 中文 BPB | 代码 BPB | 数学 BPB | Agent 成功率 |
|---|---:|---:|---:|---:|---:|---:|
| TinyLlama-1.1B | | | | | | |
| OLMo-1B | | | | | | |
| Hymba-1.5B | | | | | | |
| Qwen2.5-Coder-1.5B | | | | | | |
| SmolLM2-1.7B | | | | | | |
| Candidate | | | | | | |

### 10.2 效率

| 模型 | 4K Prefill | 8K Prefill | Decode tok/s | 首 Token 延迟 | 峰值显存 | Cache/State |
|---|---:|---:|---:|---:|---:|---:|
| TinyLlama-1.1B | | | | | | |
| OLMo-1B | | | | | | |
| Hymba-1.5B | | | | | | |
| Candidate | | | | | | |

## 11. 当前决策

已确定：

- 项目目标是小型 Agent，而非通用聊天模型；
- 主要能力是工具调用、简单代码和基础数学；
- 不重复训练完整传统 1B 基线；
- 公开模型承担最终横向比较；
- 小规模受控实验承担架构归因；
- 最终架构尚未确定。

尚未确定：

- 序列混合结构；
- 残差结构；
- 深度与宽度；
- Tokenizer；
- 上下文训练策略；
- 数据混合比例；
- 最终训练 token 数。

在上述问题完成论证前，不把任何具体新架构写成既定方案。
