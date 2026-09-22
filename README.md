# Small Agent Architecture Notes

面向约 1B 参数小型 Agent 模型的公开研究记录。

当前目标聚焦于：

- 工具调用与短程任务规划；
- 简单 Python、Shell 和代码修复；
- 基础数学与工具辅助计算；
- 小模型上的训练效率、推理延迟和状态缓存；
- 新架构与公开小模型的可复现横向比较。

## 当前状态

项目处于架构定义阶段，尚未锁定最终模型结构。本仓库目前只记录：

1. 研究目标与边界；
2. 公开对比模型；
3. 公平比较方法；
4. 实验阶段和决策记录。

在完成结构论证前，不在文档中预设具体的新型 Attention、状态空间模块或残差连接方案。

## 对比模型

- TinyLlama-1.1B：标准小型 GQA 架构参考；
- OLMo-1B：开放训练过程和中间 checkpoint 参考；
- Hymba-1.5B：公开小型混合架构参考；
- Qwen2.5-Coder-1.5B：代码能力参考；
- SmolLM2-1.7B：小模型数据工程、数学和指令能力参考。

## 文档

- [架构与横向比较方案](docs/architecture-comparison-plan.md)
- [效用巩固记忆 v1：冻结的最小原型](docs/utility-consolidation-memory-v1.md)
- [早期持久情景记忆推导（已被简化版取代）](docs/candidate-persistent-episodic-memory.md)
- [决策记录](docs/decisions.md)

## 原则

- 最终不重复训练完整传统 1B 基线；
- 公开模型用于最终横向比较；
- 必要时仅训练低成本的小规模受控对照；
- 不把不同 Tokenizer 的 perplexity 直接横向比较；
- 同时报告质量、Agent 成功率、代码/数学能力和效率；
- 架构结论必须经过参数、数据和训练 token 匹配的实验验证。

## License

Apache-2.0
