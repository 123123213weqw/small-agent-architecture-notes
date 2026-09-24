# P0 原生 GDN+GQA 小模型正确性测试

日期：2026-09-24。目标只限于验证混合结构能够从随机初始化前向、反向、逐 token 推理、短训练和断点恢复；**不是**模型质量比较或 GDN 吞吐基准。

## 配置

- 代码：`experiments/p0_gdn_hybrid_smoke.py`；使用 Transformers 4.57.1 的 `Qwen3NextForCausalLM` GDN 实现，随机初始化，未加载任何 Qwen 权重。
- 8 层：`[GDN, GDN, GDN, full GQA] × 2`；dense FFN、无 MoE、文本专用。
- 词表 32,768（冻结 `tokenizer_v1`），hidden 512，FFN 1536，GQA Q 头 8/KV 头 2/head dim 64，GDN key/value 头均为 8/head dim 64、卷积核 4；输入输出 embedding 共享。
- 实例化参数量：45,446,368。GDN 缺少 `fla` 和 `causal_conv1d` 快速内核，本次明确使用 PyTorch 回退实现。
- 数据：L40 `/data1/wangyue/experiments/small-agent-p0-engineering-v1/data/` 的冻结 token 流；许可门仍未通过，仅内部工程冒烟。

## 结果

- 12-token FP32 测试：一次整段前向与带缓存逐 token 前向的 logits 最大绝对差 `3.4571e-6`；greedy token 完全一致。
- 17-token 初始反向：loss `10.5072`、embedding 梯度范数 `25.3565`，梯度非零且有限。
- L40 单卡，BF16 autocast，短训练使用 context 128、batch 2；从 step 0 跑到 4，保存 checkpoint，再恢复到 step 16。恢复后 checkpoint 可重新加载，模型权重全有限。
- step 4/16 的验证 loss：`10.4483 → 10.0528`。总计仅处理 `4,096` 训练 token，因此这只能说明工程链路运转，不能用来判断学习能力、收敛或架构优劣。
- 结果位于 L40 `/data1/wangyue/experiments/small-agent-p0-gdn-hybrid-smoke-v1/run/`，包含 `correctness.json`、`checkpoint.pt` 和 `summary.json`。测试结束后 GPU 已释放。

## 未完成的门槛

1. 安装并锁定适配 L40 的训练快速内核，再做长序列前向/反向和 2048/4096 上下文的数值、吞吐与显存测试；不能用本次慢速回退数据评估性能。
2. 将训练从 16 步扩展到独立验证的更大且训练许可已确认的数据；固定同参数/同 token/同预算 Transformer 对照后，才允许作质量比较。
3. 本次继承的是 Qwen3Next 的 GDN 核心实现，未冻结约 1B 的最终模型配置，也未实现或评价旧效用记忆结构。
