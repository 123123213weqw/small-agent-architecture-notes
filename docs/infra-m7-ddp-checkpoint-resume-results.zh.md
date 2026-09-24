# M7：8 卡 checkpoint 保存与恢复实验

日期：2026-09-24
阶段：1B 工程基础设施验证

## 1. 实现内容

训练器现在支持单机 DDP 完整 checkpoint。格式版本为：

```text
small_agent_distributed_full_checkpoint_v1
```

每个 checkpoint 包含：

- 一份共享 `model.safetensors`；
- 一份共享 AdamW optimizer state；
- 一份共享学习率 scheduler state；
- 8 份 rank-local sampler state；
- 8 份 rank-local PyTorch、CUDA 和 DataLoader generator RNG state；
- 一份记录兼容条件和全部文件 SHA-256 的 manifest。

共 19 个受校验的 artifact。保存时先写临时目录；所有 rank 交换写入结果，只有全部成功后 rank 0 才生成 manifest、原子重命名并更新 `latest.json`。加载时先由 rank 0 校验全部大小与 SHA-256，再由各 rank 恢复共享训练状态和自己的游标、RNG 状态。

当前格式故意要求恢复时 world size、模型配置、数据 manifest、batch 结构、DDP bucket 和通信 hook 不变。它不是 elastic checkpoint。

## 2. 实验设计

不能用两个独立初始化的训练任务直接判断恢复误差，因为当前 FLA/Triton 快速内核与 BF16 通信路径没有配置成跨进程启动的 bitwise deterministic 模式。

因此采用同源分支：

1. source 从第 1 步训练到第 20 步；
2. source 在第 10 步保存 checkpoint，但不退出，继续训练到第 20 步；
3. branch 复制 source 的同一个第 10 步 checkpoint；
4. branch 启动新进程，从第 10 步恢复并训练到第 20 步；
5. 比较 source 与 branch 的第 11～20 步。

两组均使用 C 配置：8 张 L40、4096 上下文、microbatch 1、累积 4、关闭 activation checkpointing、BF16 DDP 通信。

## 3. Checkpoint 结果

| 指标 | 结果 |
|---|---:|
| 单份 checkpoint 大小 | 11.235 GiB |
| artifact 数量 | 19 |
| 第 10 步保存耗时 | 23.761 秒 |
| source 第 20 步保存耗时 | 23.344 秒 |
| branch 第 20 步保存耗时 | 22.757 秒 |
| 完整性校验 | 通过 |
| 8 份 sampler/RNG 状态 | 齐全 |
| 恢复后完成步数 | 20/20 |

第 10 步 source 与 branch 的全部 artifact 哈希完全相同。加载时也重新计算并验证了所有文件的 SHA-256。

## 4. 数据连续性

第 11～20 步中，两条路径的以下项目逐步完全相同：

- epoch；
- sampler cursor；
- tokens seen；
- learning rate。

第 20 步均为：

```text
epoch           2
sampler_cursor  16
tokens_seen     2,620,800
```

这说明恢复后没有重复读取或跳过训练序列。

## 5. 数值连续性

| 指标 | source 与 resumed 的差异 |
|---|---:|
| 最大训练 loss 绝对差 | 0.000254 |
| 平均训练 loss 绝对差 | 0.0000766 |
| 最大梯度范数绝对差 | 0.00833 |
| 最终验证 loss 差 | 0.000951 |

最终模型 SHA-256 不相同，因此结果不是逐 bit 相同。当前验收结论是：

- 数据轨迹严格一致；
- checkpoint 文件完整且兼容条件受约束；
- 训练轨迹在工程容差内数值等价；
- 不宣称跨进程启动 bitwise reproducibility。

如果以后需要逐 bit 重放，需要单独建立 deterministic 模式，逐项替换或约束 FLA/Triton 内核、CUDA 算法和集合通信；这不应该拖慢默认性能配置。

## 6. 显存影响

| 路径 | 峰值已分配 | 峰值保留 |
|---|---:|---:|
| source 连续训练 | 34.99 GiB | 36.75 GiB |
| resumed | 34.99 GiB | 38.21 GiB |

恢复后的实际峰值分配没有增加，但 allocator 保留显存增加约 1.46 GiB。L40 仍能稳定运行，不过正式长跑应继续监控恢复后的 reserved memory。

## 7. 局限与下一步

该实现是 replicated full checkpoint：简单、可审计，但每次保存会暂停约 23 秒，并由每个 rank 读取同一份模型和 optimizer 文件。对当前 1B 模型可接受；模型继续扩大时，应迁移到 sharded/distributed checkpoint，避免 rank 0 写入和共享存储读取成为瓶颈。

机器可读摘要与原始指标位于：

```text
results/infra_m7_ddp_checkpoint_resume_v1/
```
