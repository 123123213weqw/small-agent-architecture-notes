# M4：单节点 DDP 与性能分析计划

日期：2026-09-24。M4 只回答训练基础设施能否在 L40 单节点多卡正确运行，以及时间主要消耗在哪里；不使用这轮 loss 判断模型能力。

## 实现范围

正式训练器 `src/small_agent/training/trainer.py` 同时支持单卡和 `torchrun`：

- 根据 `WORLD_SIZE/RANK/LOCAL_RANK` 初始化 NCCL；
- `StatefulDistributedSampler` 将训练样本确定性分给各 rank；
- DDP 使用 `broadcast_buffers=false` 和 `gradient_as_bucket_view=true`；
- 梯度累积的非末尾 microstep 使用 `no_sync()`，每个优化器步只同步一次梯度；
- 学习率调度器按所有 rank 的全局有效 token 数前进；
- loss 取跨 rank 均值，耗时和显存取跨 rank 最大值；
- 验证集按 rank 交错切分，再按有效 token 加权汇总；
- 只有 rank 0 写 JSONL、TensorBoard、状态和运行清单。

M4 暂时禁止 DDP checkpoint。M3 的单卡完整恢复已经通过，但多 rank 的 RNG、sampler 和分布式优化器状态需要独立验收，不能把 rank 0 状态冒充所有 rank 状态。

## 基准矩阵

配置：`configs/runs/base_1b_m4_ddp_l40_v1.json`。

| 组 | GPU | 每卡 microbatch | 累积 | 每步序列数 | 目的 |
|---|---:|---:|---:|---:|---|
| S1 | 1 | 1 | 4 | 4 | 新训练器单卡基线 |
| D2 | 2 | 1 | 4 | 8 | 小规模 NCCL 扩展 |
| D4 | 4 | 1 | 4 | 16 | 同节点四卡扩展 |
| D8 | 8 | 1 | 4 | 32 | 目标单节点配置 |

每组运行 12 个优化器步。step 1–3 包含 Triton/图形预热，不进入稳定吞吐统计；报告 step 4–12 的中位数、P10/P90、最大显存以及：

\[
E_N=\frac{T_N}{N T_1}
\]

其中 `T_N` 是 N 卡全局 token/s，`E_N` 是相对单卡的扩展效率。不同卡数组每步全局 batch 不同，因此 loss 只做有限值检查，不横向解释收敛优劣。

## Profiler

配置：`configs/runs/base_1b_m4_profiler_l40_v1.json`。在基准完成后单独启动，避免 profiler 开销污染吞吐结果：

- wait 3 steps；
- warmup 2 steps；
- active 3 steps；
- rank 0 保存 TensorBoard trace 和按 self CUDA time 排序的 operator table；
- 重点分离 GDN/卷积、GQA、FFN、AdamW、重计算和 NCCL all-reduce。

## 验收

1. 1/2/4/8 卡均完成前反向和 AdamW 更新，无 OOM、NaN、死锁；
2. 每步 `tokens_seen` 增量等于 `(4096-1) × 1 × 4 × world_size`；
3. rank 0 日志中的 loss 是跨 rank 均值，吞吐按最慢 rank 耗时计算；
4. 产出可打开的 profiler trace 与 operator table；
5. 根据实测决定继续 DDP、降低同步频率，或进入 ZeRO/FSDP，而不是先重写 Triton 内核。
