# M4：L40 单节点 DDP 与 Profiler 结果

日期：2026-09-24。模型为 1,005,213,696 参数的 Base-1B-v1，4096 上下文，每卡 microbatch 1、累积 4。输入仍是内部冒烟语料，本轮只验证工程正确性与性能，不能作为模型能力结论。

## 结果

稳定区间取 step 4–12，吞吐按最慢 rank 的耗时计算：

| GPU | 每步有效 token | 中位 token/s | 相对单卡加速 | 扩展效率 | 最大 allocated |
|---:|---:|---:|---:|---:|---:|
| 1 | 16,380 | 8,405 | 1.00× | 100.0% | 19.24 GiB |
| 2 | 32,760 | 12,404 | 1.48× | 73.8% | 22.99 GiB |
| 4 | 65,520 | 16,753 | 1.99× | 49.8% | 22.99 GiB |
| 8 | 131,040 | 31,939 | 3.80× | 47.5% | 22.99 GiB |

四组都完成 12 次真实前反向和 AdamW 更新，没有 OOM、NaN 或 NCCL 死锁。`tokens_seen` 的每步增量严格等于：

\[
(4096-1)\times 1\times 4\times\text{world size}.
\]

不同 GPU 数处理的全局 token 数不同，学习率也按全局 token 前进，因此最终 validation loss **不能横向用于模型优劣比较**。

## Profiler 定位

8 卡另跑了一次独立 profiler，采集 3 个优化器步。rank 0 operator table 的 self CUDA time：

- NCCL all-reduce：7.525 秒，53.10%，291 次调用；
- `aten::mm`：2.829 秒，19.96%；
- AdamW：0.165 秒，1.17%；
- GDN 的 `ChunkGatedDeltaRuleFunction` 前向和反向各自没有接近 NCCL 的占比；
- Flash Attention 前后向也不是最大项。

Profiler 的 CUDA kernel 时间可能重叠，因此 53.10% 不能直接解释为“删掉通信就一定加速 53%”。但结合 8 卡只有 3.80× 单卡吞吐，可以明确把**梯度通信**列为第一优化对象，而不是先重写 GDN Triton 内核。

服务器保留一个 356,654,016 字节的 trace：

```text
/data1/wangyue/experiments/small-agent-base-1b-v1-engineering/runs/
  m4_d8_profiler_20260924_v1/profiler/
  node5_1223092.1790232093152549164.pt.trace.json
```

SHA-256：`3a5468af20718b188cff1ea5fedc5e091fef21285f7e9fa7f17c13395816e926`。仓库只保存小型 operator table、运行清单和指标，不复制 340 MiB trace。

## 拓扑解释

GPU 0–3 位于 NUMA 0，GPU 4–7 位于 NUMA 1；组内是 PIX，跨组是 SYS，`nvidia-smi topo -m` 没有显示 NVLink。约 10 亿个 FP32 参数对应很大的梯度同步量，当前 DDP 通信效率不高是符合实测拓扑的，不是推测 GDN 数学本身慢。

## 工程结论

1. 正式训练器已经从单卡扩展到 1/2/4/8 卡 DDP，并正确处理分片采样、`no_sync`、全局 token 学习率、跨 rank 指标和分片验证。
2. 8 卡当前约 31.9k token/s，能用，但只有 47.5% 线性扩展效率。
3. 下一组实验应先比较 DDP bucket、BF16 通信 hook、累积步数和 FSDP/ZeRO；每种方案同时检查吞吐、数值有限性和短程 loss 偏差。
4. M4 暂时禁止多卡 checkpoint。下一阶段必须保存每个 rank 的 sampler/RNG，并验证 DDP 中断恢复，不能只保存 rank 0 状态。

机器上的五个 M4 run 均位于统一的 `.../runs/` 目录；没有生成大型模型 checkpoint。结构化汇总见 `results/infra_m4_ddp_l40_v1/summary.json`。
