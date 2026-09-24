# 2B：分片读取性能与预取决策

2B 在 2A 的确定性数据日程上增加可选的 DataLoader worker/pinned-memory 预取及可观测性，**不改变分片格式、样本顺序或 checkpoint 已提交游标的定义**。

## 配置与指标

Run config 可以加入：

```json
"data_pipeline": {
  "num_workers": 0,
  "prefetch_factor": 2,
  "persistent_workers": false,
  "pin_memory": true,
  "measure": true,
  "trace_sample_refs": true
}
```

- `num_workers=0` 是默认的简单路径；`num_workers>0` 才启用 `prefetch_factor` 和持久 worker。
- `measure=true` 启用计时：`data_wait_ms` 记录训练主进程等待下一批的 CPU 墙钟时间；`h2d_copy_ms` 使用 CUDA event 记录三个模型输入的设备复制时间。指标跨 DDP rank 取最大值，代表可能拖慢同步的最慢 rank。
- `data_wait_fraction` 和 `h2d_copy_fraction` 相对于同一日志窗口内的训练耗时。它们是定位指标，不应解释为严格可相加的算子时间分解。
- `trace_sample_refs=true` 只用于短跑，记录 rank 0 每个日志窗口实际消费的 sample ref；正式长跑应关闭，避免扩大日志。
- TensorBoard 增加 `data/wait_ms`、`data/h2d_copy_ms` 和 `data/wait_fraction`。

多 worker 的 sampler 可能比训练前进数批，但 checkpoint 仍只保存 `committed_global_position`。恢复时新建 loader，从已完成 optimizer step 的位置重新生成未提交样本。测试覆盖 0/2 workers 序列一致、2-rank worker 预取和 checkpoint 边界恢复。

## 8 张 L40 的 A/B

使用同一份 2A 百万 token 冒烟分片、相同模型和种子，分别运行 12 步；汇总时排除前 2 步。配置：

- `configs/runs/base_1b_sharded_2b_profile_workers0_l40_v1.json`
- `configs/runs/base_1b_sharded_2b_profile_workers2_l40_v1.json`

可重复生成对比报告：

```bash
python scripts/report_sharded_data_profile.py \
  --baseline /path/to/runs/sharded_2b_workers0_profile_v1 \
  --candidate /path/to/runs/sharded_2b_workers2_profile_v1 \
  --warmup-steps 2 \
  --output /path/to/runs/sharded_2b_profile_comparison_v1.json
```

| 指标（稳态中位数） | workers=0 | workers=2 |
|---|---:|---:|
| token/s | 50,749 | 50,544 |
| data wait / step | 7.02 ms | 3.39 ms |
| data wait 占比 | 0.271% | 0.129% |
| H2D / step | 0.130 ms | 0.150 ms |
| 峰值已分配显存 | 34.99 GiB | 34.99 GiB |

rank 0 的逐步 sample ref 轨迹完全一致；两组 loss 的最大绝对差约 0.00137。2 workers 减少了等数据时间，但没有改善总吞吐（比值约 0.996，属于短跑波动范围）。

另用 `configs/runs/base_1b_sharded_2b_workers2_resume_l40_v1.json` 做 8 卡 2-worker 断点实验：第 1 步保存 checkpoint，重启后恢复完成第 2 步。处理 262,080 token；rank 0 实际消费序列与 0-worker 同种子运行的前两步一致，8 个 rank 的提交位置均为 64，下一条 shard/sample/token offset 均通过日程校验。验收报告为 `runs/sharded_2b_workers2_resume_v1/sharded_2b_validation.json`。约 12 GB 临时 checkpoint 验收后已删除，该 canary 不再可继续恢复。

**当前选择：保留 `num_workers=0` 为默认，不实现 CUDA 双缓冲。** H2D 占比约万分之一，增加 CUDA stream、record_stream 和额外缓冲会增加维护成本，却没有已观察到的瓶颈可解决。`num_workers=2` 作为可选配置保留；真实 5000 万～5 亿 token、多领域数据准备好后重新测量，再决定是否启用。

此结果仅针对 2.3 MB 冒烟分片和当前机器缓存状态，不应外推为所有未来数据规模的吞吐结论。未来重新评估时先复用相同的轨迹校验，再看 data wait 是否持续超过 5%。
