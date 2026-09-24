# Base-1B-v1：L40 GPU 工程验收

日期：2026-09-24。仅测试结构与训练链路；使用许可未全部放行的 P0 小语料作工程输入，**不用于能力或正式预训练结论**。

## 设置

- 配置：[`configs/gdn/base_1b_v1.json`](../configs/gdn/base_1b_v1.json)，精确参数量 1,005,213,696，32 层 `[GDN×3,GQA]×8`。
- 脚本：[`experiments/base_1b_v1_engineering.py`](../experiments/base_1b_v1_engineering.py)；L40 项目局部快速 GDN/卷积依赖、BF16 autocast、FP32 权重、fused AdamW、梯度裁剪、4096 token、每卡 microbatch 1、梯度 checkpointing。
- 输入：既有 `p0_1m_token_stream_v1` train token 流；manifest SHA-256 `4b62a5cd59e4615de411854b4c997ad566bcc2ea66748e3ea2055cec6f1584d0`。
- L40 全部日志、代码快照、缓存和唯一保留 checkpoint：`/data1/wangyue/experiments/small-agent-base-1b-v1-engineering/`；依赖以符号链接复用先前工程目录，未改系统/原始 venv。
- 每进程的第一个优化器步包含 Triton 编译，**不能拿来算稳定吞吐**。以下吞吐只取预热后的单个步，尚不是长跑基准。

## 已完成的测试

| 测试 | 结果 |
| --- | --- |
| 单卡 128 token | 1 次真实优化器更新；loss/梯度有限，峰值显存分配 15.03 GiB。首步 236 s 含大量编译。 |
| 单卡 4096 token | 优化器更新、保存约 12.06 GB checkpoint、从新进程加载并继续到 step 3，均成功。恢复进程第 3 步 0.525 s / 4095 token，约 7796 token/s；峰值分配约 15.44 GiB。单卡 checkpoint 在验证后删除，保留日志。 |
| DDP 6 卡，累积 1 | 使用 GPU 0–4、7（GPU 5、6 当时有其他任务）。4096 token/卡，step 0→2 保存约 12.06 GB checkpoint，再从新进程恢复至 step 4；loss/梯度有限、未 OOM。预热后 step 2 为 2.50 s / 24570 token，约 9827 token/s；峰值分配约 19.19 GiB/卡。此版脚本日志里的 loss 仅是 rank 0 本地 loss。 |
| DDP 4 卡，累积 1 | 同 NUMA 域 GPU 0–3；预热后约 6569 token/s，未优于单卡。 |
| DDP 6 卡，累积 4 | 使用 DDP `no_sync`，每次参数更新处理 98280 token；预热后 step 2 为 3.92 s，约 **25092 token/s**；峰值分配约 22.74 GiB/卡。新版脚本记录跨 rank 平均 loss 和最大显存。 |

这里 `token/s` 只计实际 next-token 监督位置：每条 4096 长序列计 4095 个 token。以上每项只跑 1–4 次优化器更新，**绝不能外推为持续吞吐、稳定收敛或模型能力**。损失的下降来自极小且反复抽取的冒烟语料，不是泛化证据。

## 工程判断

1. 1B 配置已经能在 L40 上完成真实 BF16 前反向、AdamW 更新与单卡/6 卡断点恢复；4096 上下文没有 OOM 或 NaN。
2. **朴素 DDP 扩展效率很差**：6 卡累积 1 的总吞吐仅略高于单卡，同 NUMA 4 卡也未更快。可能是约 10 亿 FP32 梯度的同步通信及节点跨 NUMA 拓扑所致；这是诊断假设，不是已分离测量的因果结论。累积 4 后 6 卡约 25k token/s，说明减少同步频率有用，但训练配方仍需优化。
3. GPU 6 正在运行其他任务，因此本轮未强占它做 8 卡测试；**8 卡目标配置仍待实测**。现有 6 卡通过不等于 8 卡吞吐已验收。
4. 现有 Python 3.10 / Triton 3.1 低于 FLA 建议版本，虽然本次工程测试成功，正式长跑前仍需固定生产环境并做更长稳定性试验。
5. P0 数据 `license_gate.training_eligible=false`，且量级只有约 114 万 token。不能把本轮当成正式预训练开始。

## 下一个门槛

- 等 GPU 6 空闲后做 8 卡、4096、累积 4 的短跑；比较稳定吞吐与显存，不中断已有任务。
- 一次性等待脚本 `code/base_1b_v1_8gpu_when_idle.sh` 曾短暂排队，但用户决定“八张卡到时候再做”；已停止等待进程，状态 `run/ddp8_4096_accum4/status=deferred_by_user`。**当前没有自动启动 8 卡测试的后台任务**。脚本保留，未来需用户重新决定启动；该短跑不会保存第二份大型 checkpoint。
- 对训练数据完成许可、清洗、去重和验证集冻结；再用目标训练配方做更长的学习率/损失/数值稳定性试训。
