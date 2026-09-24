# M4-B：BF16 梯度通信实验

日期：2026-09-24。目标是验证 M4 Profiler 指出的 NCCL 瓶颈能否通过压缩 DDP 通信缓解。模型计算、参数和 AdamW 状态仍保持原设置；只把 all-reduce 时的梯度 bucket 临时转换为 BF16。

## 方法

普通 DDP 对每个 rank 的 FP32 梯度执行：

\[
\bar g=\frac{1}{W}\sum_{r=1}^{W}g_r.
\]

BF16 communication hook 执行近似计算：

\[
\tilde g_r=\operatorname{BF16}(g_r/W),\qquad
\hat g=\operatorname{FP32}\left(\sum_{r=1}^{W}\tilde g_r\right).
\]

这样通信 payload 约从 32 bit/元素降为 16 bit/元素，all-reduce 后再转回参数梯度的原始类型。代价是引入 BF16 舍入误差，所以吞吐和数值偏差必须同时测量。

配置：`configs/runs/base_1b_m4b_ddp_bf16_hook_l40_v1.json`。除通信 hook 外，与 M4 的 8 卡基线相同：

- 8 张 L40；
- 1,005,213,696 参数；
- 4096 上下文；
- 每卡 microbatch 1；
- 梯度累积 4；
- DDP bucket 25 MiB；
- 同一 seed、数据顺序和 12 个优化器步。

## 吞吐结果

稳定区间为 step 4–12：

| 方案 | 中位 token/s | P10–P90 | 相对单卡加速 | 扩展效率 |
|---|---:|---:|---:|---:|
| FP32 all-reduce | 31,939 | 30,617–32,078 | 3.80× | 47.5% |
| BF16 all-reduce | **45,762** | **45,186–45,838** | **5.44×** | **68.1%** |

BF16 通信相对 FP32 通信提升 **43.3%**，超过预先提出的 40k token/s 目标。峰值 allocated 22.99 GiB，与基线基本相同；优化解决的是带宽，不是模型显存。

## 短程数值比较

两组逐 step 比较：

- 最大训练 loss 绝对差：0.001665；
- 平均训练 loss 绝对差：0.000499；
- 最大梯度范数绝对差：0.00662；
- 最终 validation loss：FP32 8.216950，BF16 8.216492，差 -0.000458。

这说明 12 步内没有明显数值异常，但**不能证明长期收敛完全等价**。当前冒烟语料很小，8 卡约每 9 个优化器步就进入下一 epoch，继续在它上面跑很久只能证明重复小数据上的稳定性。

## 结论

BF16 communication hook 是当前 8 卡训练的默认候选：它直接命中了 M4 中占 53.1% self CUDA time 的 NCCL all-reduce，并把实测吞吐从约 31.9k 提升到 45.8k token/s。

正式预训练前仍需在更大的冻结数据集上做较长 A/B，检查 validation 曲线、梯度分布和 checkpoint 恢复。现阶段不需要先重写 GDN Triton 内核。

服务器运行目录：

```text
/data1/wangyue/experiments/small-agent-base-1b-v1-engineering/runs/
  m4b_d8_bf16hook_20260924_v1/
```

该目录约 24 KiB，没有保存大型 checkpoint；实验结束后 8 张 GPU 已释放。结构化结果见 `results/infra_m4b_bf16_comm_v1/summary.json`。
