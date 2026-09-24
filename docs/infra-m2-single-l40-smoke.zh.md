# Infra M2：单张 L40 的 1B Trainer 短测

## 定位

这是内部工程短测，不是正式预训练，也不用于判断最终模型能力。输入仍是约 114 万 token 的 P0 冒烟语料，训练过程中会重复读取。

## 实现

- `IndexedTokenDataset`：校验数据 SHA、文档 span 和 EOS，训练窗口相邻重叠一个 token；
- `StatefulDistributedSampler`：确定性 epoch 顺序和可恢复 cursor；
- `factory.py`：训练器与 Qwen3Next/GDN 结构解耦；
- `TokenLRScheduler`：按照累计 causal target token 做 warmup + cosine；
- `validation.py`：按照有效 target token 加权验证 loss，PAD 标签为 `-100`；
- `trainer.py`：单卡 BF16 autocast、FP32 权重、梯度累积、裁剪、独立验证、JSONL 和 TensorBoard。

## 运行配置

| 字段 | 数值 |
|---|---:|
| 参数量 | 1,005,213,696 |
| 上下文 | 4096 |
| micro batch | 1 |
| 梯度累积 | 4 |
| optimizer steps | 100 |
| 训练 target token | 1,638,000 |
| GPU | 单张 NVIDIA L40 |
| gradient checkpointing | 开启 |

## 结果

| 指标 | 起点 | step 50 | step 100 |
|---|---:|---:|---:|
| train loss | 10.7108（step 1） | 7.9892 | 6.9933 |
| validation loss | 10.7228 | 7.1626 | 6.7257 |

稳态吞吐（去掉首次 Triton/算子预热步）中位数为 **8,138 token/s**，范围约为 7,515～8,508 token/s。峰值 allocated 显存 19.25 GiB，峰值 reserved 显存 20.35 GiB。

训练跨过 epoch 边界后继续正常取样；最终 sampler 位于 epoch 1、cursor 129。训练和验证 loss 均明显下降，没有 NaN/Inf，TensorBoard event 文件和 `status.json=completed` 均正常生成。

## 能说明什么

已经验证：新数据层、模型工厂、token 学习率、单卡训练、独立验证、TensorBoard 和状态记录可以在真实 1B GDN 模型上联合工作。

还没有验证：完整 checkpoint、精确中断恢复、DDP 训练、NCCL 性能、正式大数据吞吐和最终能力。

详细机器可读结果见 `results/infra_m2_single_l40_v1/summary.json`。
