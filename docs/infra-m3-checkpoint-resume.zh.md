# Infra M3：完整 checkpoint 与恢复一致性

## 保存内容

完整恢复点采用原子目录写入，包含：

- `model.safetensors`：模型参数；
- `optimizer.pt`：AdamW 状态；
- `runtime.pt`：PyTorch CPU/CUDA RNG；
- `scheduler.json`：累计 token 与 LR 日程；
- `sampler.json`：epoch、rank 和 cursor；
- `manifest.json`：配置、数据、模型 SHA-256 及全部产物哈希；
- `latest.json`：最近完整 checkpoint 的原子指针。

恢复时先校验文件大小和 SHA-256，再拒绝 tokenizer/数据/模型、seed、上下文、batch、梯度累积或 gradient-checkpointing 漂移。

## 1B GDN 实验方法

控制组在同一进程中训练 20 步，并在 step 10 保存完整 checkpoint 后继续。恢复组从控制组 **同一个 step-10 checkpoint** 启动新进程，继续训练 step 11～20。因此两组在分叉点的模型、optimizer、RNG、scheduler 和 sampler 文件完全相同。

step-10 checkpoint 为 12,062,929,538 bytes，保存并计算 SHA-256 用时 23.286 秒。其中模型约 4.02 GB，optimizer 约 8.04 GB。

## 结果

| 项目 | 结果 |
|---|---:|
| LR / tokens_seen / epoch / sampler cursor | 完全一致 |
| step 11～20 最大 loss 差 | 0.001054 |
| step 11～20 平均 loss 差 | 0.000428 |
| 最大 grad-norm 差 | 0.014404 |
| 最终 validation loss，连续 | 8.939317 |
| 最终 validation loss，恢复 | 8.940120 |
| validation loss 差 | 0.000803 |
| 最终参数相对 L2 差 | 0.000715 |
| 最终参数最大绝对差 | 0.001308 |

最终 RNG、scheduler 和 sampler 文件的 SHA-256 相同，模型和 optimizer 不满足 bitwise equality。独立的 CPU 确定性模型测试可做到中断恢复后参数逐 bit 相同，说明 checkpoint 状态本身保存和恢复完整；1B 路径的微小差异来自当前 CUDA/FLA/Triton GDN 计算的非确定性。

因此 M3 的结论是：**逻辑恢复通过、1B 数值轨迹等价，但不宣称 GPU bitwise reproducibility**。正式训练应以 loss 容差和状态一致性作为恢复门槛，同时保留一个小型确定性测试作为 checkpoint 回归测试。

机器可读结果见 `results/infra_m3_checkpoint_resume_v1/summary.json`。
