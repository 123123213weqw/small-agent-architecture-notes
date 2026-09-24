# 统一训练入口与启动前检查

正式训练不再直接手写 `torchrun`。统一入口是：

```bash
python scripts/trainctl.py preflight configs/runs/RUN.json --run-id RUN_ID
python scripts/trainctl.py launch configs/runs/RUN.json --run-id RUN_ID --detach
```

## 为什么需要它

此前一次测试没有加载隔离的 FLA 依赖，`Qwen3Next` 静默退回 PyTorch 实现并发生 OOM。`trainctl.py` 采用 **fail-closed**：任何关键条件无法确认时都不启动训练，而不是猜测或退化运行。

## 启动前会检查什么

1. **配置一致性**：运行配置、模型配置和数据 manifest 存在，context、词表大小和 EOS ID 相容。
2. **数据完整性**：逐一核对 train/validation token 文件的字节数和 SHA-256。
3. **数据资格**：不可发布且 license gate 未通过的数据，只允许用于明确标记的 internal smoke run。
4. **GPU**：卡数与 `nproc_per_node` 相等、型号均为锁定的 L40，且所选卡没有计算进程。
5. **运行环境**：使用固定 Python 和隔离依赖目录，核对锁定包版本；确认 CUDA、Qwen3-Next fast path 和 fused RMSNorm 均启用。
6. **磁盘**：结合模型参数量和 checkpoint 保留数估算最低空间，并额外应用 50 GiB 下限。
7. **可复现性**：记录 source revision、源码树 SHA-256、配置 SHA-256、数据 manifest SHA-256 和环境锁 SHA-256。
8. **运行冲突**：新任务拒绝覆盖既有 run ID；恢复任务必须已有 manifest 和 latest checkpoint。

通过后，报告写入：

```text
<output_root>/.preflight/<run_id>.json
```

## L40 标准用法

先只检查，不占用 8 张卡训练：

```bash
cd /data1/wangyue/experiments/small-agent-base-1b-v1-engineering/code
python scripts/trainctl.py preflight \
  configs/runs/base_1b_m7_ddp_resume_equivalence_l40_v1.json \
  --run-id base_1b_preflight_v1
```

后台启动，SSH 断开后由 user systemd 继续维护：

```bash
python scripts/trainctl.py launch \
  configs/runs/base_1b_m7_ddp_resume_equivalence_l40_v1.json \
  --run-id base_1b_train_v1 \
  --detach \
  --save-final-checkpoint
```

查看日志：

```bash
journalctl --user -fu small-agent-base_1b_train_v1
```

恢复时必须复用原 run ID，并显式加入 `--resume`：

```bash
python scripts/trainctl.py launch configs/runs/RUN.json \
  --run-id RUN_ID --resume --detach
```

`--dry-run` 会完成全部预检并打印最终命令，但不启动。`--max-steps N` 只用于工程短测或受控覆盖，正式训练默认采用运行配置中的步数。
