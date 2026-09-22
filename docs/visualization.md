# 实验可视化

项目使用两种互补的可视化：

1. TensorBoard：训练曲线、评估指标和少量记忆淘汰案例；
2. 记忆淘汰查看器：逐 episode、逐决策检查完整竞争集合。

## 输出文件

运行 Phase 0B2 后，输出目录增加：

```text
tensorboard/                 TensorBoard event 文件
eviction_trace.jsonl        可审计的逐决策原始记录
eviction_viewer.html        无外部依赖的交互查看器
```

默认每个测试集记录前三个 episode。可以通过下面的参数修改：

```bash
--trace-episodes 5
```

设为 `0` 时不记录具体淘汰案例。使用 `--no-tensorboard` 可以关闭 TensorBoard，但仍会生成 JSONL 和 HTML。

## 在 L40 上运行

```bash
cd /myjfs/94f3304c-d49d-4e45-bd8c-69cea6ddfe0c/25212408112/small-agent-architecture-notes

CUDA_VISIBLE_DEVICES=2 \
  /home/wangyue/.venvs/small-agent-l40/bin/python \
  experiments/phase0_b2_text.py \
  --output runs/b2_visualized \
  --trace-episodes 3
```

启动 TensorBoard：

```bash
PYTHON_BIN=/home/wangyue/.venvs/small-agent-l40/bin/python \
  scripts/start_tensorboard.sh runs 6006
```

服务只监听服务器的 `127.0.0.1`。在本机建立 SSH 隧道：

```bash
ssh -N -L 6006:127.0.0.1:6006 WZU_L40
```

随后访问：

```text
http://127.0.0.1:6006
```

TensorBoard 中主要查看：

- `training/*`：训练和验证损失、epoch 时间、学习率；
- `evaluation/*`：不同 split 和策略的成功率、recall、regret；
- `memory_eviction/*`：少量逐步淘汰表格；
- `run/config`：本次实验完整配置。

## 记忆淘汰查看器

把服务器生成的 `eviction_viewer.html` 同步到本机后直接用浏览器打开。它提供：

- split、episode 和决策步选择；
- 当前目标、任务、容量和最终成功状态；
- 每条竞争记录的真实效用与预测效用；
- 模型淘汰记录、Oracle 可淘汰记录和新候选标记；
- 单步 regret；
- 上一步和下一步浏览。

真实效用只用于实验审计和训练监督，不进入预测器输入。
