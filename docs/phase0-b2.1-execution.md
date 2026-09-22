# Phase 0B2.1 第一阶段执行说明

## 实际模型

六个实验共享以下设置：

| 项目 | 配置 |
|---|---:|
| hidden size | 256 |
| attention heads | 8 |
| FFN | 1024 |
| dropout | 0.1 |
| 记录长度 | 256 bytes |
| 目标与近期上下文 | 512 bytes |
| 逐记录输入长度 | 384 bytes |
| 参数量 | 约 6.78M |
| batch size | 16 |
| epochs | 6 |
| optimizer | AdamW |
| 初始学习率 | $3\times10^{-4}$ |
| 最低学习率 | $3\times10^{-5}$ |
| scheduler | cosine |
| Seed | 0 |

实际参数量低于早期计划中的 8M–10M 估计，但六个模型的参数差距小于 0.01%，因此不影响本阶段的结构归因。第一阶段不依靠扩大参数量获得结论。

## 参数量匹配

A 使用八层字节 Transformer，不包含集合交互层。

B–F 使用：

- 六层共享字节 Transformer；
- 两层集合 Transformer。

因此 A 与 B–F 的 Transformer 层参数基本一致。目标、近期上下文和记录均使用同一个字节编码器，避免增加独立语言编码器造成参数不匹配。

## A：Pointwise

每条待评分记录可以读取：

- 当前目标；
- 两条近期上下文；
- 自己的文本；
- 最多四条有界竞争记录；
- 合法元数据。

它不能在同一次前向中读取完整的 $S+1$ 条记录，也不能在记录之间执行集合注意力。

## B–E：完整集合

每条记录先独立编码，然后加入：

- 目标与近期上下文向量；
- 相对年龄 embedding；
- event index embedding；
- 新候选 embedding；
- 消息来源 embedding。

随后完整的 $S+1$ 条记录进入两层 Set Encoder，并联合输出效用。

## F：部分集合

F 与 E 参数量、训练目标完全相同，但每条记录在 Set Encoder 中最多只能关注四条记录：

- 自己；
- 最老记录；
- 当前新候选；
- 按位置距离最近的其他记录。

所有记录仍然产生预测，但不能建立任意全局竞争关系。这用于隔离完整集合交互的贡献，而不是简单减少输入记录数量。

## 类别不平衡处理

正式训练集中只有约 5.1% 的记录具有正效用。完全不加权的 MSE 很容易通过预测接近零获得较低损失。

因此所有 A–F 的回归项都统一使用正样本权重 8：

$$
L_{\mathrm{reg}}
=
\frac{\sum_i w_i(\hat u_i-u_i)^2}{\sum_i w_i},
\qquad
w_i=
\begin{cases}
8,&u_i\gt0,\\
1,&u_i=0.
\end{cases}
$$

六个实验使用相同权重，因此不会把类别处理差异误认为结构收益。离线报告中的 MSE 仍然使用不加权 MSE。

## 六个实验

| 实验 | 架构 | Regression | Ranking | Eviction |
|---|---|:---:|:---:|:---:|
| A | Pointwise | ✓ |  |  |
| B | Joint | ✓ |  |  |
| C | Joint | ✓ | ✓ |  |
| D | Joint | ✓ |  | ✓ |
| E | Joint | ✓ | ✓ | ✓ |
| F | Joint Partial | ✓ | ✓ | ✓ |

Ranking 系数为 0.5，Eviction 系数为 1.0。

## 评估

每个模型执行两类评估。

### 离线决策组

- 不加权 utility MSE；
- pairwise ranking accuracy；
- 最低效用选择准确率；
- 单步 regret；
- 吞吐和峰值显存。

### 模型自身连续 rollout

评估重新生成同一批测试 episode，但记忆状态由被评估模型自己的历史淘汰决定，不使用离线行为策略状态。

报告：

- 最终任务成功率；
- 必需记录召回率；
- 旧值误用率；
- 淘汰准确率；
- 平均 regret；
- FIFO 与 Future Oracle 同预算基线。

## 运行

```bash
scripts/run_b2_1_stage1_6gpu.sh full
scripts/status_b2_1_stage1.sh
```

六个实验分别使用物理 GPU 2–7。每个进程独立保存配置、checkpoint、TensorBoard、原始指标和记忆淘汰轨迹。

## 完成状态

2026-09-22，A–F 六个实验已经全部完成。训练中间产物保存在 L40 工作区：

```text
/myjfs/94f3304c-d49d-4e45-bd8c-69cea6ddfe0c/25212408112/small-agent-architecture-notes/
```

仓库保存可复核的聚合报告、完整指标表，以及每个实验的解析后配置和 `summary.json`：

```text
results/phase0_b21_stage1/
```

主结论：预注册主候选 E 未通过全部门槛，正式决定是停止扩展并诊断。诊断性消融中 C 最好，下一轮应重新预注册 C 的多种子验证，而不是把本轮事后选择当成已经通过。
