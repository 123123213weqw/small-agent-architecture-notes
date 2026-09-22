# Phase 0B2.1 第一阶段计划：完整竞争集合联合排序

## 1. 第一阶段只回答一个问题

在不引入语言模型预训练、不扩大到 Reader 的前提下，验证：

> 同时读取完整竞争集合并直接训练淘汰排序，是否比当前逐记录 MSE 模型更有效？

第一阶段是结构消融，不是最终 B2.1 结论。所有实验使用同一份数据、同一个随机种子、相近参数量和相同训练步数，避免同时改变数据、模型规模和训练目标。

## 2. 冻结内容

第一阶段不修改以下定义：

$$
u_{t,i}
=
L(Y_t\mid B_t,\mathcal R_t\setminus\lbrace m_i\rbrace)
-
L(Y_t\mid B_t,\mathcal R_t).
$$

同时冻结：

- 每次一个候选进入；
- 容量超限时只淘汰一条；
- 未来只生成标签，不进入预测器；
- 所有策略使用相同容量；
- 事件记录保持不可变；
- 最终评估执行连续淘汰 rollout。

## 3. 数据单位

每条训练数据是一整个淘汰决策，而不是一条独立记录：

```text
goal
recent_context
records[0:S+1]
event_index[0:S+1]
relative_age[0:S+1]
candidate_index
utilities[0:S+1]
```

允许输入的元数据：

- event index；
- 相对年龄；
- 是否是本轮新候选；
- 消息来源或工具来源；
- 已经观察到的工具调用 ID。

禁止输入：

- `important`；
- `consumed`；
- `unresolved`；
- 未来是否使用；
- 是否属于最终答案；
- 任何由真实效用标签直接推导的字段。

## 4. 第一阶段数据规模

| 数据 | 数量或范围 |
|---|---:|
| 训练 episode | 3,000 |
| 验证 episode | 400 |
| 每个测试集 episode | 600 |
| 每个 episode 最多采样决策组 | 12 |
| 训练长度 | 32、48、64 |
| 长度外推 | 96、128 |
| 容量 | 4、8、12 |
| hard negatives | 0、4、8、14 |
| 任务 | 6类均衡 |
| 随机种子 | 0 |

所有划分按照完整 episode 隔离。训练、验证和测试之间不得共享实体 ID、事件流或决策状态。

查询可见性单独记录：

- `announced_query`：目标提前给出；
- `hidden_query`：未来才出现目标。

两类结果分别报告，不合并掩盖不可预测性。

## 5. 模板划分

第一阶段建立四个中文模板族：

- A、B、C：用于训练；
- D：组合泛化测试。

D 只能重新组合训练中已经出现过的词汇与语义片段，不引入完全未见的同义词。这样主要测试组合和排序能力，而不是测试模型是否已经具备语言预训练知识。

完全未见同义词的语义压力测试保留，但不作为第一阶段阻塞条件。

## 6. 联合排序模型

共享记录编码器：

$$
h_i=\mathrm{Encoder}(m_i),
\qquad
h_g=\mathrm{Encoder}(g_t,B_t).
$$

融合当前目标和合法元数据：

$$
z_i
=
W_mh_i+W_gh_g
+E_{\mathrm{age}(i)}
+E_{\mathrm{candidate}(i)}.
$$

完整集合交互：

$$
(\tilde z_1,\ldots,\tilde z_{S+1})
=
\mathrm{SetEncoder}(z_1,\ldots,z_{S+1}).
$$

联合输出：

$$
\hat u_i
=
w^\top\mathrm{GELU}(W\tilde z_i).
$$

建议配置：

| 模块 | 配置 |
|---|---:|
| 字节级记录 Encoder | 6层 |
| hidden size | 256 |
| heads | 8 |
| FFN | 1024 |
| 单记录上限 | 256 bytes |
| 目标与短期上下文 | 512 bytes |
| Set Encoder | 2层 |
| 最大竞争记录 | 13 |
| 目标总参数 | 8M–10M |

每条记录单独编码，因此不会把完整集合压进一个 384-byte 字符串。

## 7. 三种训练目标

### 7.1 回归

$$
\mathcal L_{\mathrm{reg}}
=
\frac{1}{S+1}\sum_i(\hat u_i-u_i)^2.
$$

### 7.2 集合内排序

$$
\mathcal L_{\mathrm{rank}}
=
\sum_{u_i\gt u_j}
|u_i-u_j|
\max(0,\delta-\hat u_i+\hat u_j).
$$

### 7.3 淘汰分类

令最低真实效用记录组成集合 $\mathcal A$，并列最低者共享目标概率：

$$
q_i
=
\begin{cases}
1/|\mathcal A|,&i\in\mathcal A,\\
0,&\text{其他}.
\end{cases}
$$

$$
p_i=\mathrm{softmax}(-\hat u_i/\tau),
\qquad
\mathcal L_{\mathrm{evict}}=-\sum_iq_i\log p_i.
$$

完整目标：

$$
\mathcal L
=
\mathcal L_{\mathrm{reg}}
+0.5\mathcal L_{\mathrm{rank}}
+\mathcal L_{\mathrm{evict}}.
$$

## 8. 六张 L40 的并行实验矩阵

六张卡不做一个小模型的数据并行，而是同时运行六个严格对照实验。

| 物理 GPU | 实验 | 完整集合 | 损失 | 目的 |
|---:|---|:---:|---|---|
| 2 | A：旧逐记录模型 | 否 | MSE | 复现当前基线 |
| 3 | B：Joint-MSE | 是 | Regression | 测集合输入本身 |
| 4 | C：Joint-Rank | 是 | Regression + Ranking | 测排序监督 |
| 5 | D：Joint-Evict | 是 | Regression + Eviction | 测直接淘汰监督 |
| 6 | E：Joint-Full | 是 | Regression + Ranking + Eviction | 主候选方案 |
| 7 | F：Joint-Partial | 只看4条 | 与 E 相同 | 隔离完整集合的贡献 |

六个实验：

- 使用同一份不可变数据 manifest；
- 使用相同 Seed 0；
- 使用相同训练 token 数和优化器步数；
- 参数量控制在主干的 $\pm10\%$；
- 独立保存配置、日志和 checkpoint。

目录：

```text
data/b2_1_stage1/
runs/b2_1_stage1/A_pointwise/
runs/b2_1_stage1/B_joint_mse/
runs/b2_1_stage1/C_joint_rank/
runs/b2_1_stage1/D_joint_evict/
runs/b2_1_stage1/E_joint_full/
runs/b2_1_stage1/F_joint_partial/
checkpoints/b2_1_stage1/
```

## 9. 评估

离线指标：

- utility MSE；
- pairwise ranking accuracy；
- 最低效用选择准确率；
- 单步淘汰 regret；
- 按容量和 hard-negative 数量分层的指标。

连续 rollout 指标：

- 最终任务成功率；
- 必需记录召回率；
- 旧值误用率；
- Oracle−FIFO 差距恢复比例；
- 每次决策延迟；
- 峰值显存；
- 每秒处理的决策组数。

必须按以下维度拆分报告：

- 六类任务；
- announced/hidden query；
- 容量 4/8/12；
- hard negatives 0/4/8/14；
- ID、组合泛化、长度外推、语义压力测试。

## 10. 第一阶段判定

第一阶段不是最终通过判定，而是选择进入多种子验证的结构。

E 相对 A 至少满足：

1. 组合泛化成功率提高至少 5 个百分点；
2. 平均淘汰 regret 降低至少 20%；
3. 六类任务中至少四类不下降；
4. F 明显弱于 E，证明完整集合确实有贡献；
5. 推理延迟和显存没有出现不可接受增长。

如果 E 没有超过 A，不扩大参数、不进入多种子实验，先检查集合编码、标签和损失。

如果 E 通过，从 B、C、D 的结果判断收益主要来自集合输入、排序损失还是淘汰损失，再冻结第二阶段配置。

## 11. 第一阶段交付物

1. 固定的数据生成配置和 manifest；
2. 无未来泄漏与 episode 隔离测试；
3. 六份可复现实验配置；
4. 六卡启动与状态汇总脚本；
5. 原始逐决策和逐 episode 指标；
6. 中文实验报告；
7. 第一阶段继续或停止决定。

