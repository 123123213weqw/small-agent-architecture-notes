# Phase 0A/0B：符号记忆选择验证

## 1. 目的

在接入语言模型之前分别回答两个问题：

- **0A 可行性**：相同容量下，Future Oracle 是否稳定优于 FIFO？
- **0B 可学习性**：只读取决策时信息的小型效用预测器，能否恢复一部分 Oracle 优势？

这不是模型能力结论。它只验证淘汰问题是否存在，以及当前特征中是否包含可学习信号。

## 2. 符号 Reader

每个 episode 是一串不可变事件记录。最终任务由一个或多个依赖集合组成；只有依赖集合中的记录全部保留，相关子目标才成功。符号损失为失败子目标比例：

$$
L(Y_t\mid\mathcal R_t)
=
\frac{1}{K}\sum_{k=1}^{K}
\mathbf 1[C_k\nsubseteq\mathcal R_t].
$$

决策时对每条竞争记录执行一次 mask，得到：

$$
u_{t,i}
=
L(Y_t\mid\mathcal R_t\setminus\{m_i\})
-L(Y_t\mid\mathcal R_t).
$$

未来到达的记录只参与标签计算，不进入预测器特征。这个实现对应单次淘汰的反事实监督，不声称求出了整个 episode 的全局最优策略。

## 3. 六类任务

1. `delayed_query`：早先事实经过干扰后被查询；其中一半不提前公开具体查询键，用来测量不可预测未来造成的上限。
2. `state_overwrite`：同一状态多次覆盖，只有最终版本有效。
3. `long_instruction`：长期输出约束与答案记录必须同时保留。
4. `completed_intermediate`：已消费的中间结果应淘汰，未完成结果应保留。
5. `unresolved_subgoal`：子目标与后续证据必须组合保留。
6. `relation_chain`：两条边联合构成最终关系，检查非单条事实依赖。

关键记录的位置覆盖早期、中期和近期，避免把 FIFO 人为设置为必败策略。

## 4. 对比策略

- `Random`：随机淘汰；
- `FIFO`：淘汰最早进入的记录；
- `Future Oracle`：淘汰真实反事实效用最低的记录；
- `Predicted Utility`：使用两层 MLP 回归反事实效用，并淘汰预测值最低的记录。

训练轨迹由 FIFO、Random 和 Oracle 混合采集，避免预测器只看到单一策略诱导的状态。正效用标签进行有限过采样，但回归目标仍是原始反事实效用。

## 5. 数据划分

- 训练长度：32、48、64；
- 同分布测试长度：48、64；
- 长度外推测试：96、128；
- 长期容量：4、8、12；
- 测试 episode 与训练 episode 的随机实体、值和事件流完全分离。

## 6. 预注册判据

Phase 0A 通过需要同时满足：

1. 聚合成功率的 Oracle−FIFO 差距至少为 0.10；
2. 至少四类任务上的差距不低于 0.05。

Phase 0B 使用恢复比例：

$$
\operatorname{RecoveredGap}
=
\frac{S_{\mathrm{pred}}-S_{\mathrm{FIFO}}}
{S_{\mathrm{oracle}}-S_{\mathrm{FIFO}}}.
$$

第一阶段要求长度外推测试中的恢复比例至少为 50%。

## 7. 指标和输出

- episode 成功率；
- 子目标正确率；
- 必需记录召回率；
- 淘汰选择正确率；
- 单步淘汰 regret；
- 状态旧值残留率；
- 各任务族的 Oracle−FIFO 差距；
- Predicted Utility 恢复的 Oracle 差距。

运行后生成：

- `RESULTS.md`：人工可读结论；
- `summary.json`：聚合结果与实验元数据；
- `episode_metrics.csv`：逐 episode 原始结果，默认不提交 Git。

## 8. 限制

符号实验有意省略语言理解、事件切分、表示学习、Reader 误差和训练非平稳性。特征中还包含已经结构化的 `consumed`、`unresolved` 和目标匹配信息，因此通过 0B 只说明效用信号在理想化记录中可学，不能证明语言模型能从原始 token 自动抽取这些信号。

下一阶段必须用 20M–50M Reader 替换符号 Reader，并增加无人工结构字段的 token 输入对照。

## 9. 第一轮结果

在 WZU V100 服务器上使用 10 个随机种子运行；每个种子包含 600 个训练 episode，以及各 240 个同分布和长度外推测试 episode。

| Split | FIFO | Oracle | Predicted | Oracle−FIFO | 恢复比例 |
|---|---:|---:|---:|---:|---:|
| 同分布 | 0.105 | 1.000 | 0.943 | $0.895\pm0.012$ | $93.6\%\pm1.8\%$ |
| 长度外推 | 0.052 | 1.000 | 0.927 | $0.948\pm0.014$ | $92.3\%\pm1.9\%$ |

Phase 0A 和 0B 均达到预注册阈值。最难的子任务是没有提前公开查询键的 `delayed_query`：预测策略在同分布和长度外推测试上分别达到 0.660 和 0.560，而 Oracle 为 1.000。这符合预期，因为一部分未来需求在决策时不可识别。

完整聚合结果见 [`../results/phase0_ab/RESULTS.md`](../results/phase0_ab/RESULTS.md)，机器可读结果见 [`../results/phase0_ab/summary.json`](../results/phase0_ab/summary.json)。
