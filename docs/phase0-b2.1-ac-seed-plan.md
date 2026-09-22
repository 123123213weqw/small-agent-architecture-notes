# Phase 0B2.1 A/C 多种子确认计划

## 1. 状态与目的

C 是第一阶段六个 Seed 0 实验完成后，根据诊断性消融选出的候选。因此“选择 C”本身是事后决定，不能用 Seed 0 结果直接宣称通过。

本轮在查看 Seed 1、2 结果之前冻结，只回答：

> 在训练数据和评估 episode 完全不变时，C 相对 A 的收益能否跨模型初始化与训练顺序稳定复现？

## 2. 固定模型

- A：8 层逐记录 Byte Transformer，只有加权 Regression；
- C：6 层共享 Byte Transformer + 2 层完整 Set Transformer，使用加权 Regression + 0.5 Ranking；
- 不修改 hidden size、层数、margin、正样本权重、学习率、训练步数或任何输入字段；
- 不加入 Eviction loss；
- A 与 C 参数量继续保持在 0.01% 以内。

## 3. 分离三类随机种子

配置显式拆成：

```text
model_seed       模型初始化、dropout 和 DataLoader 顺序
data_seed        已物化训练、验证和离线测试数据
evaluation_seed  连续 rollout episode
```

本轮固定：

```text
data_seed = 0
evaluation_seed = 0
model_seed ∈ {0, 1, 2}
```

Seed 0 复用第一阶段已经完成的 A/C 结果；只新增训练 A/C 的 Seed 1、2。这样测试的是优化随机性，不把数据变化混入同一结论。

## 4. 实验矩阵

| GPU | 实验 | model seed | data seed | evaluation seed |
|---:|---|---:|---:|---:|
| 2 | A | 1 | 0 | 0 |
| 3 | A | 2 | 0 | 0 |
| 4 | C | 1 | 0 | 0 |
| 5 | C | 2 | 0 | 0 |

所有实验仍然使用 36,000 个训练决策组、6 epochs、batch size 16、AdamW 和相同的四个测试集。

## 5. 主要指标

主要比较：

1. 组合泛化 rollout 成功率；
2. 四个测试集平均 eviction regret；
3. 组合测试六类任务的成功率；
4. ID、长度外推和语义压力测试成功率；
5. 参数量、吞吐和显存。

每个 Seed 使用配对的相同评估 episode，先计算同 Seed 的 $C-A$，再报告三种子的均值、标准差和逐 Seed 结果。

## 6. 冻结判定门槛

C 只有同时满足以下条件，才通过 B2.1 多种子确认：

1. 三种子平均组合成功率相对 A 提高至少 5 个百分点；
2. 三种子四测试集平均 regret 相对 A 降低至少 20%；
3. 六类组合任务中至少四类的三种子平均成功率不下降；
4. 至少两个 Seed 的组合成功率和平均 regret 同时优于对应 A；
5. 不允许任何 Seed 的组合成功率低于对应 A，且不允许任何 Seed 的平均 regret 相对 A 增加超过 10%；
6. 参数量差距不超过 1%，C 的吞吐不低于 A 的 50%，峰值显存不超过 A 的 1.25 倍。

如果失败，不调参重跑来追门槛，直接报告失败模式。如果通过，只能说明当前合成任务中的优化稳定性通过；语言表示和真实 Reader 仍需后续阶段验证。

## 7. 交付物

- 四份解析后配置和训练摘要；
- 每个 Seed 的离线指标与连续 rollout 指标；
- A/C 配对差值和三种子聚合；
- TensorBoard 与淘汰轨迹；
- 中文结果报告和继续/停止决定。
