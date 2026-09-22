# Phase 0B2.2 C-Robust Dev 结果

## 实验状态

- 模型：C，6,780,417 参数；
- model seed：1；data/evaluation seed：0；
- 训练：36,000 decisions，验证：4,800 decisions，6 epochs；
- 唯一训练数据变化：关系边使用前缀、关系子句、后缀的组合语法；
- 验证关系语言构件与训练构件完全分离；
- 四个冻结测试 JSONL 与 B2.1 的 SHA-256 逐字节一致；
- checkpoint 依据未见验证模板上的 on-policy 关系链成功率选择，平局时取 validation total 更低者。

## Checkpoint

最佳 checkpoint 为 epoch 5：

| 指标 | 数值 |
|---|---:|
| 未见验证关系 rollout success | 100% |
| 未见验证关系 required recall | 100% |
| validation total | 0.004440 |

Epoch 3–6 的未见验证关系 rollout success 都为 100%，不是单个 epoch 的偶然峰值。

## 与原 Seed 1 比较

| `test_composition` 任务 | B2.1 Relation Data | B2.2 Robust Dev | 增益 |
|---|---:|---:|---:|
| delayed_query | 100% | 100% | 0pp |
| state_overwrite | 100% | 100% | 0pp |
| long_instruction | 87% | 100% | +13pp |
| completed_intermediate | 100% | 100% | 0pp |
| unresolved_subgoal | 100% | 100% | 0pp |
| relation_chain | 18% | 100% | +82pp |

D 关系链 required recall 从 42% 提升到 100%。该任务冻结 offline 数据上的
pairwise accuracy 从 77.0% 提升到 100%，argmin accuracy 与 regret 分别为 100%
和 0。

## 四测试集完整 rollout

| split | success | required recall | eviction regret |
|---|---:|---:|---:|
| test_id | 100% | 100% | 0 |
| test_composition | 100% | 100% | 0 |
| test_length | 100% | 100% | 0 |
| test_semantic_stress | 86% | 86.3% | 0.002898 |

Seed 1 原 B2.1 Relation Data 的 semantic stress 总成功率约为 76.8%，因此本轮没有
以修复 D 为代价破坏 E。

## 前缀压力测试

在同一批100个 D 关系链episode上，只改变edge句式：原句、删除前缀、等字节中文
前缀、ASCII前缀、关系后缀及删除冒号共7种变体，全部达到：

```text
success = 100%
required recall = 100%
decision accuracy = 100%
```

原 Seed 1 在这些变体上只有18%–25%，删除前缀或移到句尾才达到100%。这说明本轮
确实消除了已诊断的前缀/实体位置捷径，而不只是记住D测试句。

## 冻结门槛

| 条件 | 结果 | 结论 |
|---|---:|:---:|
| D关系链成功率至少80% | 100% | 通过 |
| required recall至少90% | 100% | 通过 |
| D关系链offline pairwise至少95% | 100% | 通过 |
| 最差前缀压力组至少70% | 100% | 通过 |
| 其他组合任务平均下降不超过2pp | 无下降 | 通过 |
| 任意其他组合任务下降不超过5pp | 最大下降0pp | 通过 |

**B2.2-Dev 通过。**

该结果支持“C 的主要失败来自文本模板/位置捷径，而不是联合效用机制本身”。但这是
针对已知失败 Seed 1 的开发实验，不能作为最终跨 Seed 结论。下一步应冻结全部代码、
数据和门槛，在全新 model seed 3/4/5 上做 B2.2-Confirm。
