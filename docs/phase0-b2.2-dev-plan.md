# Phase 0B2.2 C-Robust Dev 实验

## 目标

只改变关系文本的训练与验证分布，检验组合式模板和实体位置扰动能否修复
`C Seed 1` 的模板捷径。模型、损失、训练规模、测试集和随机种子均不改变。

## 唯一训练变量

关系边由独立的前缀、关系子句和后缀组合生成。训练与验证使用完全分离的语言构件，
必要边与干扰边共享同一渲染规则。其他五类任务保持 B2.1 不变。

## Checkpoint 选择

每个 epoch 在 66 个未见验证关系 episode 上执行 on-policy rollout。优先选择关系链
成功率最高的 checkpoint；成功率相同时选择 validation total 更低者。只保存一个
最佳 checkpoint。

## 冻结项

- `model_seed=1`、`data_seed=0`、`evaluation_seed=0`；
- 6 层 Byte Encoder、2 层 Set Transformer、6,780,417 参数；
- $L=L_{reg}+0.5L_{rank}$；
- 36,000 train decisions、4,800 validation decisions、6 epochs；
- 四个原有测试 split 逐字节不变。

## Dev 门槛

- `test_composition/relation_chain` 成功率至少 80%；
- required recall 至少 90%；
- D 关系链 offline pairwise accuracy 至少 95%；
- 其他五类组合任务平均下降不超过 2pp，任何一类下降不超过 5pp。

本轮只决定是否值得冻结方案后使用全新 Seed 3/4/5 做正式确认，不作为最终结论。
