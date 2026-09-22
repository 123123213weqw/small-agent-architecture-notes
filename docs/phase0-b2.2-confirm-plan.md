# Phase 0B2.2 C-Robust Confirm 预注册

## 目的

B2.2-Dev 已针对已知失败的 Seed 1 通过。本轮不再修改数据、模型、损失、训练参数、
checkpoint 规则或测试集，只使用从未观察过的 model seed 3/4/5 验证结果是否稳定。

## 冻结内容

- 使用 B2.2-Dev 的同一份 `data/b2_2_robust`；
- 四个测试集 SHA-256 与 B2.1 完全一致；
- C 模型 6,780,417 参数；
- $L=L_{reg}+0.5L_{rank}$；
- 36,000/4,800 train/validation decisions，6 epochs；
- 每 epoch 在66个未见语言构件的验证关系 episode 上做 on-policy rollout；
- checkpoint 首先最大化验证关系成功率，平局时最小化 validation total；
- data/evaluation seed 固定为0，model seed依次为3、4、5。

看到任意确认结果后，不允许修改门槛或对单个 Seed 补训练。

## 判定门槛

必须同时满足：

1. 三个 Seed 的 `test_composition/relation_chain` 均不低于80%；
2. 三 Seed 关系链均值不低于85%，标准差不超过8个百分点；
3. 每个 Seed 的 D 关系链 required recall 不低于90%；
4. 每个 Seed 的 D 关系链 offline pairwise accuracy 不低于95%；
5. 七种前缀压力变体中，每个 Seed 的最差成功率不低于70%；
6. `test_composition` 其他五项跨 Seed 均值不低于95%，且任意 Seed/任务不低于85%；
7. `test_id` 跨 Seed平均成功率不低于99%；
8. 四测试集平均 eviction regret 不超过0.003。

通过后冻结 C-Robust 作为记忆选择器基线，停止继续优化该680万参数验证编码器，转入
1B 主模型的集成设计。
