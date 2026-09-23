# Phase 0 B2.4 关系消歧训练计划

## 冻结项

- 记忆公式不变；
- 损失保持 $L=L_{reg}+0.5L_{rank}$；
- 原 B2.2 的 36000 个训练决策组和 4800 个验证决策组原样保留；
- 最终 B2.3-D 的长度 256、$K=32/64$ 测试集不参与训练或 checkpoint 选择。

## 新数据

增加 36000 个训练决策组和 4800 个验证决策组。每个新决策组必须同时包含已经出现的 required edge 和 decoy edge，并保证关系边总数超过容量。训练范围：

- 长度 32、48、64；
- 容量 4、8、12；
- 跳数 2、3、4；
- decoy 4、8、12、16；
- disconnected、spoke、dead branch 三种拓扑；
- 训练使用 RT 模板，验证使用 RV 模板。

## 模型对照

| 变体 | d_model | record layers | set layers | ff_dim |
|---|---:|---:|---:|---:|
| Base | 256 | 6 | 2 | 1024 |
| DeepSet | 256 | 6 | 6 | 1024 |
| Wide | 384 | 6 | 2 | 1536 |

每个变体训练 model seed 6、7；data/evaluation seed 均固定为 0。所有变体训练 6 epoch，并使用混合 validation total 选择 checkpoint。

## 解释

- Base 恢复：主要是训练分布问题；
- DeepSet 明显领先：需要更多集合消息传播深度；
- Wide 明显领先：主要受宽度/参数量限制；
- 三者均失败：进入显式关系消息传递设计。
