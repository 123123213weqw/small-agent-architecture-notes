# Phase 0B2.1 数据生成与审计

## 目标

B2.1 不再把单条记录作为一个样本。每行 JSONL 表示一次完整的淘汰决策，包含相同容量约束下的全部 $S+1$ 条竞争记录。

未来记录只参与反事实效用标签计算，不会序列化到模型输入。

## 决策组格式

```text
schema_version
split
episode_id
decision_id
decision_position
task
query_visibility
template_family
capacity
hard_negative_count
behavior_policy
goal
recent_context[]
records[]
candidate_index
utilities[]
oracle_eviction_indices[]
input_sha256
```

每条 `records[]` 只包含：

```text
uid
text
event_index
relative_age
is_candidate
source
```

训练代码必须通过 `group_model_input()` 取得模型输入。效用向量、Oracle 集合、行为策略、任务名和 split 不进入预测器。

## 模板与测试集

- A、B、C：训练、验证和 ID 测试；
- D：只重组 A、B、C 已出现的表达片段，用于组合泛化；
- E：未在训练模板中使用的表达，用于语义压力测试；
- 长度外推使用 D，长度为 96 或 128。

动态实体统一加 split 命名空间，例如：

```text
train_state_123
test_composition_state_123
```

生成器会检查任意两个 split 之间不存在 episode ID 或动态实体交集。

## 查询可见性

当前符号环境只在 `delayed_query` 任务中自然产生隐藏查询：目标可能只说明以后会提问，而不公布具体实体。其他五类任务的目标在决策时可见。

生成器不会为了追求数量平衡而人为隐藏其他任务的目标。评估时必须分开报告 `announced_query` 与 `hidden_query`，并承认后者存在不可预测上限。

## 行为策略

数据状态由以下三种策略按 episode 均衡收集：

- FIFO；
- Random；
- Future Oracle。

行为策略只决定下一个记忆状态，不进入模型输入。每个 episode 最多均匀采样 12 个决策组。

## 自动检查

每个决策组写入前检查：

1. 记录数严格等于 `capacity + 1`；
2. 只有一个新候选；
3. 候选索引正确；
4. 竞争记录不晚于当前决策；
5. 近期上下文严格早于当前决策；
6. utility 数量与记录数量一致；
7. Oracle 集合恰好等于最低效用集合；
8. 模型输入不包含效用、Oracle、结构化真值或行为策略；
9. 模型输入哈希可以重新计算；
10. split 之间 episode 和实体完全隔离。

## 生成命令

审计集：

```bash
python experiments/phase0_b21_data.py \
  --preset audit \
  --seed 0 \
  --output data/b2_1_stage1_audit
```

正式第一阶段数据：

```bash
python experiments/phase0_b21_data.py \
  --preset stage1 \
  --seed 0 \
  --output data/b2_1_stage1
```

输出包括：

```text
manifest.json
AUDIT.md
train.jsonl
validation.jsonl
test_id.jsonl
test_composition.jsonl
test_length.jsonl
test_semantic_stress.jsonl
```

正式数据只在审计集人工检查通过后生成。

## Seed 0 审计集

审计集包含：

- 训练 100 episodes、1,200 个决策组；
- 验证 24 episodes、288 个决策组；
- 四个测试集各 24 episodes、288 个决策组；
- 六类任务、三个容量、四种 hard-negative 数量和三种行为策略均已覆盖。

训练集 36.2% 的决策组至少包含一条正效用记录，正效用记录占全部竞争记录的 5.2%。这说明数据具有明显类别不平衡；后续训练需要在 loss 或 batch sampler 中处理，而不能把 `important` 一类真值字段放入模型输入。

完整文件哈希与分布见：

```text
results/phase0_b21_data_audit/manifest.json
results/phase0_b21_data_audit/AUDIT.md
```

## Seed 0 正式第一阶段数据

审计集与全部自动测试通过后，已经在 L40 的私有 JuiceFS 工作区生成正式数据：

```text
/myjfs/94f3304c-d49d-4e45-bd8c-69cea6ddfe0c/25212408112/
small-agent-architecture-notes/data/b2_1_stage1/
```

规模：

- 训练 3,000 episodes、36,000 个决策组；
- 验证 400 episodes、4,800 个决策组；
- ID、组合、长度和语义压力测试各 600 episodes、7,200 个决策组；
- 总体积约 153 MiB；
- 生成耗时约 43 秒；
- 训练集有信息决策比例 36.3%，正效用记录比例 5.1%；
- 长度外推测试有信息决策比例 26.2%，必须单独报告，不能与其他测试集合并。

正式数据 manifest 与审计摘要见：

```text
results/phase0_b21_stage1_data/manifest.json
results/phase0_b21_stage1_data/AUDIT.md
```
