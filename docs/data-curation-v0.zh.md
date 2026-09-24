# 400 条语料分流流水线 v0

这是一条**离线、可重跑、非破坏性**的小规模验收流水线。它只读取已有的 400 条摘录与 v1/v2 模型预测，不再次调用 API、不使用 GPU，也不修改 Label Studio。输出的 `train` 是内容候选，**不是来源许可已核对的最终训练数据**。

## 输入与单条记录

- 冻结输入：`label_studio_text_math_tasks.json`，来源文件与行号共同组成稳定 ID。
- 两份预测：`deepseek_flash_predictions_v1.jsonl` 和 `deepseek_flash_predictions_v2.jsonl`。两者都必须 400/400 成功、没有重复 ID，且每条文本 SHA-256 与冻结输入一致。
- 策略：`configs/data_curation_policy_v0.json`。其中只有高精度的可见来源信号与明显提取损坏模式；没有把“娱乐/冷门/短”当作自动删数据条件。
- 记录包含 `extraction`、`content`、`provenance` 三个独立维度，以及 `route`、规则名、证据、文本 SHA-256 和策略哈希。结果中不复制原始正文。

## 路由顺序

1. 完全相同的正文：保留稳定 ID 排序第一条，其余放 `drop` 候选。原文不会删除。
2. 明确的付费预览、锁定题解等来源线索，或模型提出隐私/版权问题：放 `source_check`；**不宣称已侵权**。
3. 高精度的核心格式损坏信号：放 `repair`，需回原文重提取，不等同于内容低质。
4. v1 与 v2 都判剔除：放 `drop` 候选；两次预测相关，不能视为人工真值。
5. v1 与 v2 都判保留：放 `train` 内容候选；来源权限仍未确认。
6. 其他情况：放 `review`，专门表示评分分歧或模型弃权，不混入修复队列。

## 运行

在仓库根目录：

```bash
PY=/opt/homebrew/bin/python3.11
DATA=/Users/wangyue/Documents/Codex/2026-09-21/wo-z/outputs/data-audit-2026-09-23

"$PY" scripts/run_data_curation_v0.py \
  --tasks "$DATA/label_studio_text_math_tasks.json" \
  --v1 "$DATA/deepseek_flash_predictions_v1.jsonl" \
  --v2 "$DATA/deepseek_flash_predictions_v2.jsonl" \
  --policy configs/data_curation_policy_v0.json \
  --run-dir "$DATA/curation_v0_400"
```

一个 run 目录内只有：`records.jsonl`、五个 `*_ids.jsonl`、`manifest.json`、`report.md`。输出按稳定 ID 排序，不写时间戳；用相同输入重跑应得到相同字节。所有文件只允许当前用户读取。

## 当前 400 条结果

| 路由 | 条数 | 含义 |
| --- | ---: | --- |
| `train` | 244 | 两次评分均保留，尚非最终训练集 |
| `repair` | 3 | 可见提取损坏；包括压平代码/损坏公式 |
| `drop` | 64 | 两次评分均剔除或精确重复；不删除原文 |
| `source_check` | 8 | 有显式来源/使用限制线索 |
| `review` | 81 | v1/v2 去留分歧或弃权 |

这是**队列分流结果而非准确率**。上面两次评分来自同一模型体系，v2 的默认采样温度还出现过重复运行不稳定。接下来应先检查 `repair` 和 `source_check` 的规则命中，再从 `train` 与 `drop` 中分层抽检；对 `review` 按来源与风险排序处理。40 条未见样本仍可用于独立抽检，不应作为改写此 v0 规则的训练集。
