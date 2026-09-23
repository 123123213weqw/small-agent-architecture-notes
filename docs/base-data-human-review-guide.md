# 基础数据人工复核包（给独立审核 Agent）

本仓库公开[400 条待审索引](../results/base_data_audit_pilot_20260923/review_queue.csv)：四源各 100 条。索引只有来源、行号、长度、分层、评分和内容哈希，**不重新发布第三方原文**。完整内容可由审核者从各来源的 Hugging Face Dataset Viewer 读取到其本地工作目录。

## 1. 获取待审全文

在仓库根目录执行：

```bash
python3 -m pip install requests
python3 experiments/base_data_review.py materialize \
  --queue results/base_data_audit_pilot_20260923/review_queue.csv \
  --output /tmp/base_data_review_text.jsonl
```

脚本按 20 个随机窗口/来源批量获取，再只保存待审 100 条/来源；逐条校验 `normalized_sha256`。如果来源内容变化，脚本会报错，**不要把新内容误当本轮样本**。可用 `--source python_edu` 等参数只取一个来源。API 可能限流，脚本会退避重试。此文件是本地临时材料，不要提交 Git。

审核数据来自[第一轮审计汇总](../results/base_data_audit_pilot_20260923/summary.json)；脚本见 [`experiments/base_data_review.py`](../experiments/base_data_review.py)。Dataset Viewer 的请求没有锁定 revision，哈希校验是必要的，但不替代日后从固定 revision 正式入库。

## 2. 抽样设计

- 每源从先前随机窗口样本的 1,000 条中再抽 100 条。
- 非代码源按原始评分排名划成四组，每组 25 条；组内按文本长度划五档，每档 5 条。评分只是**分层工具**，不自动代表质量。
- Python 源按 `int_score=1/2/3/≥4` 四组各抽 25 条，组内也按长度五档抽取。高分代码被**刻意超采样**；总体可用率必须按 `stratum_population` 加权，不能直接把 100 条平均。
- 随机种子 `20260923`；原始 1,000 条来自 20 个随机起点，每起点连续 50 行，因此样本并非完全独立。

## 3. 逐篇标签

请阅读全文，而非只看开头。将索引 CSV 复制到自己的工作目录，填写以下列：

- `decision`: `accept` / `repair` / `reject` / `uncertain`。`accept` 意味着内容本身有用且结构完整；**不代表许可已经批准**。
- `reason`: 可用逗号分隔的原因，例如 `off_topic`, `boilerplate`, `seo`, `wrong_language`, `too_short`, `truncated`, `repetitive`, `formula_corrupt`, `answer_missing`, `code_fragment`, `generated_file`, `outdated`, `pii`, `license_unknown`, `eval_overlap`。其中 `outdated` 应基于正文中的过时 API/事实判断，不能只凭抓取年份。
- `notes`: 简短中文判断依据，不复制长篇原文或敏感信息。

重点核查：FineWeb-Edu 是否真正提供解释性文本；FineWeb2-HQ 中文是否通顺、有信息量、不过度新闻/营销化；FineMath 的题目、公式和答案是否完整；Python 的 3 分与 4 分内容是否值得训练、元数据前缀是否可清洗。**不要运行来源代码。**

## 4. 汇总要求

报告每源 `accept/repair/reject/uncertain` 数、主要排除原因和具有代表性的结构性问题。代码源另报告各 `int_score` 组的可用率，并用 `stratum_population / 1000` 对四组加权估计本轮 1,000 条中的可用比例；给出“不确定”的范围，不要把小样本比例写成全库真值。

这轮不解决全库近重复、评测污染、逐站点/逐仓库许可或最终 tokenizer token 数；把发现的问题转成下一轮清洗规则和复核实验即可。
