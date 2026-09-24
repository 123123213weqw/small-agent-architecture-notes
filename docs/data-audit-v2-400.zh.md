# 400 条语料摘录：v2 评分验收

## 范围与冻结项

- 输入保持原来的 400 条摘录（四个来源各 100 条），不重新截取全文；本次只比较**摘录层**判断。
- 模型仍为 `deepseek-flash`，输出仍用 `data_audit_schema_v1.json`。只修改提示词为 `data_audit_prompt_v2.txt`，结果写入新文件，不覆盖 v1。
- 先前独立 Agent 的 80 条样本是开发集。13 条分歧仲裁作为回归用例；它们不是独立测试，更不是人类金标准。
- 其余 320 条是未参与 v2 制定的测试池。从中提前固定每来源 10 条、合计 40 条的人工抽检 ID（`v2_holdout40_manifest.json`）。不因为 v2 输出而重新抽样。

## v2 相比 v1 的关键变化

1. 判断当前摘录，不把“原网页可修复”混同为“待复核”。
2. 只有确实无法决定当前摘录去留才标“待复核”；题材争议、普通截断、复杂推导暂未验算都不是充分条件。
3. 检查数学事件能否合法取到 0 概率、近似与精确的区别，以及论坛后续是否纠正了前文。
4. 代码、公式等核心内容被采集破坏时，不能仅凭主题有价值而放行。
5. “事实或推导错误”须有可见证据；付费墙不是已证明侵权。

## 执行

在仓库根目录使用带 `jsonschema` 的 Python（本机 `/opt/homebrew/bin/python3.11`）。先准备固定回归集和抽检清单：

```bash
DATA=/Users/wangyue/Documents/Codex/2026-09-21/wo-z/outputs/data-audit-2026-09-23
PY=/opt/homebrew/bin/python3.11

"$PY" scripts/data_audit_v2_eval.py prepare \
  --tasks "$DATA/label_studio_text_math_tasks.json" \
  --cases configs/data_audit_adjudication13_v2.json \
  --dev-sample "$DATA/independent_review_sample80_v1.jsonl" \
  --regression-output "$DATA/v2_regression13_tasks.json" \
  --holdout-output "$DATA/v2_holdout40_manifest.json"
```

DeepSeek 密钥只从本机钥匙串读取，不写入仓库、输出文件或命令参数，也不持久留在交互式 shell 环境。先运行 13 条回归：

```bash
DEEPSEEK_API_KEY="$(security find-generic-password -a "$USER" -s deepseek-api-key -w)" \
"$PY" scripts/annotate_jsonl.py \
  --input "$DATA/v2_regression13_tasks.json" \
  --output "$DATA/deepseek_flash_predictions_v2_regression13.jsonl" \
  --prompt-file configs/data_audit_prompt_v2.txt \
  --schema-file configs/data_audit_schema_v1.json \
  --text-field data.text \
  --id-fields data.source_file,data.source_row_idx \
  --carry-fields data.source_dataset,data.source_file,data.source_row_idx \
  --model deepseek-flash --workers 4
```

先检查回归结果。若有系统性错误，修改提示词并从新的结果文件重跑；不要通过反复微调只追求 13/13。回归可接受后，再对 400 条运行同一提示词（`--input "$DATA/label_studio_text_math_tasks.json"`，`--output "$DATA/deepseek_flash_predictions_v2.jsonl"`）。两个输出文件分开保存，避免测试与正式运行混用。

对比报告：

```bash
"$PY" scripts/data_audit_v2_eval.py report \
  --tasks "$DATA/label_studio_text_math_tasks.json" \
  --cases configs/data_audit_adjudication13_v2.json \
  --v1 "$DATA/deepseek_flash_predictions_v1.jsonl" \
  --v2 "$DATA/deepseek_flash_predictions_v2.jsonl" \
  --report-output "$DATA/data_audit_v2_400_report.md" \
  --changes-output "$DATA/data_audit_v1_v2_decision_changes.jsonl"
```

## 完成条件

- 400/400 条 v2 输出成功，且与冻结输入的文本 SHA-256 全部一致；提示词/Schema/模型版本未混用。
- 逐条检查全部去留变化；用固定的 40 条未见样本抽检两版共同判断的误收/误剔。
- 报告按来源列出保留/待复核/剔除数量、去留变化、13 条回归通过数和错误类型。
- 不将两版一致率或 13 条回归通过率宣称为真实准确率；有人工仲裁后另存独立标签。

## 2026-09-23 首轮结果与下一步

- v2 严格版用 API 默认温度跑完 400/400，哈希全部对齐；相对 v1，去留变化 83 条。详见本地 `data_audit_v2_400_report.md`。
- 在此前独立 Agent 的 80 条开发样本上，去留一致数从 v1 的 67/80 变成 v2 的 63/80，质量一致数从 59/80 变成 50/80。该复核不是人类金标准，但目前没有证据说明 v2 整体更准确。
- 同一提示词的 13 条回归，单独运行与全量运行有 3/13 个决定不同；因此默认温度的输出不宜直接作为稳定自动标签。
- 增加可选 `--temperature 0` 后，13 条独立重复运行结果 13/13 一致，但相对本次暂定仲裁标签均为 9/13。稳定不等于正确；其中 4 条分歧需要审查规则与标签，而非继续针对回归集硬调提示词。
- 40 条未见样本已经预先固定，但尚未得到人工标签。v2 当前是**候选实验**，不应覆盖 Label Studio 的 v1 Prediction，也不应推广到全量语料。
