# 可复用语料评分管线

这个管线把「一段提示词 + 输入 JSON/JSONL」转换为逐条 JSONL 评分结果。核心脚本不依赖 Label Studio；平台导入由单独适配器完成。

## 输入与输出

- 输入：`.json` 数组或 `.jsonl`，每条记录为对象。
- `--text-field`：要交给模型的文本字段，支持 `data.text` 这样的嵌套路径。
- `--id-fields`：共同定义稳定样本 ID；跨数据源时应同时传来源与行号。
- `--carry-fields`：复制少量来源字段到输出，便于回填；不会复制全文。
- `--prompt-file`：逐条应用的系统提示词，可以替换为其他审核任务。
- `--schema-file`：模型输出必须满足的 JSON Schema。
- 输出：追加式 JSONL，每条有 `source_id`、文本/提示词/Schema 哈希、模型名、`annotation`、token 用量或错误。成功结果在重跑时自动跳过；提示词、Schema、文本或模型变化则重新评分。

失败项保留 `status=error`，下次运行会重试。密钥只从环境变量读取，结果文件权限为 `0600`，不输出原始文本或密钥。

## 安装与运行

```bash
python3 -m pip install -r requirements-annotation.txt
export DEEPSEEK_API_KEY='从你的密钥管理器读取，不要写进仓库'

python3 scripts/annotate_jsonl.py \
  --input /path/to/samples.jsonl \
  --output /path/to/scores.jsonl \
  --prompt-file configs/data_audit_prompt_v1.txt \
  --schema-file configs/data_audit_schema_v1.json \
  --text-field excerpt \
  --id-fields row_idx \
  --carry-fields row_idx \
  --model deepseek-flash \
  --workers 4
```

先用 `--dry-run` 检查字段和条数，或用 `--limit 10` 做小样本试跑。`--api-base` 和 `--api-key-env` 可替换为其他 OpenAI Chat Completions 兼容服务；若服务不支持 DeepSeek 的 `thinking` 参数，传 `--thinking omit`。JSON 输出模式仍须由服务端支持。

当前 Label Studio 项目使用 `--text-field data.text --id-fields data.source_file,data.source_row_idx --carry-fields data.source_dataset,data.source_file,data.source_row_idx`。模型 API 名称为 `deepseek-flash`，对应 DeepSeek V4.1 Flash；输出 JSON 模式按[官方说明](https://api-docs.deepseek.com/guides/json_mode/)设置。

## Label Studio 适配

`scripts/import_labelstudio_predictions.py` 在 **Label Studio 所在机器的 Python 环境**运行，读取通用评分 JSONL，通过本地 API 创建 `Prediction`，不会改写人工 `Annotation`。它核对项目中的来源键和文本 SHA-256，并按 `model_version` 去重。先 `--dry-run`，再实际导入。项目可以继续人工复核模型建议。

切换提示词或 Schema 版本时建议换一个输出文件和 `model_version`，避免将两个评分版本混在同一次平台导入中。

## 2026-09-23 首轮验收

项目：400 条文本/数学摘录，四个来源各 100 条。7 条现有人工标注做盲评：去留决策一致 7/7，质量档位一致 4/7；**样本太少且全是英文网页，不代表真实准确率**。全部评分都只是摘录层面的机器预测。下一步应按来源分层抽检，重点看中文、数学和模型标「保留」的误收率。

v2 的固定回归集、400 条复测及当前限制见 [400 条 v2 验收记录](data-audit-v2-400.zh.md)。v2 还未通过未见样本的人工抽检，不要覆盖 v1 预测。

现有 400 条的非破坏性离线分流见 [语料分流流水线 v0](data-curation-v0.zh.md)。它不再调用 API，而是把提取损坏、内容评分分歧和来源线索分开排队。
