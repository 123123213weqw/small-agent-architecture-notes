# 1B 候选语料诊断模型：评测流水线 v1

## 边界

本流水线只针对 `unreviewed_candidate_diagnostic_50m_v1` 的第 382 步。训练数据尚未完成质量批准，`quality_approved=false`；模型导出和分数都**仅供内部工程诊断**，不能称为合格 Base 或最终能力分数，也不上传 HF/GitHub。

评测分三层，不能混用：

1. **训练数据独立验证集**：下一 token loss / perplexity，检测训练是否正常；不是通用能力分数。
2. **公开基准**：使用 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md) 的官方任务定义，离线镜像固定版本的测试数据。先做 `--limit` 冒烟；正式统计必须去掉 `--limit`。
3. **推理一致性与 KV/cache**：相同 token 前缀比较全量前向与增量 cache 的 logits；同时记录 cache 张量层数、形状和字节数。此测试不能以验证集 loss 替代。

代码生成后续接 [EvalPlus 的 HumanEval+/MBPP+](https://github.com/evalplus/evalplus/blob/master/docs/cli.md)。当前 50M-token 早期 Base 尚不能可靠完成代码提示，先不把生成代码的 pass@1 当成架构证据；届时使用其 `samples.jsonl` 格式，并把执行评测放在隔离容器内。

## 实现文件

| 文件 | 作用 |
| --- | --- |
| `scripts/export_hf_diagnostic_checkpoint.py` | 核验 run、tokenizer、数据和 checkpoint 哈希；BF16 HF 格式导出；重新加载并对照 logits |
| `scripts/eval_sharded_checkpoint_domains.py` | 固定验证集分领域 loss / perplexity，已有输出 |
| `scripts/eval_cache_parity.py` | 验证集真实 token 上测全量/增量 logits 与 cache 占用 |
| `scripts/prepare_local_lm_eval_tasks.py` | 保持 lm-eval 官方题目格式和评分方式，只将数据源改为固定 revision 的本地 Parquet |
| `scripts/run_lm_eval_diagnostic.py` | 离线跑任务，核验文件哈希，记录版本、配置、日志、样本和状态；拒绝覆盖已有输出 |

L40 上统一目录：

```text
/data1/wangyue/experiments/small-agent-base-1b-v1-engineering/
├── eval-data/                    # 固定版本的公开基准 Parquet
├── eval-deps/                    # 隔离的 lm-eval 依赖，不改训练 venv
├── eval-cache/                   # datasets/HF 缓存
└── runs/unreviewed_candidate_diagnostic_50m_v1/
    ├── hf_export_step382_bf16/  # 非分发式 HF 模型
    └── eval/
        ├── cache_parity_step382.json
        ├── local_tasks_v1/
        └── lm_harness_local_pilot_v1/
```

## 已完成的工程冒烟

- 第 382 步原始 checkpoint 的 `model.safetensors` 哈希与 manifest 一致；导出后重新加载，相同 prompt 最后一个位置 logits 的最大绝对差为 **0.03125**（BF16）。
- 官方 ARC-Easy 任务定义通过本地 Parquet 离线运行，8 题、0-shot、32 个候选项 log-likelihood 请求顺利完成；小样 `acc=0.25`、`acc_norm=0.125`。**8 题没有统计意义，不应拿来比较模型。**
- 官方 HellaSwag 任务定义同样完成离线 8 题、0-shot 冒烟；小样 `acc=0.25`、`acc_norm=0.375`，也不能当作能力结论。两组输出均保存了逐题样本和汇总 JSON。
- cache 审计在验证集样本上比较 32/128/512 token 前缀，各续接 8 个真实 token。cache 确认有 **8 层全注意力 K/V** 和 **24 层 GDN 卷积/递归状态**；32-token 前缀 cache 约 15.4 MB，512-token 前缀约 21.3 MB。最大 logits 绝对差约 0.09；512 长度有 1/8 个位置发生贪心 top-1 分歧（候选分数非常接近）。因此目前**不能声称 cached 与 uncached 严格等价**，需要扩大样本、分析 BF16 数值误差并测真实生成轨迹。
- L40 无法直接连 Hugging Face，所以首次在线任务读取失败；改为 Mac 下载固定 revision 后传入 L40，评测端设置 `HF_DATASETS_OFFLINE=1`。这不是模型或任务失败。

## 可复现调用

下面命令在 **L40** 的项目代码目录运行；先检查 GPU 空闲。训练 venv 路径为 `/home/wangyue/.venvs/small-agent-l40/bin/python`，`eval-deps` 只给评测进程使用。

```bash
B=/data1/wangyue/experiments/small-agent-base-1b-v1-engineering
R=$B/runs/unreviewed_candidate_diagnostic_50m_v1
PY=/home/wangyue/.venvs/small-agent-l40/bin/python
cd "$B/code"
export PYTHONPATH="$B/deps:$B/code/src:$B/code"
export PYTHONNOUSERSITE=1 TRITON_CACHE_DIR="$B/triton-cache" TMPDIR="$B/tmp"
export CUDA_VISIBLE_DEVICES=0

# cache 的数值与形状审计；结果即使未达严格门槛也保存 JSON 并以非零状态退出
$PY scripts/eval_cache_parity.py \
  --model "$R/hf_export_step382_bf16" \
  --data "$B/data/unreviewed_candidate_100m_sharded_v1" \
  --output "$R/eval/cache_parity_step382.json"

# 只做输入检查，不启动 GPU、不覆盖已有结果
$PY scripts/run_lm_eval_diagnostic.py \
  --model "$R/hf_export_step382_bf16" \
  --task-dir "$R/eval/local_tasks_v1" \
  --eval-data "$B/eval-data" \
  --eval-deps "$B/eval-deps" --cache-dir "$B/eval-cache/hf" \
  --output "$R/eval/lm_harness_full_v1" \
  --tasks local_arc_easy local_hellaswag --dry-run

# 需要真实全量指标时去掉 --dry-run，且不要传 --limit；输出目录必须为空
```

`local_tasks_v1/manifest.json` 保存官方任务定义及 Parquet 的 SHA-256、数据集 revision：`allenai/ai2_arc@210d026...`、`Rowan/hellaswag@218ec52...`。它们只更换 `dataset_path` 为本地 Parquet，不改 prompt、choice、metric；HellaSwag 的 `process_docs` 复制官方 `utils.py`。

## 下一步门槛

1. 全量 ARC-Easy/HellaSwag 0-shot；不使用 `--limit`，原样保存逐题样本与官方聚合结果。
2. 增加 held-out 文本的语言建模指标和数学任务；先确认训练池与公开基准无重叠（至少文档哈希和近重复抽查），否则标注污染风险。
3. 将 cache 对照扩至不同长度、不同验证样本、长生成，区分 BF16 近似差异与真实状态更新错误。
4. 后续训练出会生成代码的模型再接 EvalPlus；代码执行评测与 GPU 推理分开。
