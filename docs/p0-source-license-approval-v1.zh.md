# P0 来源许可批准记录 v1（2026-09-24）

本记录按 [`docs/p0-50m-data-pipeline-plan.md`](p0-50m-data-pipeline-plan.md) 第 3 节的批准门，逐个核实
P0-A 六个可用来源的数据卡与许可条款，作为把适配器与来源注册表里的 `license_status`
从 `pending` 改为 `approved` 的依据。**批准的是来源许可，不是内容质量**；P0 第 14 节的验收条件
（分桶 100 条抽检、评测污染、近重复聚类等）仍未完成。

## 1. 批准结果

核实方式：对注册表中固定的 40 位 revision，直接读取 Hugging Face API
（`/api/datasets/<repo>?revision=<sha>&full=true` 的 `cardData.license` 与 `tags`）和该 revision 的
`README.md` 原文（`/datasets/<repo>/raw/<sha>/README.md`），不依赖移动中的 `main`。

| `source_id` | 数据集 | 固定 revision | 许可 | 数据卡原文 |
| --- | --- | --- | --- | --- |
| `fineweb_edu` | `HuggingFaceFW/fineweb-edu` | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | ODC-By 1.0（`license:odc-by`） | “released under the **Open Data Commons Attribution License (ODC-By) v1.0**… also subject to [CommonCrawl's Terms of Use]” |
| `fineweb2_zh` | `HuggingFaceFW/fineweb-2` | `af9c13333eb981300149d5ca60a8e9d659b276b9` | ODC-By 1.0（`license:odc-by`） | 同上，另注明 “available under the permissive **ODC-By 1.0 license**” |
| `finemath` | `HuggingFaceTB/finemath` | `e92b25a616738fe95dc186b64dfb19f9c8525594` | ODC-By 1.0（`license:odc-by`） | 同上 |
| `github_code_clean_python` | `codeparrot/github-code-clean` | `c48d40f9e70f0196f8236901ee35807f7d6c44c0` | Apache-2.0（`license:apache-2.0`）+ 逐文件许可白名单 | 数据卡仅有 `license: apache-2.0`；行过滤器只放行 `mit`、`apache-2.0`、`bsd-3-clause`、`bsd-2-clause`、`isc`、`cc0-1.0`、`unlicense` |
| `github_code_clean_shell` | 同上 | 同上 | 同上 | 同上 |
| `github_code_clean_javascript` | 同上 | 同上 | 同上 | 同上 |

未批准、保持 `pending` 的来源：

- `stack_edu_python`、`stack_edu_shell`、`stack_edu_javascript`（`HuggingFaceTB/stack-edu`
  @ `eeec5caac5cc3758a18f1d3ba4416837a9ba814c`）：该 revision 的 `cardData` **没有任何 license 字段**，
  数据卡明确要求按 [the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2-train-full-ids)
  的许可执行，且正文必须另行从 Software Heritage S3 下载。当前适配器不含正文
  （`adapter_status=rejected`），因此既无许可结论、也无正文可用，不能批准。
  重新启用需先解析并缓存正文、且取得 The Stack v2 的许可与退出机制结论。

## 2. 逐条核对批准门

| 批准条件 | 结论与证据 |
| --- | --- |
| 固定数据集 revision/commit | 六个来源在 `configs/data/p0_sources_v1.json` 中均为 40 位 revision，并记录 `remote_file`、`raw_relpath`、`actual_bytes`、`sha256` |
| 记录数据卡和许可条款 | 本文件第 1 节 |
| 正文确实可取得 | V100 上六个来源的 `normalized/*/manifest.json` 均以真实 Parquet schema 校验通过并产出正文；例如 `fineweb_edu` seen 726,000 / kept 725,771，`fineweb2_zh` seen 1,973,000 / kept 1,931,068 |
| 来源字段可以追溯 | 每个适配器声明 `locator_fields`、`family_fields`、`metadata_fields`，`validate_source_fields` 对正文列与定位列 fail-closed |
| 小样本确认提取质量 | P0-A 全量教师门把 2,058 篇候选全文送审（来源分布：`fineweb_edu` 688、`fineweb2_zh` 781、`finemath` 153、三个代码桶合计 436），并逐条记录保留理由；规范化阶段另有逐规则丢弃计数 |
| 没有把测试 split 混入训练 | 只读取 train 文件：`sample/100BT/000_00000.parquet`、`data/cmn_Hani/train/000_00000.parquet`、`finemath-3plus/train-00000-of-00128.parquet`、`data/train-00000-of-00880.parquet`。`fineweb-2` 的 `cmn_Hani` 配置同时含 test split，管线未取用；`github-code-clean` 该 revision 只有 `data/train-*-of-00880.parquet`。管线另按 `family_id` 哈希切分 train/validation/test |

## 3. 批准后仍然存在的义务与风险

1. **署名（ODC-By 1.0）**：`fineweb_edu`、`fineweb2_zh`、`finemath` 发布的任何派生数据、报告或模型卡
   必须保留来源名称、固定 revision 与 ODC-By 1.0 说明；缺失署名即违反许可，而不是可选项。
2. **CommonCrawl 使用条款**：三个网页/数学来源的正文来自 Common Crawl，使用同样受
   CommonCrawl ToU 约束（数据卡原文如此）。本轮未逐站点核查 robots/ToU 例外。
3. **代码来源是“声明许可”而非“验证许可”**：`github-code-clean` 的 `license` 列来自上游启发式识别，
   逐文件可能判错；数据集自身的 Apache-2.0 标签覆盖的是汇编，不改变各文件原有许可。
   代码 shard 必须保留 `license`、`repo_name`、`path` 元数据，以便日后按仓库追溯或下架。
   该来源没有 The Stack 那种 am-I-in-the-stack 退出机制，属已知残余风险。
4. **许可批准 ≠ 质量验收**：P0-A 教师门的保留率显示中文通用（8.6%）与英文数学（10.5%）明显不足，
   结论仍是“补来源”，而不是放宽质量门。`training_eligible=true` 只表示许可门放行，
   不表示该 100 万 token 冒烟集已达到 P0 第 14 节的验收标准。
5. **Stack-Edu 仍未解决**：见第 1 节。

## 3.1 许可门本身曾有假阳性缺陷（2026-09-24 修复）

核实过程中发现，报告 `training_eligible` 的五个位置里有四个存在**空集真值**问题，
`build` 一处更是完全不看内容：

| 位置 | 修复前 | 问题 |
| --- | --- | --- |
| `p0_candidate_pipeline.py` build | `not args.allow_pending_license` | **只回显命令行开关**，与实际内容无关 |
| `p0_candidate_pipeline.py` consolidate | `all(... for record in retained)` | 空池时 `all([])` 为 `True` |
| `p0_freeze_accepted.py` | `set(licenses) <= {"approved"}` | 空集是任何集合的子集 |
| `p0_merge_smoke_dataset.py` | `set(licenses) <= {"approved","generated"}` | 同上 |
| `p0_tokenizer_corpus.py` | `set(licenses) <= {"approved"}` | 同上 |

已在 V100 上实测复现：用 `approved` 配置配旧 `normalized`（逐行 `pending`），
`build` 不加 `--allow-pending-license` 时写出

```json
"license_gate": {"allow_pending_for_smoke": false, "training_eligible": true}
```

而同一份 manifest 的 `index_counts` 是 `{"rejected_license": 161503, "rule_and_license_pass": 0}`，
即**零篇入选却报告可训练**。这正是许可门要防的失败模式，因此五个位置统一改为
「非空 **且** 全部为 `approved`」，并新增回归测试
`test_empty_accepted_set_is_not_training_eligible`（已验证在修复前失败、修复后通过）。
同时 `build` 的 manifest 增加 `statuses` 字段，使该阶段与 `freeze` 一样按内容取证。

## 4. 对既有产物的影响：必须重跑，不得手工改 manifest

`training_eligible` 不是配置项，而是由产物内容计算出来的：

- `scripts/p0_candidate_pipeline.py`：`build` 在 `--allow-pending-license` 下写入
  `license_gate.training_eligible=false`；`index_normalized_documents` 在没有该开关时直接跳过
  非 `approved` 行；
- `scripts/p0_freeze_accepted.py`：`training_eligible = set(license_status) <= {"approved"}`，
  统计对象是接受 shard 里每一行的 `license_status`。

因此本次改动之后：

- 仓库配置：六个适配器与注册表条目已改为 `approved`，`validate-registry` 要求两者一致；
- V100 既有产物（`/home/data/wangyue/datasets/small-agent-p0/p0_1m_smoke_v1/`）是在 `pending` 门下
  生成的，`normalized/*/` 的每一行、`candidates_consolidated_v1/manifest.json` 与
  `accepted_real_v1/manifest.json`（`statuses: {"pending": 690}`）都已把 `pending` 固化进内容与哈希。
  它们**仍是未获许可批准的产物**，必须按同一实现重跑
  `normalize → build → consolidate → freeze` 才能得到 `training_eligible=true` 的新 manifest。
- 重跑后必须逐项比对文档 ID 与 shard 行数，确认候选集未因许可门变化而改变；
  严禁直接编辑既有 manifest 的 `license_gate` 字段来“通过”许可门。
