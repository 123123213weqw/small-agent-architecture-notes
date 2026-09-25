# P0-B 候选试产与分歧复核记录

日期：2026-09-24。目标是先用已批准的真实来源扩建候选池，并校准本地质量评审；**候选池不等于已验收训练集**。

## 输入与隔离

- 上游为 `p0_1m_approved_rebuild_v1/normalized/` 的六个已批准来源，约 283 万篇规范化文档；P0-B 通过只读符号链接复用，不复制 6.8 GiB Parquet。
- 新目录为 `/home/data/wangyue/datasets/small-agent-p0/p0_b_candidate_25m_v1/`，独立保存配置快照、候选分片、评分校准和日志，不改写 P0-A。
- `p0_gen_verified_v2` 合成数据继续独立放置，不混入本轮真实语料候选。
- 候选配额用原先固定的 Qwen3-0.6B tokenizer 计数；最终训练数据仍须用自有 Byte-level BPE 32k tokenizer 重新计数。

## 候选配额（非最终训练量）

配置：`configs/data/p0_b_candidate_25m_v1.json`，总计 2,500 万 Qwen token：

| 域 | 候选 token |
| --- | ---: |
| 中文通用 | 1,000 万 |
| 英文通用 | 800 万 |
| Python | 300 万 |
| Shell | 35 万 |
| 其他代码 | 65 万 |
| 英文数学 | 300 万 |

当前批准来源没有独立的中文数学或技术文档桶。质量评审可把部分通用网页重新推荐到这些桶，但候选配额**不能证明最终领域目标已经满足**。

候选构建与全局汇总已经完成，明细如下。此表仍用候选构建的 Qwen 参考 tokenizer，**不是质量合格后的训练 token 数**。

| 域 | 文档 | 候选 Qwen token |
| --- | ---: | ---: |
| 中文通用 | 10,110 | 10,000,587 |
| 英文通用 | 7,667 | 8,000,886 |
| Python | 1,895 | 3,001,109 |
| Shell | 486 | 351,831 |
| 其他代码 | 466 | 650,085 |
| 英文数学 | 2,077 | 3,001,277 |
| **合计** | **22,701** | **25,005,775** |

全局精确重复、近重复各移除 0 篇；文档 ID 和 family/split 一致性检查均通过。原 P0-A 候选 2,066 篇全部包含在新池中，ID、规范化文本 SHA、token 数、family 和 split 一致，因而旧裁决可以按 ID 复用；其余 **20,635 篇是新增未评分候选**。旧 P0-A 的 2,066 篇里有 8 篇预筛排除、2,058 篇教师评分，不应把 2,066 都算成质量通过。

用冻结的自有 Byte-level BPE 32k tokenizer 重数，22,701 篇候选共 **26,974,286 文本 token（不含 EOS）**，其中 train 26,427,326、validation 249,852、test 297,108；抽查的无损还原失败数为 0。这个数仅用于候选规模判断，**不是可训练数据量**。

## 不依赖 DeepSeek 的质量步骤

1. 规范化时的字符、长度、重复行等规则已应用；候选构建时执行许可门、精确去重、固定抽样秩、MinHash 近重复与 family split 检查。汇总阶段再做跨来源去重。
2. 对全部候选跑低成本预筛：55 篇 / 86,469 Qwen token 命中可解释排除规则（依赖库路径、付费答案预览、彩票垃圾内容等）。这是候选级排除清单，**尚未创建新的训练集**。
3. Python 候选运行只解析、不执行的 Python 3 `ast.parse` 结构检查。1,895 篇中 1,653 篇通过，242 篇失败；失败项共 551,590 Qwen token。样例以 Python 2 `print` 和旧式异常语法为主。由于 P0-A 教师曾保留 17 篇不能按 Python 3 解析的旧代码，此检查先作为**单独分流标签**，不擅自修改旧版教师裁决。
4. V100 上本地 `qwen3.8-27b-uncensored` 服务以 temperature=0 对旧标签做了均衡分层校准。36 篇来自六个域，每域旧 keep/drop 各 3 篇；沿用原 DeepSeek v2 prompt/schema 的 SHA-256。34 篇输出通过 Schema，2 篇多次因输出截断仍未解决。

   | 本地判定 | 旧 keep 中（18 篇） | 旧 drop 中（16 篇成功输出） |
   | --- | ---: | ---: |
   | 原始 `decision=keep` | 17 | **7 误放行** |
   | 原始 `decision=drop` | 1 | 9 |
   | 严格门通过（keep、置信度≥0.85、质量≥3、完整度≥4、教育价值≥3、格式≥4、bucket≠drop） | 14 | **3 误放行** |

   因样本小且人为均衡，这些比例**不能**外推到实际总体；目前尚不足以证明“仅靠这个本地 Qwen 自动放行 20,635 篇新候选”可靠，也不能据此断定 Qwen 不可用。尤其 Shell 三篇旧 drop 均被原始输出判为 keep，值得复核原文和旧标签。该结论是与**旧教师标签的一致性**，不是绝对真值。校准成功输出耗费本地模型约 77,082 输入+输出 token；直接全量评审的吞吐/成本也未验收。
5. 最终如要冻结训练集，必须记录评分器版本、输出 Schema 成功率、与旧标注分歧、分层人工抽检、代码/数学验证、全局去重和自有 tokenizer 计数。未通过这些门的输出仍为候选。

## 判定与下一门槛

**候选扩池通过；质量自动放行尚未获证。** 目前没有 P0-B 1,000 万“已验收训练 token”，也不应从 2,500 万候选中直接打包训练 shard。下一步先复核模型分歧、在旧标注上定可解释的高精度预筛/分流规则，对新候选分层抽样做人工复核，再决定本地模型是只负责排序待审样本，还是经单独校准后负责某些高把握域。模糊样本可以弃用，但应统计各域接受率和最终自有 tokenizer token 数，避免总量/配比被误判。

服务器记录均位于 `/home/data/wangyue/datasets/small-agent-p0/p0_b_candidate_25m_v1/`：`candidates_consolidated_v1/manifest.json` 是候选主清单；`logs/candidate_own_tokenizer_counts.json` 是自有 tokenizer 计数；`prefilter_v1/prepare_manifest.json` 与 `code_python_ast_audit.json` 为规则审计；`calibration/local_qwen_calibration_report_v1.json` 与相邻 JSONL 为本地评审校准。配置位于仓库 `configs/data/p0_b_candidate_25m_v1.json`。配置中继承的 teacher v1 元数据**未用于本次本地校准**；本地校准实际使用的 v2 prompt/schema SHA 已单独记入报告。构建时的 2.2 GiB SQLite 临时索引在四个构建任务 exit=0、确认无相关进程后清理，候选、审计和日志保留。

## 分歧复核准备

把本地 Qwen 原始判定或严格门与旧裁决不一致的并集整理为 **11 篇**（原始判定分歧 8 篇、严格门分歧 7 篇），盲审输入为 `calibration/deepseek_disagreement_review_input_v1.jsonl`；旧/新判定只放在单独的 `calibration/deepseek_disagreement_review_manifest_v1.json`，不进入提问文本。V100 的密钥不在 SSH 会话环境变量，而在当前账号权限为 0600 的凭据文件；使用时仅注入子进程环境，不写入输入、结果或日志。

2026-09-24 用 `deepseek-flash`、原 v2 prompt/schema、temperature=0、thinking disabled 重评了这 11 篇：**11/11 Schema 成功**。重评结论与旧 DeepSeek 裁决一致 **9/11**，与 Qwen 原始判定一致 **5/11**，与 Qwen 严格门一致 **6/11**。发生两处旧标签翻转：一篇 JavaScript 模块由旧 drop 变为新 keep，与 Qwen 一致；一篇拼接的英文数学/商业计算文章由旧 keep 变为新 drop，也与 Qwen 一致。后者的库存周转示例确有输入数字与计算数字不一致的问题。其余 Shell 样本仍倾向 drop，原因是依赖外部二进制、硬编码绝对路径或只是环境修补包装脚本。

复核输出与逐篇对照保存在 `calibration/deepseek_disagreement_review_results_v1.jsonl`、`calibration/deepseek_disagreement_review_report_v1.json`。**旧标签原本也出自 DeepSeek，且此次沿用同一 rubric，因此这属于同模型稳定性/分歧分析，不构成独立真值，也不能把 9/11 外推为总体准确率。** 两处翻转提示旧标签不应被当作无误金标准；关键分歧仍需人工核看原文。
