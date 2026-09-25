# P0-A 真实语料重建与新增合成语料审计 v1

日期：2026-09-24

## 范围与目录

- 原始 P0-A：`/home/data/wangyue/datasets/small-agent-p0/p0_1m_smoke_v1/`，保留不动。
- 新重建：`/home/data/wangyue/datasets/small-agent-p0/p0_1m_approved_rebuild_v1/`。
- 本轮只重建已批准的六个真实来源：FineWeb-Edu、FineWeb2 中文、FineMath、GitHub Code Clean 的 Python/Shell/JavaScript。三个 `stack_edu_*` 来源仍不进入数据集。
- 新生成的 `p0_gen_verified_v1` 单独审计，**不并入真实语料**。`p0_10m_v1` 只有两个完整 FineWeb-Edu 原始分片及一个未完成的 FineWeb2 `.part` 文件，也不进入本轮。

## 真实语料的重建规则

1. 沿用原始 raw 文件、相同规范化实现及相同 P0-A 配置，只改变来源配置中已批准的许可状态；重跑 `normalize → build → consolidate → freeze`，不得改写旧产物的 `license_gate`。
2. 候选选择沿用原先的 Qwen3-0.6B tokenizer，SHA-256 为 `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`，以保证原有候选选择及教师评分能够逐项对齐。
3. 只有在新旧候选的文档 ID、规范化内容 SHA-256、token 数完全一致时才复用原有教师裁决；若不一致则停止，不根据 ID 猜测沿用标签。
4. 最终模型输入用冻结的自有 Byte-level BPE 32k tokenizer 编码，SHA-256 为 `9d0b99ab9cf20d337217c96a5b39e9ff87590eb462abcbcc8f1119c3d43911a7`；它与候选配额使用的 tokenizer 不同，两个 token 数不可直接混用。

六个来源重新规范化后的保留文档数与旧版一致：FineWeb-Edu 725,771、FineWeb2 中文 1,931,068、FineMath 161,503、Python 4,623、Shell 695、JavaScript 8,287。数学和代码部分候选集的新旧 ID、内容哈希及 token 数也已完全一致。

## 新增合成语料独立审计

审计程序：`scripts/p0_audit_generated.py`。机器可读结果位于新重建目录的 `logs/new_generated_audit.json`。

`p0_gen_verified_v1` 的 manifest 报告 1,533 篇、617,834 个 Qwen tokenizer token；按自有 32k tokenizer 重算为 **590,171 token**（不含 EOS）。其中 `gen_verified_v2` 945 篇，`distill_reasoning_v1` 588 篇。两个 Parquet shard 的 SHA-256、全部 1,533 篇正文哈希、文档 ID 唯一性、tokenizer 无损往返和 family split 检查通过；与旧 P0-A 721 篇相比没有 ID 或完全相同正文的交集。这里仅检查了精确重复，**不能据此宣称没有语义近重复**。

但是打包后的 Python 代码块存在严重结构错误：

| 来源 / 域 | Python 代码块 | `ast.parse` 通过 | 失败 |
| --- | ---: | ---: | ---: |
| `distill_reasoning_v1/code_python` | 300 | 20 | **280** |
| `gen_verified_v2/code_python` | 481 | 330 | **151** |
| `gen_verified_v2/math_zh` | 464 | 432 | **32** |

原始 `p0_distill_v1/accepted.jsonl` 的 300 个代码答案 **300/300** 可通过语法解析，而打包后只有 **20/300**，说明问题发生在已验证答案到打包语料之间。抽样显示多级缩进被压成相同的单空格；因此原始答案的执行验证**不能证明最终 Parquet 代码仍可执行**。另有 945 篇知识点语料的原始 `verify` 字段 **945/945** 可语法解析，但打包后仍有代码块解析失败。

已定位到直接原因：`/home/wzu/train-data/pack_p0_shards.py` 的 `norm_text()` 第 34 行使用 `re.sub(r"[ \t]+", " ", t)` 全文压缩连续空格，并在第 140 行应用于每篇正文。它将 Python 代码块中表示不同嵌套层级的 4 空格、8 空格都变成 1 空格。打包脚本没有对**打包后的**代码块重新运行语法检查，因此原始验证虽然通过，产物仍可损坏。

旧版结论：`p0_gen_verified_v1` 保持隔离，不标记为训练就绪，不并入真实语料。该旧版的 `license_gate.training_eligible` 为 `null`，不能将其当作正式训练批准。

### 缩进修复：`p0_gen_verified_v2`

2026-09-24 使用 `scripts/p0_pack_generated_shards_v2.py`（部署副本 `/home/wzu/train-data/pack_p0_shards_v2.py`）从原始 `corpus.jsonl` 与 `accepted.jsonl` **重新**打包到 `/home/data/wangyue/datasets/small-agent-p0/p0_gen_verified_v2/`；没有覆盖旧版。新版 `norm_text()` 只统一换行并去掉最外侧空行，保留代码缩进，且写 Parquet 前对最终文本里的每个 Python fenced block 执行 `ast.parse`，失败即停止。

新版仍为 1,533 篇；Qwen tokenizer 计 **623,169 token**，自有 32k tokenizer 计 **590,246 token**（不含 EOS）。最终 Parquet 中的 **1,245/1,245** 个 Python 代码块全部可语法解析：蒸馏代码 300/300、知识点代码 945/945。逐篇比对确认 **1,533/1,533** 的最终正文与原始记录在仅统一换行后的文本完全一致；蒸馏 `doc` 的代码块也与此前验证的 `code` 字段 **300/300** 一致。最终 shard 哈希、正文哈希、ID 唯一性、split、tokenizer 往返及与旧 P0-A 的精确去重检查均通过。机器可读审计：`p0_1m_approved_rebuild_v1/logs/new_generated_v2_audit.json`。

这些结果修复了**打包导致的代码结构损坏**，不等同于已完成数学推导的逐步正确性审计、语义近重复检查或训练许可决策。新版仍与已批准真实语料隔离，`license_gate.training_eligible` 仍为 `null`，本轮不并入训练流。

## 完成状态与核验

重建已完成，所有阶段位于上述新目录内，临时 SQLite 索引 `work/` 已清理；旧目录未覆盖。

| 阶段 | 结果 |
| --- | --- |
| 三组候选 | 网页 1,472 篇、数学 155 篇、代码 439 篇；逐组与旧版的文档 ID、正文哈希、token 数完全一致；新行的许可状态均为 `approved` |
| 汇总候选 | 2,066 篇、2,264,045 个 Qwen tokenizer token；全体的文档 ID、正文哈希、token 数、family、split 与旧版完全一致；精确及近重复移除数均为 0；`training_eligible=true` |
| 教师裁决复用 | 因候选内容逐项一致，沿用旧的 2,058 条教师裁决和 8 条预筛排除；未重新调用付费模型 |
| 真实语料冻结 | `accepted_real_v1`：690 篇、943,806 个 Qwen tokenizer token；与旧版的 ID、正文哈希、token 数、family、split、推荐域完全一致，唯许可状态由 `pending` 变为 `approved`；`training_eligible=true` |
| 自有 tokenizer 编码 | `approved_real_tokens_v1`：train 675 篇 / 1,051,265 token，validation 9 篇 / 15,329 token，test 6 篇 / 15,902 token；总计 **1,082,496 token（含每篇 EOS）**，均为 `uint16` |

编码产物已独立核对三个 `.npy` 的 SHA-256、长度、文档数和每篇末尾的 EOS。打包阶段还逐篇检查了自有 tokenizer 的无损往返。新目录下的 `logs/` 保存各阶段输出、`rebuild_verification.json` 和 `new_generated_audit.json`。

该产物只是约 **108 万 token 的 P0-A 真实语料小样**，不是 5,000 万 token 数据集，更不能据此评价 1B 模型能力。旧版 31 篇合成补齐数据和这次新增的 1,533 篇合成数据都没有混入本次正式真实语料流。
