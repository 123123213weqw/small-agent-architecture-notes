# P0：5000 万 token 自动预训练数据管线计划

> 状态：计划已冻结；P0-A（100 万 token 烟雾验收）的步骤 1、2 已开始实施。
>
> 目标：构建一套可重跑、可追溯、可扩展的预训练数据管线，并产出首个 5000 万训练 token 的验收版本。P0 验证数据工程，不用其绝对能力判断最终约 1B 模型上限。
>
> 当前训练路径：先供开源 Base 的继续预训练和数据消融使用；未来训练自有模型时复用规范文本，重新训练 tokenizer 并生成新 token shard。

## 1. 范围与非目标

P0 必须覆盖：

1. 数据源注册与固定 revision；
2. 流式抽取、规范化和确定性规则过滤；
3. DeepSeek 主教师的结构化质量评分；
4. V100 上本地 Qwen3.8-27B 的抽样复核和备用路径；
5. 有来源种子的技术、代码和数学合成；
6. 代码与数学的程序化验证；
7. 全库精确去重、近重复聚类和评测污染检查；
8. 按文档族切分 train/validation/test；
9. 使用固定 tokenizer 计数、打包和生成 manifest；
10. 用短训练验证 shard、loss 和 checkpoint 恢复。

P0 不负责：

- 证明长期记忆结构优于 FIFO 或 Transformer；
- 确定最终约 1B 模型的数据规模和完整混合比例；
- 将 400 条审计摘录当作预训练语料；
- 让教师模型代替代码执行、数学验证或训练消融；
- 只保存 token ID 而丢弃规范文本。

## 2. 冻结的产出规模

P0 的 5000 万是**去重、验证并用学生 tokenizer 计数后的训练 token**。validation/test 另外保留约 100 万至 200 万 token，不计入该数字。

| 数据桶 | 训练 token | 说明 |
|---|---:|---|
| 中文通用 | 1200 万 | 完整中文正文、说明和知识文本 |
| 英文通用 | 1000 万 | 英文教育与知识文本 |
| 中英技术文档 | 800 万 | API、工具、系统和编程说明 |
| Python | 700 万 | 完整、有意义且可追溯的 Python |
| Shell | 200 万 | Shell、命令行和自动化脚本 |
| 其他代码 | 100 万 | 少量 JavaScript、TypeScript、JSON 等 |
| 中文数学 | 300 万 | 完整题目、教材解释和例题 |
| 英文数学 | 200 万 | 英文数学教育文本 |
| 教师合成 | 500 万 | 技术 200 万、代码 150 万、数学 150 万 |
| **总计** | **5000 万** | 按 Qwen3-0.6B Base tokenizer 统计 |

这些比例只冻结到 P0 验收，不自动继承为最终预训练比例。

## 3. 候选来源与批准门

### 3.1 P0-A 当前实施状态（2026-09-23）

- 来源总表：`configs/data/p0_sources_v1.json`；Schema：`schemas/p0_source_registry_v1.json`。
- 每个来源固定 dataset、40 位 revision、远端文件、落盘相对路径、许可状态、实际字节数和 SHA-256。
- 来源适配器位于 `configs/data/sources/*.json`，只负责把各源字段映射到统一记录；Python、Shell、JavaScript 使用独立适配器和数据桶。
- 适配器执行前会校验真实 Parquet schema。正文列不存在或所有定位列都不存在时直接失败，不允许静默使用错误字段。
- FineWeb-Edu 首个 shard 已下载并校验：726,000 行、726 个 row group；1,000 行双跑输出 SHA-256 完全一致，保留 999 行，1 行因超过长度上限被规则拒绝。
- FineWeb2 中文首个 shard 也已下载并通过真实 schema 校验：1,973,000 行、1,973 个 row group。
- Stack-Edu 的 Python、Shell、JavaScript ID shard 已下载并校验，但真实字段只有 `blob_id`、仓库、路径、评分和许可元数据，**不含代码正文**。官方说明正文需要再从 Software Heritage S3 解析；当前机器没有对应 AWS 凭据，因此三个直接正文适配器标记为 `rejected`，不能把 ID 元数据误当训练文本。后续需要选择“按官方方式解析并缓存正文”或“改用带正文的批准来源”。
- FineMath 首个 shard 已下载并通过真实 schema 校验：167,232 行、168 个 row group，正文列、URL、语言和质量分字段均可用。
- 替代代码源使用 `codeparrot/github-code-clean` 的固定 shard。该 shard 含 126,925 个完整代码文件，其中 Python 8,028、Shell 1,537、JavaScript 12,315；适配器按语言和保守许可白名单分别产出三个数据桶。曾下载验证的 `smollm-corpus/python-edu` 同样只有 Software Heritage ID，没有正文，因此不接入直接文本管线。

P0-A 原始候选暂存在 V100：

```text
/home/data/wangyue/datasets/small-agent-p0/p0_1m_smoke_v1/
```

这只是拉取和管线验收区。P0 全量产物再复制到 JFS 的正式目录，不把原始大文件提交 Git。

第一批候选：

| 数据桶 | 候选来源 | P0 用法 |
|---|---|---|
| 英文通用/技术 | FineWeb-Edu | 流式抽取高质量英文正文 |
| 中文通用/技术 | FineWeb2 `cmn_Hani` | 中文正文与技术内容 |
| 代码 | Stack-Edu（由 The Stack v2 筛选） | P0 直接使用 Python、Shell、JavaScript 子集；后续保留回到 The Stack v2 重筛的能力 |
| 数学 | FineMath | 数学教育网页和例题 |
| 合成 | 高分训练集种子 | DeepSeek/Qwen 有来源生成 |

每个来源必须先写入 `configs/data/sources/*.json`，至少记录等价字段：

```yaml
source_id: fineweb2_zh
dataset: HuggingFaceFW/fineweb-2
subset: cmn_Hani
revision: null
domain: general_zh
language: zh
license_status: pending
access: public
streaming: true
```

只有满足以下条件才把 `license_status` 改为 `approved`：

- 固定数据集 revision/commit；
- 记录数据卡和许可条款；
- 正文确实可取得；
- 来源字段可以追溯；
- 小样本确认提取质量；
- 没有把测试 split 混入训练。

**当前状态（2026-09-24）**：`fineweb_edu`、`fineweb2_zh`、`finemath` 与三个
`github_code_clean_*` 已按上述条件逐条核对并批准，证据见
[P0 来源许可批准记录 v1](p0-source-license-approval-v1.zh.md)。`stack_edu_*` 仍为
`pending`：该 revision 的数据卡没有许可字段，且正文需另从 Software Heritage S3 取得，
当前适配器不含正文。许可批准只是其中一个门，不替代第 14 节的验收条件。

## 4. 存储结构

大数据、token shard 和 checkpoint 放在 L40 可访问的 JFS 工作区，不提交 Git：

```text
small-agent-data/p0_50m_v1/
├── raw/                  # 原始快照，只读
├── normalized/           # 统一 UTF-8 规范文本
├── filtered/             # 规则过滤后候选
├── teacher_requests/     # 给教师的不可变请求 shard
├── teacher_outputs/      # 原始结构化响应
├── synthetic/            # 未验证的教师合成数据
├── verified/             # 自动验证通过的数据
├── deduped/              # 全库去重结果
├── splits/               # train/validation/test 文档
├── tokenized/            # 按 tokenizer 版本保存
├── packed/               # 按训练框架和长度打包
├── manifests/            # 输入、配置、哈希和统计
└── reports/              # 人工可读报告和样本
```

Git 仓库只保存配置、Schema、处理脚本、测试和小样本。

## 5. 统一文档记录

规范化文档使用 Parquet/Arrow 作为大规模主格式；JSONL 只用于小样本、教师请求和调试。每条至少包含：

```json
{
  "document_id": "sha256:...",
  "source_id": "fineweb2_zh",
  "source_revision": "...",
  "source_locator": "...",
  "family_id": "url-or-repository-or-problem-family",
  "domain": "general_zh",
  "language": "zh",
  "text": "...",
  "raw_sha256": "...",
  "normalized_sha256": "...",
  "license_status": "approved",
  "pipeline_version": "p0_50m_v1"
}
```

`document_id` 与内容哈希不能依赖运行时间。`family_id` 用于防止同一网页族、仓库或题目族跨训练/测试集合。

## 6. 候选池规模

为了在严格筛选和去重后得到目标 token，P0 先流式建立更大的候选池：

| 类别 | 最终目标 | 初始候选 token |
|---|---:|---:|
| 通用与技术 | 3000 万 | 6000 万至 9000 万 |
| 代码 | 1000 万 | 2000 万至 3000 万 |
| 数学 | 500 万 | 1000 万至 1500 万 |
| 合成 | 500 万 | 生成 700 万至 1000 万 |

候选池达到上限后停止流式读取；不足时追加新 shard，而不是覆盖旧结果。

## 7. 确定性规则清洗

### 7.1 通用与技术文本

过滤空白、乱码、正文过短、导航/广告占主体、异常字符、段落高重复、SEO 堆积、只有标题、明显预览残缺和提取损坏。保留中文标点、代码块、公式、换行和列表结构。

### 7.2 代码

首轮聚焦 Python、Shell，少量保留 JavaScript/TypeScript/JSON。过滤 vendor、`node_modules`、lock 文件、minified 文件、构建产物、二进制、超大生成文件、只有 import 的文件和重复模板。原始代码只做静态解析；不得把未知仓库文件直接作为本机脚本执行。

### 7.3 数学

保留有完整上下文的题目、教材式解释、公式和例题。过滤公式损坏、只有答案、无意义 LaTeX、页面模板占主体、付费预览和明显自相矛盾的文本。

每个过滤器只写 `keep/drop`、`reason_code` 和证据，不修改或删除 `raw/`。

## 8. 教师分工

### 8.1 DeepSeek：主教师

DeepSeek 负责候选语义质量评分、技术改写、代码/数学合成和困难样本分析。评分使用非思考模式、低随机性和严格 JSON Schema；生成允许适度多样性。

每次调用保存：

```json
{
  "request_id": "sha256:model+prompt+input",
  "provider": "deepseek",
  "requested_model": "deepseek-flash",
  "returned_model": "...",
  "system_fingerprint": "...",
  "prompt_sha256": "...",
  "temperature": 0,
  "reasoning_effort": "low",
  "input_sha256": "...",
  "input_tokens": 0,
  "output_tokens": 0
}
```

API 别名可能升级，因此不得只记录 `deepseek-flash`；必须保留响应返回的版本信息和 fingerprint。密钥只从环境变量或系统密钥存储读取，不写入请求文件、日志或 Git。

质量评分输出：

```json
{
  "document_id": "...",
  "keep": true,
  "quality": 4,
  "completeness": 5,
  "educational_value": 4,
  "format_integrity": 5,
  "reason_codes": []
}
```

建议 P0 门槛：`keep=true`、完整性和格式至少 4/5、教育或技术价值至少 3/5。无法稳定判断的样本直接剔除。

### 8.2 Qwen3.8-27B：本地复核与备用

V100 上的 Qwen3.8-27B 负责：

- 对 DeepSeek 保留和剔除结果各分层抽查 5% 至 10%；
- 检查合成内容的样式单一和模板化；
- 生成约 20% 的合成对照数据；
- DeepSeek API 不可用时继续处理；
- 为后续轻量分类器提供第二套标签。

模型分歧的处理顺序是：确定性规则优先；代码/数学程序验证优先；仍无法确定则 P0 直接丢弃。

## 9. 合成数据

合成种子只能来自 `train` 中的高质量文档；validation/test 不能产生训练合成数据。所有合成结果继承种子来源和 `family_id`。

| 类型 | 验收 token | 生成要求 |
|---|---:|---|
| 技术教材 | 200 万 | 自包含概念、边界、示例和常见错误 |
| 代码教材/练习 | 150 万 | 问题、解释、代码和测试 |
| 数学教材/例题 | 150 万 | 完整题目、解法、答案和可验证表达式 |

合成 token 按验证、去重后的文本统计。为得到 500 万验收 token，允许生成 700 万至 1000 万原始 token。

### 9.1 代码验证

合成代码依次通过 JSON Schema、AST/语法、隔离进程、资源限制和单元测试。只有对应测试全部通过的代码进入 `verified/`。

### 9.2 数学验证

结构化输出必须包含题目、解答、最终答案和验证表达式。使用 Python/SymPy 检查数值或代数一致性、变量定义和最终答案。无法程序化确认的样本不进入 P0 的“可验证数学”池。

教师评分不能覆盖程序验证失败。

## 10. 去重、污染与切分

顺序固定为：

1. 规范化正文 SHA-256 精确去重；
2. 段落高重复检测；
3. 文档 MinHash/LSH 近重复聚类；
4. 代码按仓库和文件内容去重；
5. 数学题干规范化与题目族去重；
6. 合成数据与种子、其他合成数据交叉去重；
7. 与冻结评测集做 n-gram 和近重复污染检查；
8. 按 `family_id` 哈希切分。

切分比例暂定：

```text
train       98%
validation   1%
test         1%
```

合成内容继承种子 split；只有 train split 允许进入合成训练池。

## 11. Tokenize 与打包

P0 使用 `Qwen/Qwen3-0.6B-Base` tokenizer 计数，序列长度为 2048。每篇文档末尾插入 EOS，并保留 `document_id` 与 token span。Canonical token shard 使用 Parquet/Arrow；训练框架需要的 `.bin/.idx` 等格式由独立适配器生成。

Qwen tokenizer 只作为数据阶段的稳定计数器，不作为 40M/120M 小模型的最终词表；其 151,665 词表会让小模型的 embedding 参数占比过高。最终 tokenizer 通过 `configs/tokenizer/p0_tokenizer_bakeoff_v1.json` 固定为两个 32,768 词表候选：

- A：无归一化的 ByteLevel-BPE，完整 256 字节初始字母表；
- B：SentencePiece-BPE，identity 归一化、禁止压缩连续空白并开启 byte fallback。

两者使用同一份一亿字符确定性训练样本和同一组特殊 token。只做 tokenizer 静态 bake-off，不分别训练语言模型。代码、连续空白、换行或 Unicode 不能精确 round-trip，或出现未知 token 的候选直接淘汰；其余候选按中文、英文、代码和数学的 token 压缩率比较。胜出版本冻结为 `tokenizer_v1`，随后重新统计 P0 token 数并打包。

#### Tokenizer 训练语料抽取（2026-09-24）

`scripts/p0_tokenizer_corpus.py` 从已规范化 Parquet 中只读取 `rule_keep=true` 且 `split=train` 的记录。每个桶先统计可用字符，再以 `SHA256(seed || document_id)` 做确定性概率过采样，按哈希顺序精确填满字符配额。全局删除正文哈希完全相同的记录，每个 family 最多保留 4 篇，并排除正文中与保留特殊 token 字符串相撞的记录。Tokenizer 语料只影响词表频率，不直接作为语言模型训练集，所以这里不运行成本较高的全量近重复检测；最终预训练语料仍维持独立的近重复门。

Shell 规则通过且属于 train split 的正文总共只有约 164 万字符，无法满足原定 500 万字符，因此没有复制或重复采样，而是把目标调整为 Shell 150 万、Python 2,200 万、其他代码 650 万，代码总量仍保持 3,000 万字符。

冻结输出位于：

```text
/home/data/wangyue/datasets/small-agent-p0/p0_1m_smoke_v1/tokenizer_corpus_v1/
```

结果为 33,778 篇、100,013,292 个 Unicode 字符、153,306,067 个 UTF-8 字节；各桶均达到目标，因完整保留最后一篇文档而总计仅超出 13,292 字符。另从非 train split 按固定哈希保留 875 篇、4,008,735 字符作为 tokenizer 静态评测集。抽取中有 4 篇因特殊 token 字符串碰撞被排除，80 篇因 family 上限被排除。Manifest SHA-256 为 `7567a1bcaf385afca752371b7f6b66faffc4c53cc00801ac3f00b774af0ea865`。上游来源许可仍为 `pending`，因此该语料只能进入内部 bake-off，不能标记为正式训练批准。

必须保留：

- 规范文本；
- tokenizer 名称、revision 和文件哈希；
- 每个 shard 的输入文档列表、token 数和 SHA-256；
- 数据顺序随机种子；
- pack 后丢弃/填充 token 统计；
- train/validation/test 独立 manifest。

普通预训练文档允许通过 EOS 打包到同一序列；后续记忆 episode 不允许跨 episode 拼接或泄漏状态。

## 12. 三段执行

### P0-A：100 万 token 冒烟

完整跑通采集、规则、DeepSeek、Qwen 抽查、合成、验证、去重、切分、tokenize 和 pack。同配置重跑必须得到相同文档顺序和 shard 哈希。

#### 已实现的离线候选池接口

`scripts/p0_candidate_pipeline.py` 将离线部分拆成三个可恢复阶段：

```bash
# 1. 原始来源 -> 规范化 Parquet；保留通过和拒绝记录及原因
python scripts/p0_candidate_pipeline.py normalize \
  --registry configs/data/p0_sources_v1.json \
  --pipeline-config configs/data/p0_1m_smoke_v1.json \
  --repo-root . \
  --raw-root "$DATA_ROOT/raw" \
  --output-root "$DATA_ROOT" \
  --skip-unvalidated

# 2. 全局精确去重、family 统计、MinHash 近重复、确定性抽样和精确 token 计数
python scripts/p0_candidate_pipeline.py build \
  --registry configs/data/p0_sources_v1.json \
  --pipeline-config configs/data/p0_1m_smoke_v1.json \
  --data-root "$DATA_ROOT" \
  --tokenizer-json "$TOKENIZER_DIR/tokenizer.json"

# 3. 合并独立构建的网页、数学、代码池，再做跨池精确/近重复与 family 检查
python scripts/p0_candidate_pipeline.py consolidate \
  --pipeline-config configs/data/p0_1m_smoke_v1.json \
  --data-root "$DATA_ROOT" \
  --candidate-dir candidates_web_partial \
  --candidate-dir candidates_math_partial \
  --candidate-dir candidates_code_partial \
  --output-dir-name candidates_consolidated_v1
```

抽样秩固定为：

\[
r_i=\operatorname{SHA256}(\text{pipeline seed}\Vert\text{document id}).
\]

全局精确重复只保留抽样秩最小的记录。随后按秩从小到大执行 MinHash LSH 候选召回和真实 shingle Jaccard 检查；若近重复被拒绝，继续读取后续秩，直到达到对应数据桶的 token 配额。因此近重复清理不会造成候选池缺额，也不依赖原始 shard 顺序。

P0-A 暂定候选配额总计 225 万 token：中文网页 80 万、英文网页 70 万、Python 35 万、Shell 10 万、其他代码 5 万、英文数学 25 万。中文网页和英文网页中包含的技术/数学内容，后续由教师的 `recommended_bucket` 再分类。

许可门默认严格：`pending` 不会进入候选池。`--allow-pending-license` 只允许做管线冒烟，并在 manifest 中写入 `training_eligible=false`，不能作为训练批准。

#### P0-A 合并结果（2026-09-23）

V100 上的三个独立候选池已合并到：

```text
/home/data/wangyue/datasets/small-agent-p0/p0_1m_smoke_v1/candidates_consolidated_v1/
```

合并器逐个校验输入 manifest 与候选 shard 的字节数、SHA-256 和行数，并检查 run、管线配置、tokenizer、文档 ID、family 与 split 一致性。结果为：

- 输入和保留均为 2,066 篇、2,264,045 token；
- 跨池精确重复 0，跨池近重复 0；
- 2,054 个 family，最大 family 6 篇；
- 文档 ID 冲突 0，family 跨 split 冲突 0；
- `unified_index.parquet` 的 2,066 个位置引用已逐条复核；
- 来源许可仍为 `pending`，因此 `training_eligible=false`。

输出保留按桶 Parquet、统一无正文索引 `unified_index.parquet`、去重审计表 `dedup_exclusions.parquet` 和完整 `manifest.json`。三个输入建立时来源注册表曾追加新来源，所以它们的 registry 哈希不同；合并 manifest 原样保存全部三个哈希，而不是错误地声称它们来自同一版注册表。

#### DeepSeek 初筛试跑（2026-09-23）

合并池先经过高精度廉价规则，命中 8 篇、10,147 token：`node_modules/vendor` 3 篇、付费答案预览 2 篇、彩票关键词垃圾 2 篇、嵌入式推荐反馈控件 1 篇。随后按固定哈希从六个桶各抽 50 篇，共 300 篇，使用 `deepseek-flash`、temperature 0、thinking disabled 和 `p0_teacher_quality_v2` Schema 评分。

严格门槛为：模型判保留、置信度至少 0.85、总体质量至少 3、完整性至少 4、教育价值至少 3、格式至少 4。结果：

| 桶 | 保留 | 剔除 | 保留率 |
| --- | ---: | ---: | ---: |
| 其他代码 | 35 | 15 | 70% |
| Python | 38 | 12 | 76% |
| Shell | 27 | 23 | 54% |
| 英文通用 | 23 | 27 | 46% |
| 中文通用 | 3 | 47 | 6% |
| 英文数学 | 7 | 43 | 14% |
| **总计** | **133** | **167** | **44.3%** |

保留文档对应 155,759 个候选 token，剔除对应 152,180 个；API 共使用 585,188 token。299 篇得到有效 Schema，1 篇多次返回非法推荐桶，按严格策略自动剔除。主要拒绝原因是信息量低、模板噪声、SEO、代码过短/不完整和无关正文拼接。

第一次长文请求只发送 22,000 字符的首中尾片段，模型将管线插入的“省略”标记误判成原文损坏，造成明显假阴性。试跑已将 9 篇长文改为全文（本批最长约 60,000 字符）并只重跑变更请求；最终 300 篇全部以全文作为输入。正式全量阶段不得把传输截断标记当成原文的一部分；超出上下文的文档必须使用显式的分块聚合协议。

这轮说明当前中文网页和 FineMath 原始候选质量明显不足，不能直接进入训练集；代码桶相对可用。它只用于确认判分口径，正式全量结果如下。

#### DeepSeek 全量严格初筛（2026-09-23）

对 2,066 篇合并候选先执行同一廉价规则，剔除 8 篇；其余 2,058 篇全部发送全文，最长正文 83,466 字符，没有人为截断或插入省略标记。经过首次评分和两轮失败重试后，2,037 篇得到合法 Schema，21 篇仍无法满足 Schema，按严格策略直接剔除。

| 桶 | 候选篇数 | 保留篇数 | 保留率 | 保留 token |
| --- | ---: | ---: | ---: | ---: |
| 其他代码 | 59 | 39 | 66.1% | 44,399 |
| Python | 231 | 183 | 79.2% | 326,670 |
| Shell | 146 | 84 | 57.5% | 77,178 |
| 英文通用 | 688 | 301 | 43.8% | 331,767 |
| 中文通用 | 781 | 67 | 8.6% | 100,316 |
| 英文数学 | 153 | 16 | 10.5% | 63,476 |
| **总计** | **2,058** | **690** | **33.5%** | **943,806** |

连同廉价规则剔除项计算，完整候选池为保留 690 篇、剔除 1,376 篇。合法结果记录的 API 用量为 4,090,452 token；失败重试的响应未计入该数字。最常见拒绝原因是低信息、模板噪声、SEO、无关拼接和不完整。

结论不是放宽门槛，而是补来源：代码量已充足；英文通用基本可用；中文通用和数学明显缺额，应更换或扩充来源后走同一严格门。690 篇仍只是 DeepSeek 单教师候选，进入训练 shard 前需要许可批准和最终去重；Qwen3.8-27B 改为在扩大数据规模前做分层抽查，而不是逐篇重复评分。

#### 保留集冻结与 DeepSeek 合成补充（2026-09-24）

人工快速浏览未发现成片质量问题后，不再对 690 篇逐条追加第二教师评分。`scripts/p0_freeze_accepted.py` 将教师结果回接到原始 Parquet，生成 `accepted_real_v1`：

- 接受 690 篇、943,806 token；
- 拒绝审计表完整记录廉价规则剔除 8 篇、教师门剔除 1,368 篇；
- 接受 shard 保留原始来源、正文哈希、family、split、教师分数、证据和决策记录哈希；
- 690 篇来源许可仍为 `pending`，所以该产物保持 `training_eligible=false`。

为补足约 5.6 万 token，DeepSeek 以 temperature 0.6、thinking disabled 生成 24 个中文技术主题和 24 个中文数学主题。生成结果经过以下自动门：

1. 严格 JSON Schema 与完整字段；
2. 中文正文长度、元话语、链接、占位符和 Markdown 围栏检查；
3. Python AST、JSON 解析和 `bash -n` 语法检查，不执行生成代码；
4. 数学附录的整数/分数算式由受限 AST 精确求值，不使用 `eval`；
5. 与 690 篇真实文本及合成文本之间重新做精确去重和 MinHash 近重复检查；
6. 对自动验证通过项再使用 temperature 0 的严格质量门评分。

48 篇均成功生成；自动验证保留 35 篇、66,423 token，主要拒绝原因是代码围栏无法安全解析、算术校验失败或存在元话语。严格质量门通过 34 篇。按固定哈希和分桶配额最终选择 31 篇：

| 合成桶 | 篇数 | token |
| --- | ---: | ---: |
| 中文技术 | 17 | 29,793 |
| 中文数学 | 14 | 28,172 |
| **总计** | **31** | **57,965** |

`scripts/p0_merge_smoke_dataset.py` 对真实集和合成集再次执行全局精确/近重复与 family split 检查，冻结得到 `p0_1m_frozen_v1`：721 篇、1,001,771 token；精确重复 0、近重复 0、family 跨 split 冲突 0。文档切分为 train 706、validation 9、test 6。

该冒烟集由 `accepted_real_v1` 派生，而后者 690 篇真实文档的 `license_status` 字段已逐行固化为 `pending`，所以在重跑前它**仍不允许标为正式训练可用**。来源许可已于 2026-09-24 批准（见 [P0 来源许可批准记录 v1](p0-source-license-approval-v1.zh.md)），但 `training_eligible` 是由产物内容计算出来的（`p0_freeze_accepted.py`：`set(license_status) <= {"approved"}`），必须按同一实现重跑 `normalize → build → consolidate → freeze` 才能得到 `training_eligible=true` 的新 manifest；**不得手工修改既有 manifest 的 `license_gate` 字段**，重跑后还要逐项比对文档 ID 与 shard 行数，确认候选集未因许可门变化而改变。

### P0-B：1000 万 token 试产

测量教师吞吐、API 失败率、输出解析率、过滤率、去重率、合成验证通过率、JFS I/O 和实际成本。任何阶段不得靠手工修改结果文件完成。

### P0-C：5000 万 token 正式验收

只有 A/B 通过后才扩展。最终产出 `p0_50m_v1` 的不可变 manifest、统计报告、人工抽样包和训练 shard。

## 13. V100、L40 与 JFS 分工

```text
L40/JFS：来源、规范文本、请求 shard、结果和最终数据
V100：本地 Qwen3.8-27B 评分/生成 worker
DeepSeek API：主教师评分和高价值合成
L40：tokenize、数据消融和模型训练
Mac：代码、配置和监控，不作为大文件中转站
```

若 V100 不能直接访问 JFS，使用现有 V100 到 L40 的本地 SSH 通道按不可变 shard 拉取请求并回传结果。每个结果 shard 写临时文件，完成校验后原子重命名；重启时通过 manifest 跳过已完成 shard。

## 14. 验收条件

P0 完成必须同时满足：

- 训练 token 为 5000 万，允许误差不超过 1%；
- validation/test 另有 100 万至 200 万 token；
- 100% 样本可追溯到来源或合成种子；
- 最终精确重复为零，近重复有聚类统计和抽样检查；
- 教师输出全部通过 Schema，失败请求有显式状态；
- 合成代码全部通过其声明的测试；
- 标记为可验证数学的样本全部通过程序验证；
- train/validation/test 没有 `family_id` 交叉；
- 冻结评测集无已知污染命中；
- 同输入、配置和种子重跑得到相同输出哈希；
- 每个数据桶随机抽查至少 100 条；
- 使用最终 shard 短训 500 至 1000 step，无 NaN，loss 合理下降，checkpoint 恢复后曲线连续。

## 15. P0 后的数据消融

P0 通过后，使用同模型、同 tokenizer、同 token、同优化器和同评测构建：

| 组 | 数据策略 |
|---|---|
| A | 只做确定性规则清洗 |
| B | A + DeepSeek 语义过滤 |
| C | B + 10% 教师合成 |
| D | B + 20% 教师合成并提高代码/数学比例 |

比较中英文验证 BPB、代码测试通过率、基础数学准确率、重复生成、训练稳定性和继续预训练后的能力变化。先单种子筛选，再对最后两个候选进行多种子确认。教师分数本身不作为数据方案胜负结论。

## 16. 实现顺序

1. 冻结来源注册 Schema、文档 Schema 和 P0 配置；
2. 实现流式抽取与规范化；
3. 实现规则过滤和原因统计；
4. 实现 DeepSeek 批处理客户端、缓存、重试和费用统计；
5. 实现 V100 Qwen worker 和 shard 交换；
6. 实现技术、代码、数学生成模板；
7. 实现代码与数学验证器；
8. 接入精确/近重复和评测污染检查；
9. 实现按族切分、tokenize、pack 和 manifest；
10. 完成 P0-A，再按验收门扩到 P0-B/P0-C；
11. 运行短训练和 A/B/C/D 数据消融；
12. 使用积累的教师标签训练轻量质量分类器，为数十亿 token 阶段替代逐篇教师调用。

## 17. 行业参考

- [FineWeb/FineWeb-Edu](https://arxiv.org/abs/2406.17557)：网页抽取、规则、MinHash 去重和教育质量过滤；
- [FineWeb2](https://github.com/huggingface/fineweb-2)：多语言、分语言阈值和中文数据；
- [Dolma](https://allenai.github.io/dolma/)：可复现多来源预训练数据工具链；
- [DataComp-LM](https://proceedings.neurips.cc/paper_files/paper/2024/file/19e4ea30dded58259665db375885e412-Paper-Datasets_and_Benchmarks_Track.pdf)：用受控小模型训练比较数据方案；
- [SmolLM](https://huggingface.co/blog/smollm)：FineWeb-Edu、代码教育分类器和合成教材；
- [Cosmopedia](https://huggingface.co/blog/cosmopedia)：以真实网页为种子的规模化合成教材；
- [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu)：P0 使用的教育质量筛选代码；
- [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2)：Stack-Edu 的上游代码来源、许可与退出机制；
- [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath)：数学教育网页筛选；
- [DeepSeek Responses API](https://api-docs.deepseek.com/api/create-response/)：结构化输出、模型参数和响应元数据。
