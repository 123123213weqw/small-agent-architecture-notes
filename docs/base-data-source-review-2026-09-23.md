# 四个基础语料候选的首次核查（2026-09-23）

> 这是官方数据卡、Hub 元数据和极少量 Dataset Viewer 行的桌面核查，不是正式的随机抽样质量评估。未下载大文件，未批准任何数据进入训练。当前登录的 Hugging Face 账号为 `wangyue114514`。

## 概览

| 数据集 | 内容与规模（发布方口径） | 当前账号访问 | 初步判断 |
|---|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 英文教育类网页；约 1.3T GPT-2 token；有 `sample-10BT` 等子集 | 可读取 Dataset Viewer 行 | 可进入**小样本审计**，不能当代码来源 |
| [CCI3-HQ](https://huggingface.co/datasets/BAAI/CCI3-HQ) | 中文网页；发布方称约 500GB，Hub 显示 518GB | 数据文件认证 HEAD 返回 403 | 暂缓；先完成访问条件与质量审计 |
| [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) | 代码；发布方称完整内容约 67.5TB；HF 上约 427GB 主要是文件 ID/元数据 | 数据文件认证 HEAD 返回 403 | **不适合作为当前直接下载的代码语料**；需访问与内容获取方案 |
| [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | 英文数学网页；`finemath-4plus` 约 9.6B token、670 万文档 | 可读取 Dataset Viewer 行 | 可进入**小样本审计**，但不是零噪声 |

上表 token 数使用发布方的分词方法，不能当作我们最终分词器的 token 数；存储大小也不是实际清洗后体积。

## 1. FineWeb-Edu

官方数据卡说明它从 FineWeb 网页中按教育质量筛选，包含 `sample-10BT`、`sample-100BT`、`sample-350BT` 和全量配置。数据字段有 `text`、`url`、抓取批次、语言与质量评分。页面标注 ODC-By，另受 Common Crawl 使用条款约束。

用 Dataset Viewer 对 `sample-10BT` 查看了极少量行：有完整的科普/技术解释，也有主题与我们的目标无关的文章；这不足以估计总体质量。数据卡明确指出代码含量可能偏低，因此它只适合候选**英文通用文本**，不能替代代码和技术文档。首轮只对 `sample-10BT` 做流式抽样，不下载 10B token 子集全集。

## 2. CCI3-HQ

数据卡描述为中文互联网文本，字段为 `id`、`text`、`score`，约 500GB。Hub 需要接受访问条件；用当前本机账号对数据文件做认证 HEAD 仍返回 403，无法进行实际随机抽样。

数据卡展示的**单条示例**把期刊介绍与多段重复的评论拼在一起，提示需要检查聚合、模板和重复问题；不能据这一条推断整个数据集质量。当前只保留为中文语料候选，不计入可用 token。若访问长期无法获得，应另找可审计的中文来源，而不是让英文网页代替中文。

## 3. The Stack v2

官方卡说明 HF 数据集主要提供 SWH 文件 ID、仓库/路径/许可等元数据，**不是可直接用于语言建模的源码文本**。正文在 Software Heritage 存储；批量获取需另行协议与凭证。当前账号对 HF 数据文件的认证 HEAD 为 403，连元数据抽样也暂不可做。

发布方提供 full、dedup、train-full-ids 和 train-smol-ids 版本，但它们仍是 ID/元数据。不同仓库有不同许可，必须按原始许可及移除请求处理。这里的 HF“总文件大小约 427GB”不能误解成可直接下载的完整代码文本。当前暂缓，不将其列入最近一轮训练配方；先找可直接审计内容、来源和许可的较小 Python/Shell 语料。`the-stack-smol` 虽有约 2.6GB 正文，仍需接受访问条件和逐来源审查，不能自动视为替代品。

## 4. FineMath

官方卡给出 `finemath-3plus` 约 34B token/2140 万文档；质量分数更高的 `finemath-4plus` 是其子集，约 9.6B token/670 万文档。另有 `infiwebmath` 两组，不应与 FineMath 主集重复计数。字段包含 `text`、`url`、抓取时间、评分、语言等。页面标注 ODC-By、Common Crawl 条款。

Dataset Viewer 对 `finemath-4plus` 的少量行既有清晰的代数/应用题材料，也有含混解释或售卖练习册的网页片段。说明“4+”仍要做模板、题目完整性和推导正确性的审计。它是英文数学预训练候选，不替代可执行验证的数学能力评测，也不提供中文数学覆盖。

## 5. 结论与下一动作

1. **先审 FineWeb-Edu 的 `sample-10BT` 与 FineMath 的 `finemath-4plus`**：各取分层样本，保留 URL/分数/抓取批次，人工核查后统计可用比例。
2. **CCI3-HQ 与 The Stack v2 暂缓**：记录访问受限，不能假定已经有可训练正文；继续寻找中文和 Python/Shell 的可访问候选。
3. **补技术文档来源**：当前四个数据集没有一个能可靠承担技术文档桶。
4. 只有通过来源审查、内容抽样、去重和分词统计后，才能决定正式数据配比。
