# P0 高质量上游分片：2026-09-25 夜间批次

**完成状态（2026-09-25 09:25 CST）：** 20/20 文件已传 V100，0 失败；在 V100 依据冻结计划独立复算 20 个文件的字节数与 SHA256，全部匹配，共 **11,110,522,426 字节**。Mac 暂存目录已无 Parquet 文件，仅剩约 16 KiB 日志、锁和进度记录。后台任务 07:44:45 报告 `ALL_FILES_VERIFIED` 后退出。这里的“完成”仅指原始文件传输与完整性，不代表质量筛选或训练集验收。

本批只下载**钉死 revision 的原始 Parquet**，不修改现有 `wave1_consolidated_100m_v1`，不把下载量当作已验收训练 token。

计划 `configs/data/p0_overnight_upstream_v1.json` 从 Hugging Face API 逐文件取得实际字节数和 LFS SHA256；三来源交错下载，避免某一来源占满夜间窗口。共 20 个分片、11.11 GB：

| 来源 | 分片 | 预期下载量 | 用途 |
| --- | ---: | ---: | --- |
| FineWeb2-HQ `cmn_Hani` | 8 | 7.40 GB | 中文精选来源试验；是 FineWeb2 子集，不额外计容量 |
| FineMath-4+ | 8 | 2.29 GB | 数学高质量子集试验；是 3+ 子集，不额外计容量 |
| `github-code-clean` 新分片 | 4 | 1.42 GB | 代码补充；仍需逐行许可和质量门 |

中转脚本 `scripts/p0_fetch_overnight_upstream.py` 在 Mac 后台运行，`caffeinate` 防止空闲休眠。Mac 的 `~/.cache/small-agent-p0-overnight-v1/` **串行下载**；若 V100 隧道暂时不通，会保留已下载文件等待下一轮中转，本批最多 11.11 GB，不声称已到服务器。下载后先按预期尺寸和 SHA256 校验，再 `rsync` 到 V100 的 `/home/data/wangyue/datasets/small-agent-p0/p0_500m_v1/upstream_v2/raw/`，在 V100 重新算 SHA256，确认无误才删除本地暂存。失败可断点续传，旧原始数据、旧候选池不覆盖。V100 同目录有 `download_plan_v1.json`，本地 `progress.json` 与 `download.log` 记录进度。`scripts/p0_overnight_runner.sh` 在九小时内自动重试未传输项。

**后续验收，不可省略：** 核验 20 个文件及 SHA256；适配并正规化（HQ 有巨大 `embeddings` 字段，读入时应只投影必要列）；来源内/跨来源去重；按中文、数学、代码和长度分层抽样双模型严格评分；统计自有 tokenizer 的实际保留 token。只有通过这些步骤的数据才能进入 5 亿目标池。

首个中文 HQ 原始分片 `000_00061` 在 Mac 校验后读取 Parquet 元数据：79,458 行；`quality_score` 分位数约为 P50=0.013、P75=0.050、P90=0.216、P99=0.891。`score≥0.5` 仅 3,470 行。个别低分可见广告/拼接，部分高分是连贯历史文本，但高分也不保证符合我们的预训练目标。**先按质量分数分层双评，再确定是否增加 HQ 内二次分数门；不能现在直接把 0.5 定成生产阈值。**

## 新增语料容量初算（2026-09-25）

V100 原始 Parquet 元数据精确行数：中文 HQ **483,876**、FineMath-4+ **837,437**、代码混合分片 **507,697**，合计 **1,829,010** 行。`scripts/p0_estimate_overnight_raw_v1.py` 从每文件 12 个间隔 row group、每组固定随机 25 行，以冻结的自有 tokenizer 抽样估算文本 token，结果保存在 V100 `p0_500m_v1/audit/overnight_raw_capacity_estimate_v1.json`：中文 HQ 约 **4.45 亿**、FineMath-4+ 约 **13.19 亿**、代码混合原始约 **13.69 亿**。后者**不能直接计入目标代码语料**。

代码分片逐行检查语言与原有宽松许可白名单后，仅 **64,120/507,697** 行同时合格（JavaScript 39,042；Python 21,198；Shell 3,880）；对这些行逐篇用自有 tokenizer 精确计数，得到 **151,104,478 token**（JavaScript 101,948,487；Python 46,109,004；Shell 3,046,987）。这仍在文本规则、质量门和跨来源去重**之前**。因此，本次新增的“领域/许可范围内原始 token”粗估约 **19.15 亿**，其中中文/数学是抽样估计、代码是精确文本计数；不是可训练 token。旧来源原始容量曾估约 29.39 亿，机械相加约 48.54 亿，但 HQ⊂FineWeb2、4+⊂3+，且尚未去重，**不得视作独立有效容量**。
