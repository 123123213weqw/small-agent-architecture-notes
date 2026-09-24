# P0 tokenizer_v1：编码、打包与小模型工程短跑

日期：2026-09-24。该实验只验证从冻结文本到 checkpoint 的工程链路，**不评价 GDN 架构或语言能力**。

## 输入与冻结

- V100 冻结 tokenizer：`/home/data/wangyue/datasets/small-agent-p0/p0_1m_smoke_v1/tokenizer_v1/`；Byte-level BPE、词表 32,768，`tokenizer.json` SHA-256 `9d0b99ab9cf20d337217c96a5b39e9ff87590eb462abcbcc8f1119c3d43911a7`。
- 原始冻结文档：`/home/data/wangyue/datasets/small-agent-p0/p0_1m_smoke_v1/p0_1m_frozen_v1/`，721 篇；原 manifest 的约 100 万 token 按旧 Qwen tokenizer 计数，不作为本实验计数。
- 源数据 manifest 的 `license_gate.training_eligible=false`，原因是 690 篇来源许可仍为 `pending`。本次仅作为内部工程冒烟；不宣称数据已获正式预训练批准。

## 第 1–2 步：重新编码与 split-safe token 流

代码：`scripts/p0_pack_smoke_v1.py`。逐 shard 验证哈希；按固定 `(sample_rank, document_id)` 排序；逐篇使用冻结 tokenizer 编码并确认无损解码；每篇后添加一个 EOS；原 train/validation/test 划分不变，不跨 split 拼接。输出 `uint16` `.npy` token 流、逐文档 token span 及带输入/输出哈希的 manifest。2048 长度为本次训练视图，token 流本身不截断文档。

V100 输出：`/home/data/wangyue/datasets/small-agent-p0/p0_1m_smoke_v1/p0_1m_token_stream_v1/`。

| Split | 文档 | 含 EOS token | 完整 2048 块 | 尾部 token |
| --- | ---: | ---: | ---: | ---: |
| train | 706 | 1,110,815 | 542 | 799 |
| validation | 9 | 15,329 | 7 | 993 |
| test | 6 | 15,902 | 7 | 1,566 |

三组总计 1,142,046 token = 上次静态评测的 1,141,325 正文 token + 721 个显式 EOS。L40 接收到相同哈希的 token 流；L40 路径：`/data1/wangyue/experiments/small-agent-p0-engineering-v1/data/`。Manifest SHA-256：`4b62a5cd59e4615de411854b4c997ad566bcc2ea66748e3ea2055cec6f1584d0`。

## 第 3 步：训练与断点恢复

代码：`experiments/p0_lm_engineering_smoke.py`。使用**普通 6 层因果 Transformer**，38,286,336 参数、宽度 512、8 注意力头、SwiGLU 1536、词表 32,768、上下文 2048、输入输出 embedding 共享。此处故意只测试数据/训练工程，不测试尚未实现的 GDN。单张 L40 GPU 0，BF16 autocast，AdamW，batch 4，固定随机种子。

- 第一次从零跑至 step 20，保存 checkpoint。
- 第二次从 step 20 恢复至 step 80；模型、优化器、采样 RNG 和 Torch RNG 状态均恢复。第一次恢复尝试发现 CPU RNG 被 `map_location=cuda` 错放到 GPU，已将脚本修正为先在 CPU 加载 checkpoint，随后通过验收。
- step 10/20/40/60/80 的 validation loss：8.762 / 7.843 / 7.337 / 7.189 / 7.151；全程有限、无 NaN。
- 总处理 token `80 × 4 × 2048 = 655,360`，train 中不同 token 仅约 111 万。短跑不能用于判断能力或预训练收敛。
- 输出：`/data1/wangyue/experiments/small-agent-p0-engineering-v1/run/`，含 `checkpoint.pt`（约 439 MB）、`summary.json`、TensorBoard 事件文件。实验完成后 L40 GPU 已释放。

下一步不是把这一百万 token 反复训练成「数千万 token」；应先扩充且批准正式数据，再做原生 GDN+GQA 小配置的前向/反向和吞吐短跑，最后冻结约 1B 配置。
