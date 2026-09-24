# 2A：正式分片数据读取器（正确性阶段）

状态：实现分片格式、流式打包、mmap 读取、确定性领域混合、DDP 样本分配与 checkpoint 恢复。**暂不启用多 worker、异步预取或 CUDA 双缓冲**；这些属于 2B，必须在 2A 的轨迹测试稳定后加入。

## 文件格式

```text
tokenized/v1/
├── train-00000.bin
├── train-00000.idx
├── train-00001.bin
├── train-00001.idx
├── validation-00002.bin
├── validation-00002.idx
└── manifest.json
```

- `.bin`：无文件头 little-endian `uint16` token 流。每条文档保留一个 EOS；同一 shard 只含一个 split 和 domain。
- `.idx`：每条样本 16 byte，`<QII`，依次为 `token_offset`、`valid_tokens`、`flags`。`flags=1` 表示验证/测试尾部需要 padding。
- 相邻样本使用 `stride=sequence_length-1`，重叠一个输入 token，避免 causal-label shift 漏掉内部 token 的预测目标。
- 训练 shard 不使用不足一个完整窗口的尾部；验证/测试 shard 使用 padding 并把对应 label 设为 `-100`。小 shard 的训练尾部损耗会略高，正式数据默认每片约 3200 万 token，损耗可忽略。
- `manifest.json` 记录 tokenizer hash、来源 hash、license gate、领域混合比例，以及每片文件大小和 SHA-256。文件名不允许路径穿越。

## 打包

从已冻结的 Parquet 来源流式编码，避免将数千万 token 先放入 Python 列表：

```bash
python scripts/p0_pack_sharded_v1.py \
  --source /path/to/accepted_real_v1 \
  --tokenizer /path/to/frozen_tokenizer \
  --mixture-json configs/data/mixture.json \
  --output /path/to/tokenized/v1 \
  --sequence-length 4096 \
  --target-shard-tokens 32000000
```

`mixture.json` 是简单的 domain → 权重映射，例如：

```json
{"general": 0.55, "technical": 0.20, "code": 0.15, "math": 0.10}
```

打包器会检查来源 Parquet 和 tokenizer hash、编码→解码一致性、重复文档 ID、EOS 冲突，并先写临时目录，完整成功后才原子改名到输出目录。已经存在的输出目录不会覆盖。

## 训练配置

旧百万 token 单流仍使用默认 `indexed_v1`。新格式在 run config 中加：

```json
{
  "data_reader": "sharded_v1",
  "data_dir": "/path/to/tokenized/v1",
  "mixture_block_size": 1000
}
```

`trainctl.py` 会核对每片 `.bin/.idx` 的大小和 SHA-256，以及 vocab/EOS、上下文长度和许可门。`ShardedTokenDataset` 使用 mmap，按需打开 shard，并限制同时打开的 shard 数。

## 确定性混合与恢复

每个 mixture block 根据权重分配固定票数，并使用 SHA-256 派生的种子打乱。每个领域按 cycle 打乱 shard 顺序，再打乱 shard 内 sample 顺序。全局位置、manifest hash、seed、block 大小相同时，得到的 `(shard_id, sample_id, token_offset)` 不变。

DDP 的全局样本位置为：

\[
p=p_0+j(WB)+rB+i,
\]

其中 `j` 是本 rank 的 microbatch 序号，`W` 是卡数，`B` 是每卡 microbatch，`r` 是 rank，`i` 是 batch 内位置。每个完整 optimizer step 后才调用 `commit_step()`，推进已提交的全局游标。checkpoint **不保存已分发但未训练的游标**；保存的是：

- `committed_global_position`；
- manifest hash、seed、block 配额、world size 和 batch 配置；
- 下一条的 `shard_id/sample_id/token_offset`，用于恢复交叉校验。

分布式保存/加载 checkpoint 时，各 rank 会核对相同的提交位置。恢复不接受 manifest 或 batch 布局漂移。2A 强制 `num_workers=0`；后续 2B 引入预取时仍必须保持“分发”和“提交”两个游标分离。

## 当前验收与边界

自动测试覆盖：格式/索引、文件篡改拒绝、领域比例、同种子序列、DDP 不重叠分配、预分发后仅恢复已提交样本、真实 checkpoint 往返。L40 冒烟配置是 `configs/runs/base_1b_sharded_2a_canary_l40_v1.json`。

2026-09-24 L40 验收：使用原百万 token 冒烟集重新打包为 14 个 shard（训练 12、验证 1、测试 1），8 卡训练第 1 步保存完整 checkpoint，随后从该 checkpoint 恢复并完成第 2 步。最终处理 262,080 token，8 个 rank 的 `committed_global_position` 均为 64；保存的下一个 shard/sample/token offset 与确定性日程逐 rank 一致。单元测试 16 项通过。为保持服务器整洁，验收后删除约 12 GB 的临时 checkpoint，仅保留小型状态、指标和 `sharded_2a_validation.json`；该 canary 因此**不再可继续恢复**。这个 GPU canary 只有 `general` 一个领域；多领域混合比例与顺序目前由单元测试验证，正式多领域数据仍需另跑端到端验收。

当前 2A 不承诺：多 worker 加速、动态修改混合比例、改变 world size 后继续精确恢复。任一数据或调度配置变化都应新建 run，不能复用旧 checkpoint。
