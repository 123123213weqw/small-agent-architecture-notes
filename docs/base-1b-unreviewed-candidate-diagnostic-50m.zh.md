# 1B Base 候选语料诊断训练（50M）

日期：2026-09-25。**这是未验收候选语料上的内部诊断，不是正式预训练 Base，也不能用于架构优越性结论。** 首批约 9,992 万自有 tokenizer 文本 token 是候选池；质量门正式验收训练 token 仍为 0。

## 数据和运行

- 来源：V100 的 `p0_500m_v1/wave1_consolidated_100m_v1`，84,142 篇，许可门通过，但内容质量未批量验收。
- 冻结 Byte-level BPE 32k tokenizer；打包为 19 个 sharded v1 分片。含 EOS 共 100,006,695 token，其中 train 98,087,349、validation 971,705、test 947,641。manifest 明确为 `internal_smoke_sharded_token_stream`、`quality_approved=false`、`training_eligible=false`，只能用于非发布诊断。
- L40 数据：`/data1/wangyue/experiments/small-agent-base-1b-v1-engineering/data/unreviewed_candidate_100m_sharded_v1`；manifest SHA-256：`3fb1f4971a8d57a09e0f7f5a0c192d9292d5f7366ff1790c49c6a3bb5d6f0d31`。冻结 tokenizer.json SHA-256：`9d0b99ab9cf20d337217c96a5b39e9ff87590eb462abcbcc8f1119c3d43911a7`。
- 配置：`configs/runs/base_1b_unreviewed_candidate_diagnostic_50m_l40_v1.json`，8×L40，1,005,213,696 参数，4096 上下文，microbatch 1、累积 4，382 步，全局每步 131,040 个预测 token。
- Run ID：`unreviewed_candidate_diagnostic_50m_v1`，L40 目录：`/data1/wangyue/experiments/small-agent-base-1b-v1-engineering/runs/unreviewed_candidate_diagnostic_50m_v1`。
- 首次启动在训练前因 NCCL 连接本机 LAN 地址中断，没有训练步数、checkpoint 或有效 run manifest；空运行目录已移除。重试明确传入 `NCCL_SOCKET_IFNAME=lo`、`GLOO_SOCKET_IFNAME=lo`，通过 8 卡预检并正常完成。

## 结果

| 项目 | 数值 |
| --- | ---: |
| 完成步数 | 382 / 382 |
| 训练预测 token | 50,057,280 |
| 训练日志中位吞吐 | 约 49,355 token/s |
| 最终训练 loss（日志窗口） | 5.8014 |
| 验证 token | 971,699 |
| 验证 loss：step 0 → 76 → 152 → 228 → 304 → 382 | 10.7038 → 7.3591 → 6.7162 → 6.2460 → 5.9919 → **5.8806** |

完整 checkpoint 位于 `step_00000038`（4,979,520 token）、`step_00000153`（20,049,120 token）、`step_00000382`（50,057,280 token）。每份 19 个 artifact、约 12.06 GB；已逐文件核对大小与 SHA-256，`latest.json` 指向 step 382 且 manifest 哈希正确。最终模型已被独立评估脚本成功加载。**本次没有从最终 checkpoint 继续训练，不能声称这次实际验证了续跑轨迹；此前 M7 已验证训练器的恢复机制。**

最终 checkpoint 的分领域 validation（`validation_by_domain_step382.json`）：

| 域 | 预测 token | loss |
| --- | ---: | ---: |
| 中文通用 | 336,519 | 7.2872 |
| 英文通用 | 344,318 | 5.3166 |
| 英文数学 | 170,595 | 4.9548 |
| Python | 50,512 | 4.7712 |
| Shell | 1,709 | 5.4159 |
| 其他代码 | 68,046 | 4.9353 |

加权合计 loss 与训练器最终验证 **5.880639866929055** 完全一致。不同语言/领域的 loss 受 tokenizer 分词率和文本熵影响，不能直接用绝对值排序数据质量；Shell 验证样本仅 1 个，尤其不稳定。

## 解释边界

这次证明 1B GDN/GQA 基座和 8 卡训练 infra 能在更大、非重复候选池上完成 50M-token 训练，loss 有序下降，三个 checkpoint 完整。它**不证明**候选数据质量合格、模型已经具备 agent/code/math 能力，也**不证明**记忆结构优于 Transformer。下一步仍是质量门与更长训练预算，而非把本 checkpoint 标记为正式 Base。
