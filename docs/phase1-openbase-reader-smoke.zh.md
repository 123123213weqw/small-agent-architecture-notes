# Phase 1：开源 Base 的记忆 Reader 短测

> 状态：首轮工程短测完成；**尚未开展 FIFO 与效用选择器的正式 A/B/C**。

## 起点和范围

- 基座：`Qwen/Qwen3-0.6B-Base`，revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`，沿用原 tokenizer。
- L40 模型文件 SHA-256：`cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`。
- 脚本：`experiments/phase1_openbase_reader_smoke.py`；单张 L40 0 号卡运行。
- 基座冻结；额外训练一层记录编码器、一个 Cross-Attention 和残差门。每例有 8 条早期 `ID→随机数字` 工具记录，查询其中一个 ID。目标答案在记录中，**不在近期 prompt 中**。训练和验证使用不同的随机 episode。
- 这个短测只检查 Reader 是否学会利用记忆，不测试效用淘汰，更不能代表最终架构。

## 已完成的检查

1. 模型可从 L40 的个人 JFS 目录加载；3 步反向传播、loss、保存 adapter 和 TensorBoard 写入均正常。
2. 未加训练时，查询 prompt 不含旧记录：128 题数字准确率 `7.0%`，答案 NLL `2.855`；把全部 8 条记录直接放进基座上下文：准确率 `100%`，NLL `0.383`。因此基座本身能解这个任务，信息也足够。
3. 记忆 Cross-Attention 在零门初始化、300 步后，128 题数字准确率仍约 `6.3%`，正确记忆与错配记忆的 NLL 差几乎为零。
4. 把门改为 `0.5` 初始化、学习率 `1e-3`、16 样本批次训练 1000 步，256 题准确率约 `9.4%`；正确记忆与错配记忆的 NLL 差仍约为零，门参数从 `0.5` 降到 `0.011`。模型学会了**忽略**这版记忆读头。

## 解释

这不是“效用数学无效”，也不是“Qwen3 Base 不会读旧记录”：把旧记录放进普通上下文时它答对全部题。当前阻塞点是**新增 Reader 接口/训练方式**，尤其是只用浅层编码器处理独立旧记录、冻结基座后用随机初始化 Cross-Attention 接入，容易被 LM loss 训练成关闭残差门。

在 Reader 给对证据仍读不出来时，启动 FIFO/效用淘汰对照没有解释力。下一实验应优先把旧记录编码为基座的上下文化隐藏状态，并比较冻结基座与逐步解冻顶部层；只有 Reader 的“正确记忆优于错配记忆”成立后，才分叉 B/C。

## 记录位置

- `runs/phase1_openbase_reader_smoke_s0/`：零门、3 步工程短测。
- `runs/phase1_openbase_reader_kv_s0_v1/`：零门、300 步。
- `runs/phase1_openbase_reader_kv_gate05_s0/`：门初值 0.5、1000 步。

每个运行目录含配置清单、逐次指标、TensorBoard 事件及 adapter 权重；这些数据是诊断性结果，不能作为正式架构胜负。
