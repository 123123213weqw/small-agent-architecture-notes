# P0 GDN 快速内核：L40 长度测试

日期：2026-09-24。目的为检验 8 层 GDN+GQA 小模型在长序列上能否使用快速内核完成 BF16 前向/反向，并核对参考实现；**不据此判断约 1B 模型的性能或质量**。

> 后续已通过显式配置 BF16 dtype 去除本页记录的 fused norm 回退，并锁定工程环境；最新结果见[环境锁定与复测](p0-gdn-l40-env-stabilization.zh.md)。本页保留第一次快速路径试验的历史记录。

## 服务器文件位置

所有新增依赖、下载轮子、Triton 编译缓存、代码和结果均集中在 L40：

```text
/data1/wangyue/experiments/small-agent-p0-gdn-hybrid-smoke-v1/
├── code/                  # 测试脚本
├── deps/                  # --target 安装；没有修改原 small-agent-l40 venv
├── wheels/                # 固定版本的 causal-conv1d wheel
├── triton-cache/          # 受控 Triton JIT 缓存
├── triton-cache-initial/  # 首次试跑生成的缓存，从用户 home 收拢至此
├── fast/                  # 快速内核测试和 JSON 结果
└── run/                   # 上次 PyTorch 回退短跑
```

输入 token 流仍复用 `/data1/wangyue/experiments/small-agent-p0-engineering-v1/data/`，没有再复制一份。测试结束后 GPU 已释放。服务器原有 Python venv 未改动。

## 版本与兼容处理

- 已有环境：Python 3.10、PyTorch 2.5.1+cu121、Triton 3.1.0、Transformers 4.57.1。
- 项目局部依赖：`flash-linear-attention==0.2.2`、`einops==0.8.1`、`causal-conv1d==1.5.0.post8`。该 FLA 版本声明支持 PyTorch ≥2.5；没有安装要求 PyTorch ≥2.7 的较新 FLA。[FLA v0.2.2 元数据](https://github.com/fla-org/flash-linear-attention/blob/v0.2.2/pyproject.toml)
- causal-conv1d 使用与当前 PyTorch ABI 对应的 `cu12torch2.5cxx11abiFALSE-cp310` 官方 wheel，SHA-256 `ec956a2d0fa48bd16a8dd5be9ad1c03db88565ea6ee2679e362940fef659284b`。[官方项目](https://github.com/Dao-AILab/causal-conv1d)
- Transformers 4.57.1 在这组环境里启用 FLA fused RMSNorm 时调用了当前 PyTorch 不存在的 `torch.get_current_dtype`。测试脚本因此只对该 fused norm 切回 PyTorch 参考实现；GDN delta-rule 与 causal-conv1d 仍使用快速内核。故下面数据是**部分融合**路径，不是未来最终内核组合。
- FLA 对当前 Triton 3.1.0 和 Python 3.10 给出版本偏旧警告；本次算子能工作，但在升级版本前不把这些吞吐数字当正式性能结论。

## 正确性

使用同一份随机初始化权重、相同 BF16 autocast 输入，对照加速与 PyTorch 参考 GDN：

| 长度 | 平均 logit 绝对差 | 最大 logit 绝对差 | loss 绝对差 | greedy 不同位置 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.00696 | 0.05078 | 0.00177 | 4 |
| 2048 | 0.00684 | 0.06348 | 0.00027 | 111 |
| 4096 | 0.00684 | 0.06836 | 0.00003 | 201 |

128 长度的 embedding 梯度余弦相似度为 **0.99965**，相对 L2 差 **2.65%**。快速路径的短训练与反向均完成且 loss 有限。BF16 下 **greedy 结果并非逐位置完全相同**，不能声称 bit-exact；随机初始化时许多最高 logits 接近，短序列整段与递推路径的两处 greedy 差异对应最大 top-2 margin 仅 0.0078。后续生成验收仍需在训练后模型上复测。

## 单卡长度测试

代码：`experiments/p0_gdn_length_benchmark.py`。同一 45,446,368 参数模型、batch 1、BF16、一次整段前向加反向，不含优化器更新；每个长度预热 3 次、计时 10 次，排除首次 JIT 编译。缓存、进程启动、读数据和模型初始化不计入下表。

| 长度 | 平均前向＋反向 | 有效 token/s | 峰值 allocated | 峰值 reserved |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 39.82 ms | 约 51,426 | 1.68 GiB | 2.07 GiB |
| 4096 | 41.80 ms | 约 97,995 | 3.08 GiB | 3.96 GiB |

首次编译开销很高，2048 测试首次预热约 148 秒，4096 首次约 41 秒；不应把预热吞吐与稳定算子吞吐混为一谈。当前模型只有 45M 参数，长短序列计时受算子启动、GPU 状态等影响；4096 的 token/s 较高**不能外推**到约 1B，也不能作为与普通 Transformer 的公平比较。完整 JSON 位于 `fast/benchmarks/`。

## 结论与下一步

8 层混合模型已在单张 L40 上完成 2048/4096 BF16 前向和反向，无 NaN/OOM；参考实现的 loss 与梯度接近。接下来应先固定生产级依赖组合（避免旧 Triton/Python 与局部 fused norm 回退），再做**参数、batch、精度、硬件及优化器步骤都匹配**的 Transformer 对照。现有百万 token 数据只够工程验收，不适合比较语言能力。
