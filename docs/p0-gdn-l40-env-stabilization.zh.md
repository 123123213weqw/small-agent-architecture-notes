# L40 GDN 工程环境锁定与复测

日期：2026-09-24。目标是固定可重复执行的**工程环境**，不升级或覆盖现有 `small-agent-l40` venv，也不把小模型跑分视作正式性能结论。

## 改动

1. 在随机初始化的 `Qwen3NextConfig` 中显式设置 `dtype=torch.bfloat16`。Transformers 4.57.1 此时不再走 `torch.get_current_dtype()` 分支；原先临时关闭 FLA fused RMSNorm 的 monkeypatch 已删除。实测 embedding 权重仍以 FP32 初始化，GDN fused norm、FLA delta-rule 和 causal-conv1d 均处于启用状态。
2. 在 `configs/gdn/p0_l40_fast_env_v1.json` 锁定 Python、PyTorch、Transformers、Triton、FLA、causal-conv1d、NumPy 及运行时小依赖版本。`scripts/l40_gdn_env.sh` 每次运行前核对版本，只使用集中目录中的 `deps/`，禁用用户 site-packages，并把 Triton/TMP 缓存定向到同一实验根目录。
3. 复用原先的 L40 根目录 `/data1/wangyue/experiments/small-agent-p0-gdn-hybrid-smoke-v1/`；新结果只放在其 `stable/` 下。旧 `run/`、`fast/` 留作有标签的历史结果，没有在别处散落新训练文件。GPU 在测试后已释放。

## 复测结果

- 环境 preflight：固定版本逐项匹配，`is_fast_path_available=True`，`FusedRMSNormGated` 可用，配置 dtype BF16。
- 8 层 45,446,368 参数模型从零进行 128 长度短训练，step 4 → 8 → 9 的 checkpoint 跨进程恢复通过；step 9 验证 loss 10.328，模型/梯度均有限。短跑只处理 2304 token，不用于能力判断。
- 同一权重下加速与 PyTorch 参考 GDN 的 128 长度 embedding 梯度余弦相似度 **0.99965**、相对 L2 差 **2.66%**；2048/4096 的 loss 差分别约 **0.00019 / 0.00017**。BF16 的 greedy 输出不是 bit-exact，分别有少量不同位置；不能声称生成逐 token 必然一致。
- 2048/4096、batch 1、BF16、单张 L40 的预热后前向＋反向均完成，无 NaN/OOM。五次计时均值分别约 **36.95 ms / 42.17 ms**，峰值 allocated **1.59 / 2.91 GiB**。另一次隔离环境 2048 重测为 48.53 ms，显示跨进程计时仍有明显波动，不把它当正式吞吐结论。

## 仍未满足“生产环境”的条件

- 当前 Python 3.10 / Triton 3.1 比 FLA 推荐版本旧，导入会警告；虽然已通过本次有限测试，仍需在独立新环境验证较新版本组合，不能无证据升级原 venv。
- 这里使用的 FLA 0.2.2 对 `datasets` 有包元数据依赖，但当前测试只调用 `fla.ops`，没有安装数据集模块；这不是一个供任意 FLA 功能使用的完整通用环境。
- 未做训练后模型的长提示缓存生成质量测试，也没有与参数、batch、精度、优化器和硬件匹配的 Transformer 对照。

**结论：当前是锁定并通过回归测试的 L40 GDN 工程环境；不是正式生产栈或架构优势证据。**
