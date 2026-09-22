# 候选结构 A（已被简化版取代）：通用持久情景记忆层的数学定义

> **状态：Superseded。** 本文保留早期推导和失败分析，不再作为实现规范。当前冻结的最小原型见 [效用巩固记忆 v1](utility-consolidation-memory-v1.md)。

> 状态：研究提案，未接受为最终架构  
> 工作名称：Latent Persistent Episodic Memory（LPEM）  
> 目标：为约 1B 参数的自回归模型增加有界、可学习、可衰减、可关联和可精确回看的跨片段状态。  
> 重要说明：本文定义的是待验证假设，不主张结构已经有效或具有确定的新颖性。

## 1. 设计问题

给定自回归序列

\[
x_{1:T}=(x_1,\ldots,x_T),
\]

普通局部 Transformer 只能直接访问最近 \(W\) 个 token。我们希望在固定显存预算下维护状态 \(\mathcal M_t\)，使模型能够：

1. 保存少数重要的历史状态变化；
2. 根据当前内容读取相关记忆，而不是按时间顺序遍历全部历史；
3. 关联跨越多个步骤的动作、观察和结论；
4. 遗忘低效、冲突或已被替代的内容；
5. 必要时从压缩记忆重新定位原始 token 片段；
6. 不把 `tool_call`、文件、测试等特定领域概念硬编码进架构。

核心优化问题写成：

\[
\min_\theta
\mathbb E\left[
\mathcal L_{\mathrm{LM}}(x_{1:T};\theta,\mathcal M)
+\lambda_w C_{\mathrm{write}}
+\lambda_r C_{\mathrm{read}}
+\lambda_m C_{\mathrm{memory}}
\right].
\]

第一项要求记忆提高语言建模和任务能力，后三项要求模型不能无限写入、无限读取或无限扩张状态。

## 2. 时间组织与因果约束

### 2.1 片段

把长序列分为长度不超过 \(W\) 的片段：

\[
X_e=x_{t_e:t_{e+1}-1},\qquad e=1,\ldots,E.
\]

片段边界可以来自：

- 固定长度；
- 消息结束；
- 环境返回；
- 模型学习出的状态变化边界。

这些边界都只提供“允许提出一次记忆写入”的时机，不保证真的写入，也不携带工具专用语义。

### 2.2 严格因果递推

第 \(e\) 个片段只能读取处理它之前已经存在的记忆：

\[
H_e=F_\theta(X_e,\mathcal M_{e-1}),
\]

处理完片段后才更新记忆：

\[
\mathcal M_e=U_\theta(\mathcal M_{e-1},H_e).
\]

因此 \(X_e\) 中任何 token 都不能通过片段摘要看到未来 token。训练时必须保持这一顺序，不能先对完整序列编码再构造记忆。

## 3. 基础局部模型

对第 \(l\) 层，首先执行普通局部因果注意力：

\[
\bar H_e^{(l)}
=H_e^{(l)}
+\operatorname{LocalAttn}^{(l)}
\left(\operatorname{RMSNorm}(H_e^{(l)})\right).
\]

局部注意力的 token 复杂度为：

\[
O(TWd).
\]

持久记忆不替代局部注意力。前者负责跨片段状态，后者负责精确的近期 token 组合。

## 4. 记忆状态

维护 \(S\) 个有界槽：

\[
\mathcal M_e=
\left(
K_e,V_e,s_e,a_e,o_e,G_e,P_e
\right),
\]

其中：

\[
K_e\in\mathbb R^{S\times d_m},\qquad
V_e\in\mathbb R^{S\times d_m}.
\]

各变量含义如下：

| 变量 | 含义 |
|---|---|
| \(K_i\) | 第 \(i\) 槽的内容寻址 key |
| \(V_i\) | 压缩后的记忆 value |
| \(s_i\in[0,1]\) | 当前显著性或有效先验 |
| \(a_i\ge 0\) | 自上次有效写入后的年龄 |
| \(o_i\in[0,1]\) | 槽占用程度 |
| \(G\in\mathbb R^{S\times S}\) | 槽之间的有向关联强度 |
| \(P_i\) | 指向外部原始片段及版本的非可微指针 |

模型参数中不保存具体任务内容；这些状态在推理时随序列递推。第一版将其限制为任务级状态，任务结束后清空。

## 5. 通用事件候选

### 5.1 片段摘要

对片段末层隐藏状态 \(H_e=(h_1,\ldots,h_n)\)，使用注意力池化：

\[
u_t=w_p^\top\tanh(W_p h_t),
\]

\[
\alpha_t=\frac{\exp u_t}{\sum_{j=1}^{n}\exp u_j},
\]

\[
c_e=\sum_{t=1}^{n}\alpha_t h_t.
\]

为防止一个摘要只能表达片段中的单一事件，可以使用 \(R\) 个池化 query 得到 \(c_{e,1:R}\)。第一版取 \(R=1\)，先验证机制本身。

### 5.2 新奇度和预测误差

候选 key/value 为：

\[
k_e=\operatorname{normalize}(W_k^c c_e),
\qquad
v_e=W_v^c c_e.
\]

相对于已有记忆的新奇度：

\[
n_e=1-\max_i\operatorname{cos}(k_e,K_i).
\]

还可以预测下一个片段摘要：

\[
\hat c_e=W_{\mathrm{pred}}[c_{e-1};r_{e-1}],
\]

\[
z_e=rac{\lVert c_e-\hat c_e\rVert_2^2}{d}.
\]

其中 \(z_e\) 表示当前观察相对内部预期的变化程度。为了避免模型通过任意放大表示作弊，需要对 \(c_e\) 和 \(\hat c_e\) 做归一化或对误差项的一侧使用 `stop-gradient`。

### 5.3 写门

写入概率定义为：

\[
g_e^w=sigma\left(
w_w^\top
[\operatorname{RMSNorm}(c_e);n_e;\log(1+z_e);b_e]
+b_w
\right),
\]

其中 \(b_e\) 只是通用边界特征，例如“消息结束”或“固定片段结束”，不包含工具名称。

写门可以从资源约束解释。若写入带来的未来期望损失下降为

\[
B_e=
\mathbb E[
\mathcal L_{\mathrm{no\ write}}
-\mathcal L_{\mathrm{write}}
],
\]

而一次写入成本为 \(\lambda_w\)，最优离散决策为：

\[
w_e^*=\mathbf 1[B_e>\lambda_w].
\]

\(g_e^w\) 是这个不可微阈值策略的连续近似。只有语言模型损失而没有写入成本时，模型很可能学成“什么都记”。

## 6. 更新已有记忆与分配新槽

### 6.1 内容匹配分布

候选事件与已有槽的匹配程度：

\[
q_i^{\mathrm{match}}
=\frac{k_e^\top K_i}{\tau_m}
+\beta_o\log(o_i+\epsilon),
\]

\[
p_i^{\mathrm{match}}
=\operatorname{softmax}_i(q_i^{\mathrm{match}}).
\]

### 6.2 是否形成新记忆

定义新槽门：

\[
g_e^{\mathrm{new}}
=\sigma\left(
\gamma_n n_e+\gamma_z\log(1+z_e)+b_n
\right).
\]

新奇度低时倾向更新已有槽；新奇度高时倾向替换一个低价值槽。

### 6.3 替换分布

低显著性、长期未使用且年龄大的槽更容易被替换：

\[
q_i^{\mathrm{replace}}
=-\gamma_s s_i
-\gamma_o o_i
+\gamma_a\log(1+a_i),
\]

\[
p_i^{\mathrm{replace}}
=\operatorname{softmax}_i(q_i^{\mathrm{replace}}/\tau_r).
\]

最终写入权重：

\[
\omega_i
=g_e^w\left[
(1-g_e^{\mathrm{new}})p_i^{\mathrm{match}}
+g_e^{\mathrm{new}}p_i^{\mathrm{replace}}
\right].
\]

训练初期使用连续权重；推理或后期训练可以使用 Top-1/Top-2、straight-through 或 Gumbel-Softmax，使槽的语义更清晰。

## 7. 可学习遗忘

### 7.1 从连续衰减方程得到保留率

假设槽显著性遵循：

\[
\frac{ds_i}{dt}=-\lambda_i(t)s_i(t).
\]

当一个事件间隔内 \(\lambda_i\) 近似常数时：

\[
s_i(t+\Delta)
=\exp(-\lambda_i\Delta)s_i(t).
\]

所以定义：

\[
\rho_i=\exp[-\Delta_e\,\operatorname{softplus}(d_i)],
\]

\[
d_i=w_d^\top
[K_i;k_e;K_i\odot k_e;s_i;\log(1+a_i);u_i]
+b_d,
\]

其中 \(u_i\) 是该槽近期被读取后的效用轨迹。这样自动保证：

\[
0<\rho_i\le 1.
\]

### 7.2 冲突和替代

是否冲突不能仅根据 key 判断，因为两个事件可能讨论同一对象却给出不同状态。定义：

\[
q_i^{\mathrm{conflict}}
=\sigma\left(
w_c^\top[K_i;k_e;V_i;v_e;V_i\odot v_e]
+b_c
\right).
\]

仅当新事件实际写入该槽附近时，冲突才增强遗忘：

\[
\bar\rho_i
=\rho_i\exp(-\kappa_c\omega_iq_i^{\mathrm{conflict}}).
\]

这使“时间衰减”和“被新信息替代”成为两条独立路径。

### 7.3 只衰减显著性，不直接缩小内容向量

直接反复执行 \(V_i\leftarrow\rho_iV_i\) 会改变表示尺度并造成数值退化。因此第一版保持 \(K_i,V_i\) 不动，通过 \(s_i\) 控制检索先验；只有写入或替换时才修改内容。

## 8. 内容更新的推导

对已分配槽，考虑局部重构损失：

\[
\mathcal L_i^{\mathrm{store}}
=\frac{1}{2}\omega_i
\lVert V_i-v_e\rVert_2^2.
\]

梯度为：

\[
\nabla_{V_i}\mathcal L_i^{\mathrm{store}}
=\omega_i(V_i-v_e).
\]

一步在线梯度下降得到：

\[
V_i'
=V_i-\eta_v\omega_i(V_i-v_e)
=(1-\eta_v\omega_i)V_i
+\eta_v\omega_i v_e.
\]

key 同理：

\[
\tilde K_i'
=(1-\eta_k\omega_i)K_i
+\eta_k\omega_i k_e,
\]

\[
K_i'=\operatorname{normalize}(\tilde K_i').
\]

显著性更新为：

\[
s_i'
=\operatorname{clip}
\left(
\bar\rho_i s_i
+\eta_s\omega_i
+\eta_u u_i,
0,1
\right).
\]

占用程度和年龄更新为：

\[
o_i'=1-(1-o_i)(1-\omega_i),
\]

\[
a_i'=(1-\omega_i)(a_i+\Delta_e).
\]

这里的 \(\omega_i\) 是软写入近似。采用硬分配时，写入槽年龄直接清零。

## 9. 潜在关联图

### 9.1 关联不是领域标签

\(G_{ij}\) 不表示固定的“文件—测试”关系，而表示在历史轨迹中，从槽 \(i\) 到槽 \(j\) 的条件依赖强度。工具调用、数学中间结论和普通对话状态都使用同一形式。

### 9.2 前驱分布

新事件对已有槽的潜在依赖：

\[
r_i=operatorname{softmax}_i\left(
\frac{(W_rk_e)^\top(W_pK_i)}{\sqrt{d_r}}
+\beta_t\log(1+\operatorname{recent}_i)
+\beta_u u_i
\right).
\]

令 \(\omega\in\mathbb R^S\) 是新事件写入位置分布，则外积

\[
r\omega^\top
\]

是本次“前驱槽到新槽”的期望转移计数。

### 9.3 图更新

若一个槽被替换，它原来的边必须同时失效。令：

\[
D=\operatorname{diag}(1-g_e^{\mathrm{new}}\omega),
\]

先清理相关边：

\[
\tilde G=DGD.
\]

再做指数移动更新：

\[
G'
=\rho_G\tilde G
+\eta_G r\omega^\top.
\]

最后按行归一化：

\[
G'_{ij}
\leftarrow
\frac{G'_{ij}}
{\sum_jG'_{ij}+\epsilon}.
\]

如果任务需要双向联想，可以额外添加 \(\eta_G^{\mathrm{rev}}\omega r^\top\)，但第一版保留有向边，以区分原因和后果。

从统计角度看，未归一化的 \(G\) 是带遗忘的转移计数估计，行归一化后近似：

\[
G_{ij}\approx p(\text{next relevant slot}=j\mid i).
\]

## 10. 记忆读取

### 10.1 检索分数的概率解释

假设归一化 query \(q\) 在槽 \(i\) 条件下服从 von Mises–Fisher 型相似度：

\[
p(q\mid i)\propto
\exp\left(\frac{q^\top K_i}{\tau}\right).
\]

记忆槽先验由显著性、年龄和关联给出：

\[
p(i\mid\text{history})
\propto
(s_i+\epsilon)^{\alpha_s}
\exp(\alpha_g\ell_i-\alpha_a a_i).
\]

根据贝叶斯公式：

\[
\log p(i\mid q,\text{history})
=\frac{q^\top K_i}{\tau}
+\alpha_s\log(s_i+\epsilon)
+\alpha_g\ell_i
-\alpha_a a_i+C.
\]

这直接得到检索 logit：

\[
z_i
=\frac{q^\top K_i}{\tau}
+\alpha_s\log(s_i+\epsilon)
+\alpha_g\ell_i
-\alpha_a\log(1+a_i)
+\log(o_i+\epsilon).
\]

其中关联先验可以由上一次活跃槽分布 \(\pi_{e-1}\) 计算：

\[
\ell=\pi_{e-1}^\top G.
\]

### 10.2 多头读取

在选定的第 \(l\) 层，对 token 隐状态 \(H\) 计算：

\[
Q^{(l)}=\operatorname{RMSNorm}(H)W_Q^{m,l},
\]

\[
K^{(l)}_m=K W_K^{m,l},
\qquad
V^{(l)}_m=V W_V^{m,l}.
\]

每个头使用上述内容相似度和状态先验：

\[
A^{(l)}
=\operatorname{softmax}
\left(
\frac{Q^{(l)}K_m^{(l)\top}}{\sqrt{d_h}}
+B_s+B_g+B_a+B_o
\right).
\]

可使用可微 Top-K 或直接对 \(S\) 个槽做完整 softmax。\(S=64\) 时完整计算通常更稳定。

读取结果：

\[
R^{(l)}=A^{(l)}V_m^{(l)}.
\]

### 10.3 是否需要回忆的门

不是每个 token 都需要长期记忆。采用逐头标量门：

\[
g^{r,l}=\sigma\left(
W_g^l\operatorname{RMSNorm}(H)+b_g^l
\right),
\]

\[
\hat H^{(l)}
=\bar H^{(l)}
+W_O^{m,l}(g^{r,l}\odot R^{(l)}).
\]

然后进入普通 MLP：

\[
H^{(l+1)}
=\hat H^{(l)}
+\operatorname{MLP}^{(l)}
(\operatorname{RMSNorm}(\hat H^{(l)})).
\]

## 11. 两阶段精确回看

压缩 value 不适合保存完整代码、长数字和逐字证据。因此每个槽可以保存：

\[
P_i=(\text{sequence id},\text{start},\text{end},\text{version}).
\]

第一阶段只读取 \(S\) 个压缩 landmark。若：

\[
g^{\mathrm{raw}}
=\sigma(w_{\mathrm{raw}}^\top[h;r;\operatorname{entropy}(A)]+b)
\]

超过阈值，则选择：

\[
\mathcal I=\operatorname{TopK}_i(A_i),
\]

并重新打开 \(P_i,i\in\mathcal I\) 指向的原始 KV 片段，执行第二次交叉注意力：

\[
R_{\mathrm{raw}}
=\operatorname{Attn}
(Q,K_{P_{\mathcal I}},V_{P_{\mathcal I}}).
\]

最终输出为：

\[
R_{\mathrm{final}}
=R+g^{\mathrm{raw}}R_{\mathrm{raw}}.
\]

原始片段存储属于运行时系统，不进入固定参数；过期指针必须通过版本号失效。若原文已被淘汰，模型仍可使用压缩 value，但不能伪装成精确引用。

## 12. 训练目标

总损失建议为：

\[
\mathcal L
=\mathcal L_{\mathrm{LM}}
+\lambda_w\mathcal L_{\mathrm{write}}
+\lambda_r\mathcal L_{\mathrm{read}}
+\lambda_b\mathcal L_{\mathrm{budget}}
+\lambda_d\mathcal L_{\mathrm{div}}
+\lambda_a\mathcal L_{\mathrm{assoc}}
+\lambda_c\mathcal L_{\mathrm{consistency}}.
\]

### 12.1 语言建模损失

\[
\mathcal L_{\mathrm{LM}}
=-\sum_t\log p_\theta(x_t\mid x_{<t},\mathcal M_{<t}).
\]

这是主目标，所有辅助目标只能作为弱约束。

### 12.2 写入预算

简单稀疏成本：

\[
\mathcal L_{\mathrm{write}}
=\frac{1}{E}\sum_e g_e^w.
\]

为了防止全部关闭，可以给定目标写入率 \(b_w\)：

\[
\mathcal L_{\mathrm{budget}}
=\left(
\frac{1}{E}\sum_e g_e^w-b_w
\right)^2.
\]

训练早期使用目标写入率，稳定后逐渐增大纯成本项，让模型从“必须写一些”过渡到“只写有用内容”。

### 12.3 读取成本

\[
\mathcal L_{\mathrm{read}}
=\frac{1}{T}\sum_t\lVert g_t^r\rVert_1
+\xi\frac{1}{T}\sum_t g_t^{\mathrm{raw}}.
\]

第二阶段原文读取成本应明显高于槽读取。

### 12.4 槽多样性

对高占用槽：

\[
\mathcal L_{\mathrm{div}}
=\sum_{i\ne j}o_io_j
\left[
\max(0,K_i^\top K_j-m)
\right]^2.
\]

它防止所有槽坍缩到相同内容，但权重不能太高，否则语义相近事件无法更新同一槽。

### 12.5 关联监督

如果训练轨迹能够得到“未来步骤依赖哪个历史事件”的弱标签 \(y_{ej}\)，则：

\[
\mathcal L_{\mathrm{assoc}}
=-\sum_{e,j}y_{ej}\log \hat p(j\mid e).
\]

标签可以来自受控合成任务、数据流依赖、变量引用或被遮蔽信息的来源位置。普通文本没有标签时只依靠 LM 梯度，不要求人工标注。

### 12.6 替代一致性

构造旧状态 \(v_{\mathrm{old}}\) 和新状态 \(v_{\mathrm{new}}\) 后，要求模型在新状态出现后选择新槽：

\[
\mathcal L_{\mathrm{consistency}}
=-\log p(v_{\mathrm{new}}\mid q)
+\max(0,p(v_{\mathrm{old}}\mid q)-\mu).
\]

这个目标直接测量并惩罚“继续使用已经过时结果”。

## 13. 防止捷径和训练坍缩

### 13.1 模型完全忽略记忆

当训练样本都短于局部窗口时，局部注意力已经足够，记忆层不会得到有效梯度。训练中必须包含：

- 依赖距离大于 \(W\) 的任务；
- 将关键事实移出局部窗口的样本；
- 多次状态更新与旧值干扰；
- 跨步骤引用但不重复原文的轨迹。

### 13.2 模型什么都写

使用写入预算、有限槽、读取成本，并在评估中报告每千 token 写入次数。

### 13.3 所有内容写进一个槽

使用低温分配、槽多样性和受控事件检索目标。必要时推理阶段硬 Top-1 写入。

### 13.4 只依赖原始回看

对原始片段进行随机淘汰和 dropout，让压缩记忆必须独立承担一部分预测；同时保留需要逐字恢复的任务，避免模型完全放弃原文接口。

### 13.5 跨片段反向传播不稳定

初期采用截断 BPTT：

\[
\operatorname{stopgrad}(\mathcal M_{e-k})
\]

每 \(k\) 个片段截断一次。随后逐步增加 \(k\)。也可以对 key/value 内容停止跨片段梯度，只训练读写控制器的短程梯度。

## 14. 计算和存储复杂度

设：

- token 数为 \(T\)；
- 局部窗口为 \(W\)；
- 记忆槽数为 \(S\)；
- 记忆维度为 \(d_m\)；
- 带记忆读取的层数为 \(L_m\)；
- 事件候选数为 \(E\)。

则：

### 14.1 计算量

局部注意力：

\[
O(LTWd).
\]

记忆读取：

\[
O(L_mTSd_m).
\]

内容写入和匹配：

\[
O(ESd_m).
\]

完整图更新：

\[
O(ES^2).
\]

若 \(S\) 增大，可把每行只保留 Top-\(k_g\) 条边，使图相关复杂度变为 \(O(ESk_g)\)。

### 14.2 每条序列的状态大小

不计原始片段：

\[
2Sd_m+S^2+O(S)
\]

个标量。以 BF16、\(S=64,d_m=512\) 为例：

- \(K,V\)：约 128 KiB；
- \(G\)：约 8 KiB；
- 其他元数据：远小于 1 KiB。

因此共享一份跨层记忆状态约为 136 KiB/序列。不要为每个 Transformer 层维护独立槽库。

## 15. 面向约 1B 模型的建议实例

第一轮受控实验建议：

| 项目 | 初始值 |
|---|---:|
| 基础层数 | 20–24 |
| hidden size | 1792–2048 |
| 局部窗口 \(W\) | 1024 或 2048 |
| 记忆槽 \(S\) | 64 |
| 记忆维度 \(d_m\) | 512 |
| 记忆读取层 | 每 4 层一次 |
| 每次读取槽数 | 先全量 64，后比较 Top-8 |
| 每片段候选数 \(R\) | 1 |
| 原文回看 Top-K | 2–4 |
| 关联图 | 64×64 有向图 |
| 状态范围 | 单任务，任务结束清空 |

若 \(d=2048,d_m=512\)，一个记忆读取层的主要投影参数约为：

\[
d d_m+2d_m^2+d_md
=2dd_m+2d_m^2
\approx2.62\text{M}.
\]

若共有 6 个读取层，约增加 15.7M 参数；加上写入控制器后，预计总增量仍可控制在约 20M，约占 1B 模型的 2%。最终比较时必须相应缩小 FFN 或 hidden size，使总参数匹配，而不是让候选模型无条件多出参数。

## 16. 最小实验矩阵

按由简到繁的顺序验证：

| 编号 | 结构 |
|---|---|
| A0 | 局部 Transformer，无持久记忆 |
| A1 | 固定槽 + 每片段强制写入 + 内容读取 |
| A2 | A1 + 学习式写门 |
| A3 | A2 + 显著性衰减 |
| A4 | A3 + 冲突替代机制 |
| A5 | A4 + 潜在关联图 |
| A6 | A5 + 原始片段二阶段回看 |

如果 A1 都不能在同参数预算下提高跨窗口召回，不应继续增加图和复杂门控。如果 A3 相比 A2 只降低召回而不减少旧信息误用，则衰减机制失败。如果 A5 对跨步骤关系没有增益，则应删除关联图，而不是仅扩大训练规模。

## 17. 主要评价指标

除 perplexity 外至少报告：

1. **跨窗口事实召回率**：相关信息离开局部窗口后还能否恢复；
2. **旧状态误用率**：状态更新后继续回答旧值的比例；
3. **跨事件依赖准确率**：能否找到当前步骤真正依赖的历史事件；
4. **精确回看准确率**：代码、数字、标识符能否逐字恢复；
5. **记忆预算**：每千 token 写入次数、活动槽数、原文回看次数；
6. **平均任务步数与成功率**；
7. **吞吐、首 token 延迟和峰值显存**；
8. **无工具域迁移**：纯文本、数学和状态跟踪任务上是否仍有效。

## 18. 与相邻工作的边界

本提案借鉴但不等同于以下方向：

- [Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881)：跨片段传递 memory token；
- [Landmark Attention](https://arxiv.org/abs/2305.16300)：先通过 landmark 选择相关历史块；
- [Infini-attention](https://arxiv.org/abs/2404.07143)：在一个模块中结合局部注意力和有界压缩记忆；
- [Gated DeltaNet](https://arxiv.org/abs/2412.06464)：使用门控衰减和 delta rule 改善线性记忆更新；
- [Titans](https://arxiv.org/abs/2501.00663)：把短期注意力与测试时更新的神经长期记忆结合。

本文当前可检验的区别在于：

1. 通用的学习式事件候选，而不是工具专用状态；
2. 内容、显著性和原始片段指针分离；
3. 时间遗忘与冲突替代分离；
4. 有界槽上的潜在有向关联；
5. 压缩回忆与原文精确回看分成两阶段；
6. 把写入、读取和原文访问都纳入显式资源目标。

这些差异是否足以构成有价值的新架构，需要进一步文献检索和受控实验，不能仅凭形式差异下结论。

## 19. 当前尚未解决的问题

1. 学习式事件边界是否比固定 chunk 更稳定；
2. 软槽写入是否会造成语义混合，硬分配是否导致高方差；
3. 关联图是否真的优于直接内容检索；
4. 显著性衰减能否从 LM 损失中学出，而不依赖大量合成监督；
5. 1B 模型是否有足够容量同时学习语言、推理和记忆控制；
6. 原始片段存储应使用 KV、token ID 还是外部文本索引；
7. 长任务训练时如何在吞吐与跨片段梯度之间折中；
8. 持久状态在不同任务、会话和用户之间应如何重置或迁移。

## 20. 当前建议

不要立即在 1B 模型上实现完整结构。先在约 50M–150M 参数模型上完成 A0–A4，对以下三个最小任务做验证：

1. 多次覆写变量后的最终状态查询；
2. 远距离 key-value 召回与干扰项；
3. 动作—观察—修正链中对早期原因的回看。

只有在“固定记忆预算下，提高远程召回并降低旧状态误用”同时成立后，再加入关联图和原始片段接口。
