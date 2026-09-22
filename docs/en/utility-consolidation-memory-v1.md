# Utility Consolidation Memory v1: The Frozen Minimal Prototype

> Version: v1.0
>
> Status: the core formulas are frozen and implementation is allowed; the concrete dimensions and the training schedule are still to be determined by small-scale experiments.
>
> Goal: under a fixed state budget, learn which single long-term record should be deleted so that the future damage after the decision is minimal.
>
> Scope: this document freezes the memory management mechanism, not the full 1B backbone configuration.

## 1. Design Principles

The first version keeps only the following mechanisms:

1. the most recent events first enter a fixed-delay buffer;
2. only one candidate at a time competes for admission to long-term memory;
3. during training, the counterfactual future loss after the decision is used to generate utility labels;
4. at inference time, the utility predictor assigns the scores;
5. when long-term memory is full, only one record is evicted;
6. records stay independent and immutable; they are neither merged nor interpolated;
7. the model reads long-term memory through standard Cross-Attention.

The first version does not use explicit age, occupancy, novelty, surprise, exponential decay, a conflict gate, an association graph, soft write, or differentiable Top-K.

## 2. Notation and Timeline

- $M_t$: the set of long-term records before decision time $t$;
- $S$: the long-term memory capacity;
- $c_t$: the single candidate that enters the consolidation decision after a fixed waiting period $D$;
- $B_t$: the recent context that has already been observed at decision time and still sits in the short-term buffer;
- $g_t$: the representation of the current task goal;
- $Y_t=x_{t+1:t+F}$: the training evaluation interval of length $F$ after the decision;
- $m_i$: the immutable event record with a stable event ID, slot ID, and time information.

Each candidate takes part in consolidation at the $D$-th scheduling point after its event occurs. The predictor may read only the information that has already been observed at decision time; $Y_t$ is used only during training to construct supervision labels.

## 3. Frozen Core Formulas

### 3.1 Candidate Admission

$$
\boxed{
\mathcal R_t=M_t\cup\lbrace c_t\rbrace
}
$$

and:

$$
|\mathcal R_t|\le S+1.
$$

If several candidates are produced at the same scheduling point, they must be processed one by one in a deterministic temporal order, and the scores for the next decision are recomputed after every decision.

### 3.2 Training-Time Counterfactual Future Utility

Using a label-generating model $\bar\theta$ that is fixed or updated with a slow EMA:

$$
\boxed{
u_{t,i} =
\mathcal L_{\bar\theta,F}
\left(
Y_t\mid B_t,\mathrm{mask}_i(\mathcal R_t)
\right) -
\mathcal L_{\bar\theta,F}
\left(
Y_t\mid B_t,\mathcal R_t
\right)
}
$$

Here $\mathrm{mask}_i$ changes only the visibility of record $m_i$ and does not reorder the positions of the other records. The interpretation is as follows:

- $u_{t,i}\gt 0$: deleting $m_i$ increases the future loss, so the record should preferentially be kept;
- $u_{t,i}\approx0$: deleting it has essentially no effect;
- $u_{t,i}\lt 0$: deleting it actually reduces the future loss, so it should preferentially be evicted.

The training loss may first use the future token loss or the supervised answer loss:

$$
\mathcal L_{\bar\theta,F} =
\sum_{\tau=1}^{F}\ell_{t+\tau}.
$$

The first batch of experiments sets $\gamma=1$ and does not assume that distant information should be discounted over time.

### 3.3 Utility Prediction

$$
\boxed{
\hat u_{t,i} =
f_\phi
\left(
m_i,g_t,B_t,\mathcal R_t
\right)
}
$$

The predictor is trained by direct regression supervision:

$$
\boxed{
\mathcal L_{\mathrm{utility}} =
\sum_i
\left(
\hat u_{t,i}
-\mathrm{stopgrad}(u_{t,i})
\right)^2
}
$$

Under squared error, the population-optimal prediction is the conditional mean:

$$
f^*(m_i,I_t) =
\mathbb E[u_{t,i}\mid m_i,I_t],
\qquad
I_t=(g_t,B_t,\mathcal R_t).
$$

Therefore the job of the predictor is not to foresee the single future accurately, but to estimate the expected future loss that deleting each record causes.

### 3.4 Single Eviction

If:

$$
|\mathcal R_t|\le S,
$$

then:

$$
M_{t+1}=\mathcal R_t.
$$

Otherwise:

$$
\boxed{
M_{t+1} =
\mathcal R_t
\setminus
\left\lbrace 
\arg\min_i\hat u_{t,i}
\right\rbrace
}
$$

Under exact utility:

$$
\arg\min_i u_{t,i} =
\arg\min_i
\mathcal L_{\bar\theta,F}
\left(
Y_t\mid B_t,
\mathcal R_t\setminus\lbrace m_i\rbrace
\right),
$$

because the full-set loss is identical for all candidates. This conclusion guarantees optimality only for the current single eviction; consecutive evictions remain a greedy process that has to be rescored.

## 4. Standard Readout Interface

Long-term memory enters the backbone through standard Cross-Attention:

$$
Q=HW_Q,\qquad K=M_tW_K,\qquad V=M_tW_V,
$$

$$
\boxed{
H' =
H+
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_m}}
\right)
VW_O
}
$$

A candidate produced after a chunk is completed can influence only later tokens; it cannot be written back to recompute earlier tokens within the current chunk.

## 5. Training and Inference

### 5.1 Training

Training additionally performs counterfactual label generation:

1. the Reader is first trained to be able to use memory;
2. the Reader/event encoder is frozen or used with EMA as $\bar\theta$;
3. the full branch and the masked branch are run on the same competing set;
4. $u_{t,i}$ is generated;
5. $\mathcal L_{\mathrm{utility}}$ is used to train $f_\phi$;
6. the Reader continues to be trained by the language-model or task loss.

Hard eviction does not take part in ordinary backpropagation. The system is not trained end-to-end differentiably through discrete choices; the utility predictor relies on direct counterfactual supervision.

If $|\mathcal R_t|=S+1$, computing all labels exactly requires one full-set evaluation and $S+1$ masked evaluations. Early in training, the following must be compared:

- the full set of labels;
- randomly sampling a few records each time;
- an offline label cache;
- periodically refreshing the labels.

### 5.2 Inference

The inference stage does not access the future interval and does not compute counterfactual branches. Whenever a candidate matures, only the following is performed:

1. the records in $\mathcal R_t$ are given the predicted values $\hat u_{t,i}$;
2. if the capacity is exceeded, the single record with the lowest predicted utility is deleted;
3. subsequent tokens read the updated $M_{t+1}$ through Cross-Attention.

## 6. Hypothesized Advantages over a Conventional Transformer

### 6.1 State Size

The KV state of a full-context Transformer grows with the total history length $T$:

$$
O(Td).
$$

This architecture uses a local window $W$ and a long-term capacity $S$:

$$
O((W+S)d),
$$

which is independent of the total task length when $W,S$ are fixed.

### 6.2 Selective Retention beyond the Local Window

A sliding-window Transformer evicts by time:

$$
C_t=x_{t-W+1:t}.
$$

This architecture evicts by the predicted future deletion cost, so it can keep very old records that are still valuable and discard newer records that are useless.

### 6.3 Explicit Revision

An ordinary KV cache appends history but does not directly delete states that have gone stale. This architecture allows an old record to be explicitly evicted through utility competition when a new candidate arrives. This advantage is an inductive bias to be validated, not a conclusion that a Transformer is incapable of expressing it.

### 6.4 Costs and Disadvantages

- while the history still fits in the full context, Full Attention usually has more complete per-token evidence;
- once an eviction is wrong and the original text cannot be recovered, the information is lost permanently;
- counterfactual labels make training significantly more expensive;
- memory decisions introduce a sequential dependency across chunks, so training parallelism is weaker than in an ordinary Transformer.

## 7. Hypothesized Advantages over Linear Attention or Recurrent State Models

Methods such as KDA and RWKV continuously compress history into a fixed matrix or recurrent state. Their main advantages are a fixed state and efficient per-token inference. This architecture does not try to beat them on raw throughput; instead, it is hypothesized to have an advantage in the following situations:

1. important information is sparsely distributed over a very long task trajectory;
2. historical events need to keep an independent identity;
3. a single event needs to be deleted, inspected, or ablated on its own;
4. old and new states overwrite each other frequently, and a continuous state is prone to interference;
5. the future use can be predicted from the goal and the recent context.

Linear attention usually has a lower and smoother inference cost; this architecture reads $S$ records per token, at an approximate cost of:

$$
O(Sd_m),
$$

candidate consolidation additionally requires extra scoring. If every piece of historical information is important, if the future use is highly unpredictable, or if the task consists mainly of dense token dependencies, linear attention or Full Attention may be more suitable.

## 8. Complexity Summary

| Item | Full Attention | Linear/recurrent attention | This architecture |
|---|---:|---:|---:|
| Total history state | $O(Td)$ | Fixed | $O((W+S)d)$ |
| Per-token long-term readout | $O(Td)$ | Approximately fixed | $O(Sd_m)$ |
| History individually addressable | Yes | Usually no | Yes, limited to retained records |
| Explicit single-record eviction | KV usually does not do it | Difficult | Core operation |
| Training parallelism | Strong | Strong or scannable | Weaker |
| Extra training cost | Low | Medium | High, from counterfactual labels |
| Main failure mode | KV growth, long-context interference | State-compression interference | Wrong evictions, label cost |

## 9. First Batch of Experiments

Fix $S=32$, use fixed-size immutable event records, and process one candidate at a time. Compare:

1. FIFO: the same total state budget;
2. random eviction;
3. predicted-utility eviction;
4. Oracle single eviction with post-decision future information;
5. a Full Attention or sufficiently long-context upper bound;
6. a linear/recurrent state baseline.

The total state budget of the short-term buffer and of the long-term records must be reported together. The main metrics are:

- long-range recall;
- the stale-value misuse rate after several state overwrites;
- the success rate as the task length grows;
- selection regret;
- training FLOPs;
- inference latency, throughput, and peak GPU memory.

Diagnostic rules:

- the Oracle brings no gain either: first check the record representation, the Reader, or the definition of the future loss;
- the Oracle brings a gain but the prediction policy does not: first check the utility predictor and the training distribution;
- the prediction policy is effective but too expensive: then study label sampling, distillation, and the update frequency.

## 10. Current Research Claim

The first version does not claim to be broadly superior to a Transformer or to linear attention. The claim to be validated is:

> Under a fixed state budget, on long tasks where critical events are sparse and the state is overwritten several times, a single-eviction policy trained with the post-decision counterfactual future loss retains future-useful information more reliably than time-based eviction and continuous state compression do.

Related references:

- [Kimi Linear / KDA](https://arxiv.org/abs/2510.26692)
- [RWKV-7](https://arxiv.org/abs/2503.14456)
- [MEM1](https://arxiv.org/abs/2506.15841)
- [Memory-R2](https://arxiv.org/abs/2605.21768)
