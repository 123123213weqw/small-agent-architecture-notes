# 1B Agent Model Architecture and Comparison Plan

> Version: v0.3
>
> Status: the core formulas of Utility Consolidation Memory v1 are frozen; the full 1B backbone is not yet frozen.
>
> Goal: train from scratch a small Agent model of about 1B parameters, targeting tool calling, simple code, basic mathematics, and long-task state retention.

## 1. Project Goals

The model is positioned as a small Agent controller rather than a general chat model that covers all knowledge. Its core capabilities include:

1. Understanding Chinese and English tasks;
2. Generating structurally correct tool calls;
3. Writing and repairing simple code;
4. Performing basic mathematical reasoning;
5. Interpreting tool results and performing limited error correction from failures;
6. Maintaining goals, constraints, state updates, and key tool results in long tasks under a fixed state budget;
7. Producing concise, verifiable final answers.

Exact computation and program execution are delegated to external tools. The model focuses on learning task decomposition, tool selection, argument construction, result interpretation, stopping conditions, and memory trade-offs under limited capacity.

### 1.1 Scope of the First Version

- Basic arithmetic, algebra, ratios, unit conversion, and simple word problems;
- Small Python programs and basic Bash/Shell tasks;
- Reading small code files and repairing code based on unit test failures;
- Calculator, Python, Shell, file, and test tool calls;
- Controlled Agent or state-tracking trajectories of 8-64 steps;
- Final state queries after variables, constraints, file state, and tool results have been overwritten multiple times;
- Selective recall of key events once they fall outside the local window.

### 1.2 Out of Scope for the First Version

- Large repository-level software engineering;
- Difficult competition mathematics;
- Permanent personalized memory across users and across sessions;
- Multi-Agent collaboration, GUI automation, and multimodality;
- Lossless retention of histories of arbitrary length;
- Surpassing Full Attention across the board under short-context or unlimited-compute conditions.

## 2. Current Architecture Status

### 2.1 Frozen: Utility Consolidation Memory v1

Only one candidate enters long-term memory competition at a time:

$$
\mathcal R_t=M_t\cup\lbrace c_t\rbrace.
$$

During training, counterfactual future loss after the decision generates the utility labels:

$$
u_{t,i} =
\mathcal L_{\bar\theta,F}
\left(Y_t\mid B_t,\mathrm{mask}_i(\mathcal R_t)\right) -
\mathcal L_{\bar\theta,F}
\left(Y_t\mid B_t,\mathcal R_t\right).
$$

The predictor may only use information already observed at decision time:

$$
\hat u_{t,i}=f_\phi(m_i,g_t,B_t,\mathcal R_t).
$$

When capacity is exceeded, only the single record with the lowest predicted utility is evicted:

$$
M_{t+1} =
\mathcal R_t\setminus
\left\lbrace \arg\min_i\hat u_{t,i}\right\rbrace.
$$

See [Utility Consolidation Memory v1](utility-consolidation-memory-v1.md) for the complete definition.

### 2.2 Explicitly Excluded from v1

- Explicit age, occupancy, or time decay;
- Novelty, surprise, or conflict gates;
- Event association graphs;
- soft write;
- Interpolation, rewriting, or merging of memory content;
- Gumbel-Softmax or differentiable Top-K;
- Backpropagation or future information access at inference time.

### 2.3 Still Not Frozen

```text
Parameter scale: about 1.0B-1.3B
Model type: autoregressive text generation model
Training precision: BF16
Primary languages: Chinese and English
Primary domains: tool calling, simple code, basic mathematics, long-task state retention
Deployment target: inference on a single 16GB-48GB GPU
```

The following still require controlled experiments:

- Whether the backbone uses Full/Local Attention, linear attention, or a hybrid structure;
- The number of layers, hidden size, FFN width, and the layer position for memory reads;
- The local window $W$;
- The long-term capacity $S$, the waiting period $D$, and the supervision horizon $F$;
- Event record granularity, encoding dimension, and how the original text is retained;
- The concrete implementation of the Utility Predictor;
- Positional encoding, Tokenizer, data mixture, and training tokens.

Freezing the memory formulas does not mean that the final 1B backbone has been decided.

## 3. Public Models and Architecture References

### 3.1 TinyLlama-1.1B: A Standard Small Transformer

| Item | Configuration |
|---|---:|
| Parameter count | about 1.1B |
| Layers | 22 |
| Hidden size | 2048 |
| FFN size | 5632 |
| Query/KV heads | 32/4 |
| Attention | GQA |
| Activation/normalization | SwiGLU/RMSNorm |
| Positional encoding | RoPE |
| Vocabulary/context | 32K/2048 |

References: [TinyLlama official repository](https://github.com/jzhang38/tinyllama), [pretraining notes](https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md).

### 3.2 OLMo-1B: An Open Training Reference

Used to compare training curves, data efficiency, and language model capability at a similar number of tokens. References: [OLMo official repository](https://github.com/allenai/OLMo), [OLMo-1B model card](https://huggingface.co/allenai/OLMo-1B).

### 3.3 RWKV-7: A Fixed Recurrent State Reference

RWKV-7 maintains a fixed recurrent state with a generalized Delta Rule and vector gating and serves as a continuous-state control. It is used to test whether discrete, addressable event records really outperform a state matrix that continuously compresses history.

References: [RWKV-7](https://arxiv.org/abs/2503.14456), [official repository](https://github.com/BlinkDL/RWKV-LM).

### 3.4 Kimi Linear / KDA: A Linear Attention Reference

The core state update of KDA can be summarized as:

$$
S_t=(D_t-a_tb_t^\top)S_{t-1}+k_tv_t^\top.
$$

It is used to compare the quality and efficiency differences between continuous matrix state, hybrid Full Attention, and discrete event retention. Reference: [Kimi Linear](https://arxiv.org/abs/2510.26692).

Kimi K3 is a large-scale combination of KDA, Gated MLA, Attention Residuals, and Stable LatentMoE and does not serve as a strict 1B capability baseline.

### 3.5 Other Capability and Hybrid Architecture References

- [Hymba-1.5B](https://arxiv.org/abs/2411.13676): small-model hybrid architecture, caching, and long-context efficiency;
- [Qwen2.5-Coder-1.5B](https://arxiv.org/abs/2409.12186): simple code and Code Agent capability;
- [SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/blob/main/README.md): small-model data engineering, mathematics, instruction following, and Function calling.

### 3.6 Adjacent Memory Methods

- [MEM1](https://arxiv.org/abs/2506.15841): a joint reasoning-memory state of fixed capacity;
- [Memory-R2](https://arxiv.org/abs/2605.21768): local and global credit assignment for long-range memory operations;
- [Titans](https://arxiv.org/abs/2501.00663): Attention and test-time neural long-term memory;
- [Infini-attention](https://arxiv.org/abs/2404.07143): local attention and bounded compressed memory;
- [Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881): recurrent memory tokens across segments.

## 4. Comparison Responsibilities

| Subject | Main comparison content | Strict architectural control |
|---|---|---:|
| TinyLlama-1.1B | Standard small GQA, parameter efficiency | No |
| OLMo-1B | Open training process, similar token budget | No |
| RWKV-7 | Fixed continuous state, throughput, state tracking | Yes in the controlled implementation |
| KDA | Linear attention, continuous matrix state | Yes in the controlled implementation |
| Hymba/Qwen/SmolLM2 | Hybrid, code, mathematics, and Agent capability | No |
| FIFO/random eviction | Lower bound of memory policies | Yes |
| Future Oracle eviction | Offline upper bound of the Reader and the record representation | Yes |
| Predicted utility eviction | Deployable policy of this project | Yes |

## 5. Core Research Hypothesis

This project does not claim to surpass Transformers or linear attention across the board. The claim to be validated is:

> Under the same resident state budget, for trajectories where key events are sparse, tasks are very long, and state is overwritten multiple times, a single eviction policy trained with post-decision counterfactual future loss retains future-useful information more reliably than time-based eviction and continuously compressed state.

### 5.1 Relative to Full/Local Transformers

The KV state of a full-context Transformer grows with the total length:

$$
O(Td).
$$

This project uses a local window $W$ and a long-term capacity $S$:

$$
O((W+S)d).
$$

The difference from a sliding window is not whether the state is bounded, but that old records are evicted by predicted future value rather than by pure temporal order.

### 5.2 Relative to Linear or Recurrent State Models

The advantages of linear attention and RWKV/KDA are a fixed state and efficient per-token updates. The hypothesized advantages of this project are:

- Event records keep independent identities;
- A single record can be addressed, masked, and deleted;
- History does not have to be continuously mixed into the same matrix state;
- Memory trade-offs have direct counterfactual future utility supervision;
- Delayed decisions can use evidence already observed after an event occurs.

This project may be at a disadvantage in the pure throughput of linear models, in dense sequence modeling, and on tasks whose future usage is highly unpredictable.

## 6. Fair Comparison Rules

### 6.1 Public Side-by-Side Comparison and Controlled Experiments Are Kept Separate

Public models are used to answer final capability level and deployment efficiency; controlled small models are used for architectural attribution. Public models with different Tokenizers, data, and training tokens cannot be used to prove that utility memory is effective.

### 6.2 Parameters and State Budget

Controlled models match total parameters, non-Embedding parameters, or training FLOPs, and report the other items in full. The state budget must include:

$$
\text{local KV}
+\text{short-term buffer}
+\text{long-term records}
+\text{metadata}.
$$

If the short-term buffer is fully contained in the local window, this should be stated explicitly to avoid double counting.

### 6.3 Cross-Tokenizer Metrics

Perplexity is not compared directly; Bits Per Byte is used uniformly:

$$
\mathrm{BPB} =
\frac{\mathrm{NLL}}
{N_{\mathrm{bytes}}\ln2}.
$$

Report Chinese, English, code, mathematics, and API/JSON BPB separately.

## 7. Evaluation Metrics

### 7.1 General, Mathematics, Code, and Agent

- Chinese and English held-out BPB;
- Accuracy on basic mathematics and tool-assisted mathematics;
- Python pass@1, syntax correctness rate, and unit test pass rate;
- Success rate of repairing code from error logs;
- Tool-call JSON validity rate;
- Tool selection and argument construction accuracy;
- Multi-step task success rate, failure recovery rate, and final answer correctness rate.

### 7.2 Long-Range Memory

- Fact recall across the local window;
- Latest-value correctness rate after multiple state overwrites;
- Stale-value misuse rate;
- Performance degradation as irrelevant tool output increases;
- Recall under different dependency distances $D,F$;
- Record eviction accuracy;
- Rank correlation between the Utility Predictor and the Oracle;
- Selection regret of the predicted policy relative to the Oracle;
- Transfer of the memory policy on a fully held-out domain.

### 7.3 Efficiency

- Prefill and Decode tokens/s;
- First-token, single-token, and consolidation latency;
- Peak GPU memory and total state size;
- Number of consolidations per thousand tokens;
- FLOPs for counterfactual label generation;
- Total training tokens/s, per-step time, and MFU.

Offline counterfactual label cost must not be counted toward online latency at inference time; training cost must be reported in full.

## 8. Experiment Stages

### 8.1 Stage 0: Selector Validation Without a Language Model

Validate on synthetic state machines and key-value overwrite tasks:

1. That the single eviction formula and the mask implementation are correct;
2. Whether Future Oracle significantly exceeds FIFO;
3. Whether the single eviction explanation holds when records are redundant or complementary;
4. The effect of candidate order on the greedy policy.

If Oracle does not exceed FIFO, do not proceed to larger-model experiments.

### 8.2 Stage 1: 20M-50M Reader and Utility Predictor

1. First train the Reader to use external records;
2. Freeze or EMA the Reader/event encoder;
3. Generate a small number of exact counterfactual labels;
4. Train the Utility Predictor;
5. Compare FIFO, random, predicted utility, and Oracle.

### 8.3 Stage 2: 100M-150M Controlled Architecture Experiments

Match Tokenizer, data order, parameters/FLOPs, batch, optimizer, training tokens, and random seeds. Compare:

- Local Transformer + FIFO;
- Local Transformer + Utility Consolidation Memory;
- One RWKV/KDA-style continuous-state baseline;
- A Full Attention upper bound when necessary.

### 8.4 Stage 3: 300M-400M Scaling Validation

Scale only the approaches that simultaneously satisfy the following conditions:

- Oracle has a clear upper bound;
- The predicted policy is significantly close to Oracle;
- It exceeds FIFO and the continuous-state baseline at the same state budget;
- Label cost is still acceptable.

### 8.5 Stage 4: Final Model of About 1B Parameters

In the end only one candidate of about 1B parameters is trained. The Transformer, linear, or hybrid backbone is determined by the quality-efficiency curves of stages 2-3.

## 9. Draft Agent Tool Protocol

The first version is expected to support:

```text
calculator
python
shell
read_file
apply_patch
run_tests
```

Initial limits:

```yaml
max_agent_steps: 64
max_retries_per_step: 2
python_timeout_seconds: 10
shell_timeout_seconds: 10
max_tool_output_chars: 12000
```

Short capability evaluations keep an 8-step subset; memory evaluations extend progressively to 16, 32, and 64 steps.

## 10. Results Recording Template

### 10.1 Memory Selection

| Policy | Total state budget | Final success rate | Stale-value misuse rate | Selection Regret | Consolidation latency |
|---|---:|---:|---:|---:|---:|
| FIFO | | | | | |
| Random | | | | | |
| Predicted Utility | | | | | |
| Future Oracle | | | | | |

### 10.2 Architecture Quality and Efficiency

| Model | Parameters | Training Tokens | BPB | Agent success rate | Decode tok/s | State size |
|---|---:|---:|---:|---:|---:|---:|
| Local Transformer | | | | | | |
| Linear/Recurrent | | | | | | |
| Utility Memory | | | | | | |
| Full Attention Upper Bound | | | | | | |

## 11. Current Decisions and Stopping Conditions

Decided:

- Positioning as a small Agent model of about 1B parameters;
- Tool calling, simple code, basic mathematics, and long-task state retention;
- Not training a full conventional 1B baseline again;
- Public models carry the final capability reference;
- Small-scale controlled experiments carry the architectural attribution;
- The core formulas of Utility Consolidation Memory v1;
- Single-candidate admission, single-record eviction, and immutable event records;
- The first version adds no other memory gates or association modules.

Still undecided:

- The full backbone sequence-mixing structure;
- Depth, width, and the layer position for memory reads;
- $W,S,D,F$;
- Event record encoding;
- Label sampling rate and the teacher update schedule;
- Tokenizer, data mixture, and final training tokens.

Stopping conditions:

1. Future Oracle does not exceed FIFO at the same budget;
2. Predicted utility cannot approach Oracle significantly over the long run;
3. Under the same state budget, it does not exceed the sliding-window or continuous-state baseline;
4. Label cost cannot be reduced to an acceptable range through sampling, caching, or distillation.
