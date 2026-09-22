# Small Agent Architecture Notes

Public research notes on a small Agent model of roughly 1B parameters.

Current goals focus on:

- tool calling and short-horizon task planning;
- simple Python, Shell, and code repair;
- basic mathematics and tool-assisted computation;
- state retention for long tasks under a fixed state budget;
- learning a memory eviction policy from the post-decision counterfactual future loss;
- training efficiency, inference latency, and state caching on small models;
- reproducible side-by-side comparison of new architectures with public small models.

## Current Status

The project is at the minimal-prototype definition stage. The core formulas of Utility Consolidation Memory v1 are frozen, but the final 1B backbone is not yet locked. Current status:

1. long-term memory uses immutable event records;
2. only one candidate enters the competition at a time, and only one record is evicted when capacity is exceeded;
3. during training, the utility predictor is supervised with the counterfactual future loss after the decision;
4. at inference time the future is not accessed; only utility prediction, single eviction, and standard Cross-Attention reads are performed;
5. whether the backbone is a Transformer, linear attention, or a hybrid model is still determined by small-scale experiments.

The first version deliberately adds no explicit decay, age, occupancy, novelty, conflict gate, association graph, soft write, or memory merging.

## Frozen Core

$$
\mathcal R_t=M_t\cup\lbrace c_t\rbrace
$$

$$
u_{t,i} =
\mathcal L_{\bar\theta,F}
\left(Y_t\mid B_t,\mathrm{mask}_i(\mathcal R_t)\right) -
\mathcal L_{\bar\theta,F}
\left(Y_t\mid B_t,\mathcal R_t\right)
$$

$$
\hat u_{t,i}=f_\phi(m_i,g_t,B_t,\mathcal R_t)
$$

$$
M_{t+1} =
\mathcal R_t\setminus
\left\lbrace \arg\min_i\hat u_{t,i}\right\rbrace
$$

The last formula is executed only when $|\mathcal R_t|\gt S$.

## Comparison Models

- TinyLlama-1.1B: reference for a standard small GQA architecture;
- OLMo-1B: reference for an open training process and intermediate checkpoints;
- RWKV-7: reference for fixed continuous state and the generalized Delta Rule;
- Kimi Linear/KDA: reference for efficient linear attention and continuous matrix state;
- Hymba-1.5B: reference for a public small hybrid architecture;
- Qwen2.5-Coder-1.5B: reference for code ability;
- SmolLM2-1.7B: reference for data engineering, mathematics, and instruction ability on small models.

## Documents

- [Architecture and Comparison Plan](docs/en/architecture-comparison-plan.md)
- [Utility Consolidation Memory v1: A Frozen Minimal Prototype](docs/en/utility-consolidation-memory-v1.md)
- [Early Persistent Episodic Memory Derivation (Superseded by the Simplified Version)](docs/en/candidate-persistent-episodic-memory.md)
- [Decision Records](docs/en/decisions.md)

## Formula Writing Conventions

- inline formulas use `$...$`, and display formulas are surrounded by separate `$$` lines; `\[...\]` and `\(...\)` do not render on GitHub;
- do not let a display formula contain a line that has only `=` or `-`, or Markdown parses it as a heading;
- write magnitude comparisons inside formulas as `\lt` and `\gt`, to avoid `<` and `>` being escaped;
- write curly braces inside formulas as `\lbrace` and `\rbrace`, because the backslashes of `\{` and `\}` are eaten by Markdown;
- write operator names as `\mathrm{...}`, not `\operatorname{...}`: the GitHub macro whitelist does not contain it, and the page shows a red box reading `The following macros are not allowed: operatorname`.

## Principles

- a complete traditional 1B baseline is ultimately not retrained;
- public models are used for the final side-by-side comparison;
- architectural attribution is completed in stages, from the no-language-model case, through 20M–50M and 100M–150M, to 300M–400M;
- perplexity is not compared side by side directly across different Tokenizers;
- quality, Agent success rate, memory selection regret, code/mathematical ability, and efficiency are reported together;
- the state budget must account for local KV, the short-term buffer, long-term records, and metadata at the same time;
- the direction is stopped when Future Oracle does not exceed a same-budget FIFO;
- architectural conclusions must be verified by experiments matched on parameters, data, and training tokens.

## Repository Layout

The Chinese notes are the primary documents; the English manuscript and the LaTeX manuscript are maintained alongside them:

```text
README.md                 Chinese original
README.en.md              English manuscript
docs/*.md                 Chinese original, one file per document
docs/en/*.md              English manuscript, one file per document
paper/main.tex            LaTeX manuscript (English, built with tectonic)
paper/preamble.tex        LaTeX preamble and macro definitions
paper/sections/*.tex      LaTeX manuscript sections
paper/check-section.sh    compile a single section on its own
```

Build the PDF:

```bash
cd paper
tectonic -X compile main.tex
```

The three texts hold the same content: formulas, heading levels, lists, tables and links correspond one to one. The Markdown uses the spellings GitHub can render (`$$`, `\mathrm`, `\lbrace`), while the LaTeX manuscript uses canonical LaTeX (`\[...\]`, `\operatorname`, `\{...\}`).

## License

Apache-2.0
