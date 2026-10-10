# Structure-Analysis-of-LLMs-

# Inside Pythia-410M: Mechanistic Interpretability and Attention-Head Ablation

**An exploratory investigation of how factual next-token predictions emerge across transformer layers, how individual attention heads influence those predictions, and how head-importance estimates relate to language-model performance after ablation.**

**Model:** [`EleutherAI/pythia-410m`](https://huggingface.co/EleutherAI/pythia-410m) · **Frameworks:** PyTorch, Hugging Face Transformers, TransformerLens · **Methods:** Logit Lens, residual-stream analysis, activation patching, causal recovery, gradient attribution, functional head ablation · **Metrics:** target-token probability, factual continuation score, perplexity (PPL)

> **Research status — exploratory archive.** All figures and numerical results below come from saved outputs of the original notebook. The notebook is preserved unchanged; no experiments were rerun and no measurements were corrected. Different evaluations use different prompts, text samples, and code versions. Treat results as *individual experimental observations*, not a standardized benchmark or validated causal explanation.

## Contents

1. [Research question and experimental workflow](#research-question-and-experimental-workflow)
2. [Model and architecture](#model-and-architecture)
3. [Experiment 1 — Logit Lens and residual-stream analysis](#experiment-1--logit-lens-and-residual-stream-analysis)
4. [Experiment 2 — Activation patching and causal tracing](#experiment-2--activation-patching-and-causal-tracing)
5. [Experiment 3 — Causal-ranked attention-head ablation](#experiment-3--causal-ranked-attention-head-ablation)
6. [Experiment 4 — Gradient-based importance](#experiment-4--gradient-based-importance)
7. [Experiment 5 — Comparison and interpretation](#experiment-5--comparison-and-interpretation)
8. [Key findings](#key-findings)
9. [Limitations and reproducibility](#limitations-and-reproducibility)
10. [Repository and detailed reports](#repository-and-detailed-reports)

## Research question and experimental workflow

The investigation asks: **Can attention heads be ranked by their contribution to factual predictions, and can low-ranked heads be disabled while retaining useful language-model behavior?** It also examines whether causal intervention and gradient-based attribution identify similar heads.

The workflow in the notebook is:

```text
Pretrained Pythia-410M
        |
        v
Architecture inspection (24 layers × 16 heads)
        |
        v
Logit Lens + residual-stream inspection
        |
        v
Clean/corrupted prompts -> activation patching -> head recovery scores
        |                                         |
        v                                         v
Causal-ranked head sets                  Gradient-based head scores
        |                                         |
        +------------------+----------------------+
                           v
            Zero selected head outputs (hook_z)
                           |
                           v
            Perplexity / factual-score evaluation
                           |
                           v
                Compare against random heads
```

**Important terminology:** In this project, “pruning” usually means **zeroing attention-head outputs at inference time via hooks**. It does *not* remove parameters, reduce model checkpoint size, or demonstrate a runtime speedup.

## Model and architecture

The notebook loads the model first through Hugging Face's `GPTNeoXForCausalLM` for structural inspection, and later through TransformerLens's `HookedTransformer` for activation caching and intervention.

| Property | Archived configuration |
|---|---:|
| Model family | GPT-NeoX, decoder-only |
| Parameters | 405,334,016 |
| Transformer blocks | 24 |
| Attention heads / block | 16 |
| Total attention heads | 384 |
| Hidden width (`d_model`) | 1,024 |
| Head width (`d_head`) | 64 |
| Feed-forward width | 4,096 |
| Vocabulary size | 50,304 |
| Context window | 2,048 tokens |

The notebook's component-level parameter inventory reports:

| Component | Parameters | Approximate share |
|---|---:|---:|
| Embeddings | 103,022,592 | 25.42% |
| Attention | 100,810,752 | 24.87% |
| Feed-forward | 201,449,472 | 49.70% |
| LayerNorm | 51,200 | 0.01% |

The notebook also writes architecture charts to a `visualizations/` directory when run; those specific files were **not** embedded in its saved outputs and are not reproduced here. The notebook's separate *per-head parameter estimate* should not be treated as validated because of its output-projection attribution calculation.

## Experiment 1 — Logit Lens and residual-stream analysis

### Objective

Inspect how intermediate transformer representations change the probability of factual continuations, and how attention/MLP blocks contribute to residual-stream updates.

### Procedure

The example prompt is `The capital of France is`, with candidate continuations including ` Paris`, ` London`, ` Berlin`, and ` Rome`. The notebook caches layer activations, projects intermediate residual representations into vocabulary logits, and visualizes target-token probabilities and activation magnitudes.

### Original figures

**Layerwise target-token signal** (notebook cell 14):

![Layerwise target-token probability](assets/figures/02_logit_lens_cell_014_output_00.png)

**Candidate-token comparison** (cell 15):

![Candidate-token comparison](assets/figures/02_logit_lens_cell_015_output_00.png)

**Probability of ` Paris` by layer** (cell 17):

![Paris probability by layer](assets/figures/02_logit_lens_cell_017_output_00.png)

**Residual-stream magnitude** (cell 18):

![Residual stream norm](assets/figures/02_logit_lens_cell_018_output_00.png)

**Attention vs. MLP residual contributions** (cell 19):

![Attention and MLP residual updates](assets/figures/02_logit_lens_cell_019_output_00.png)

**Interpretation:** These figures describe changes in intermediate representations and candidate-token scores. They do not by themselves show where a fact is stored or which component *causes* the final prediction. The intermediate projection method and layer normalization conventions also affect Logit Lens interpretation.

## Experiment 2 — Activation patching and causal tracing

### Objective

Test how restoring selected activations from a clean execution changes a score on a corrupted execution, first at the residual-stream level and then at the attention-head level.

### Procedure

The notebook builds clean/corrupted variants of capital-city prompts, caches activations, restores selected activation slices using TransformerLens hooks, and evaluates a target-token score. The head-level experiment produces a **24 × 16** matrix of normalized recovery values.

For one saved example (cell 28), the recorded scores are:

| Quantity | Archived value |
|---|---:|
| Clean target score | 0.0237457 |
| Corrupted target score | 0.0225225 |
| Clean–corrupted difference | 0.0012232 |

The recovery calculation used in the notebook is of the form:

\[ R = \frac{s_{\mathrm{patched}} - s_{\mathrm{corrupt}}}{s_{\mathrm{clean}} - s_{\mathrm{corrupt}} + \varepsilon} \]

**Residual-stream patching map** (cell 27):

![Residual stream patching](assets/figures/03_causal_tracing_cell_027_output_00.png)

**Attention-head recovery heatmap** (cell 35):

![Attention head recovery](assets/figures/03_causal_tracing_cell_035_output_00.png)

**Mean recovery by layer** (cell 36):

![Mean recovery by layer](assets/figures/03_causal_tracing_cell_036_output_00.png)

**Mean recovery by head index** (cell 37):

![Mean recovery by head index](assets/figures/03_causal_tracing_cell_037_output_00.png)

### Heads highlighted by one archived ranking

| Head | Recorded normalized recovery |
|---|---:|
| Layer 19, Head 10 | 9.098 |
| Layer 17, Head 6 | 5.236 |
| Layer 21, Head 13 | 2.166 |
| Layer 4, Head 6 | 1.385 |

The notebook also reports a target score change from **0.0237457** to **0.0090691** after jointly ablating a selected set of high-ranked heads (cell 39).

**Interpretation and caution:** Recovery values greater than 1 are possible with this normalization; they are *not* percentages of recovered factual knowledge. Here the clean–corrupted gap is particularly small, making rankings sensitive to minor score changes. Prompt construction and token-position alignment also require review. The notebook separately collects heatmaps for seven country/capital cases (cell 44), but those saved tensors should not be assumed to constitute a validated factual benchmark.

## Experiment 3 — Causal-ranked attention-head ablation

### Objective

Rank heads by causal-recovery score, disable the lowest-ranked fraction, and compare language-model performance against randomly selected heads.

### Method

For each ablation ratio, the notebook selects head indices and registers forward hooks on `blocks.{layer}.attn.hook_z`, zeroing the selected head output vectors. Perplexity is then computed on selected text samples. The baseline is the unmodified model.

| Ablation ratio | Disabled heads | Heads remaining |
|---|---:|---:|
| 0% | 0 | 384 |
| 25% | 96 | 288 |
| 50% | 192 | 192 |
| 75% | 288 | 96 |

### Archived evaluation A — short sample (cell 61)

| Ratio | Causal-ranked PPL | Random PPL |
|---|---:|---:|
| 0% | 10.635 | 10.635 |
| 25% | 28.42 | 16.69 |
| 50% | 34.75 | 43.08 |
| 75% | 200.23 | 347.27 |

![Perplexity against ablation ratio](assets/figures/04_causal_pruning_cell_062_output_00.png)

**Reading this result:** Random ablation is better at 25% in this run; causal-ranked ablation is better at 50% and 75%. The overall increase in PPL indicates degraded language-model performance as more heads are disabled. This is one sample-specific comparison, not a general pruning guarantee.

### Archived evaluation B — different texts and random trials (cell 67)

| Ratio | Causal mean PPL | Random mean PPL |
|---|---:|---:|
| 25% | 3.636 | 3.423 |
| 50% | 72.37 | 39.16 |
| 75% | 247.70 | 242.72 |

![Separate causal versus random comparison](assets/figures/04_causal_pruning_cell_068_output_00.png)

This evaluation gives a different ordering: random ablation has lower mean PPL at all three ratios. **Do not combine evaluation A and B into one curve or report their values as directly comparable**: they use different data and experimental settings.

### Factual-continuation and per-text evaluation

The notebook also measures factual continuation scores for capital-city prompts and compares text-by-text PPL after 50% head ablation.

![Factual evaluation comparison](assets/figures/04_causal_pruning_cell_069_output_00.png)

![Per-text perplexity at 50% ablation](assets/figures/04_causal_pruning_cell_078_output_00.png)

![Perplexity change from baseline](assets/figures/04_causal_pruning_cell_080_output_00.png)

![Perplexity by ablation ratio on selected texts](assets/figures/04_causal_pruning_cell_082_output_00.png)

These plots illustrate sample dependence. The factual scores are target-token measures, not necessarily full-answer exact-match accuracy.

## Experiment 4 — Gradient-based importance

### Objective

Estimate attention-head importance from the sensitivity of language-model loss to each head's activation, then use the resulting ranking to select heads for ablation.

### Method

The notebook computes gradient-based scores over selected text samples, aggregating them into a **24 × 16** importance matrix. The later implementation uses an activation–gradient product:

\[ I_{l,h} = \operatorname{mean}\left(\left| z_{l,h} \odot \nabla_{z_{l,h}}\mathcal{L} \right|\right) \]

where `z` denotes the head output activation and `L` is language-model loss. A saved output reports `mean_grad shape: torch.Size([24, 16])`; another cell writes `mean_grad.pt` in the notebook's execution environment. That tensor file is not included in this archive.

**Interpretation:** Gradient-based sensitivity is not equivalent to causal necessity. Its ranking depends on the loss, examples, and implementation. The notebook does not contain a saved standalone figure for this section; the evidence here is its printed tensor shapes, comparison metrics, and head-overlap output.

## Experiment 5 — Comparison and interpretation

### Causal versus gradient ablation (cell 92)

For one saved comparison at **50% disabled heads**, the notebook reports:

| Method | Mean PPL |
|---|---:|
| Unablated baseline | 31.923 |
| Causal-ranked ablation | 266.212 |
| Gradient-ranked ablation | 45.988 |

On these particular evaluation texts, **gradient-ranked ablation retained substantially lower PPL than causal-ranked ablation**. This result should not be combined with the earlier causal/random experiments because their datasets and procedures differ.

### Agreement between head-importance methods (cell 103)

The notebook reports a **top-50 head overlap of 0.12**: 6 shared heads among the 50 highest-ranked heads under each metric. This indicates limited agreement for that particular ranking comparison; it does not establish that either importance metric is universally better.

### Later revised functions (cells 104–110)

The final notebook cells propose revised causal tracing, unified evaluation, and comparison plotting functions. **They do not contain a complete, verified rerun of the improved pipeline.** They are retained in the original notebook for provenance, not presented here as confirmed results.

## Key findings

1. **Internal predictions evolve across layers.** The Logit Lens and residual plots visualize how intermediate representations change, without proving where factual information is stored.
2. **Head interventions can materially affect a factual target score.** In one saved example, jointly ablating selected heads lowered the measured target-token score; the underlying recovery rankings remain sensitive to the experimental setup.
3. **Disabling more heads generally harms language-model performance.** Several saved evaluations show large PPL increases at higher ablation ratios.
4. **Causal-ranked ablation does not consistently outperform random ablation.** Different archived tests give different rankings.
5. **Gradient and causal rankings differ.** The reported top-50 overlap is 12%; in one 50%-ablation comparison, gradient-ranked ablation achieved lower PPL than causal-ranked ablation.

These are **observations from the archived notebook**, not claims of a validated head-pruning method or a general theory of factual recall.

## Limitations and reproducibility

- **Small or unstable causal denominator:** normalization by a tiny clean–corrupted gap can exaggerate recovery scores.
- **Prompt and token alignment:** clean/corrupted sequences must be aligned and the target position must be defined carefully; some notebook prompts include the answer in the input.
- **Limited data:** factual prompts and PPL evaluation samples are small and differ between experiments.
- **Not a unified benchmark:** random baselines, scoring functions, and test sets are not held constant throughout the notebook.
- **Functional rather than structural pruning:** masking `hook_z` does not remove weights or establish inference-efficiency gains.
- **Notebook state:** variables/functions are redefined across cells, and later revisions lack a verified end-to-end run.
- **Artifacts:** several figures created by `savefig` and some tensors/CSV outputs were not embedded in the notebook and cannot be recovered from its saved outputs alone.

### Running the original notebook

```bash
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook notebooks/Interpretability_of_Pythia_410-01-03.ipynb
```

The environment file lists direct dependencies without pinned versions. The original notebook also contains notebook-specific installation commands and assumes model download access. **A clean top-to-bottom execution is not guaranteed.** For details, see [Reproducibility](docs/reproducibility.md) and [Scientific limitations](docs/limitations.md).

## Repository and detailed reports

The README intentionally gives the **complete project story before the per-experiment documentation**. The linked reports provide source-cell mappings, additional archived images, and implementation details.

| Resource | Purpose |
|---|---|
| [Original notebook](notebooks/Interpretability_of_Pythia_410-01-03.ipynb) | Original code and saved outputs, unchanged |
| [Architecture report](docs/01_model_architecture.md) | Model inspection and component inventory |
| [Logit Lens report](docs/02_logit_lens.md) | Intermediate logits and residual analyses |
| [Causal tracing report](docs/03_causal_tracing.md) | Patching, recovery maps and head rankings |
| [Causal ablation report](docs/04_causal_pruning.md) | Ablation curves, PPL and factual scores |
| [Gradient importance report](docs/05_gradient_importance.md) | Gradient attribution and head rankings |
| [Comparative evaluation](docs/06_comparative_evaluation.md) | Comparison and revised experimental functions |
| [Figure catalog](results/figure_catalog.csv) | Figure filenames, source cells and descriptions |
| [Limitations](docs/limitations.md) | Scientific and implementation caveats |

### Attribution

Base model: [EleutherAI Pythia-410M](https://huggingface.co/EleutherAI/pythia-410m). This is an independent exploratory study, not an official EleutherAI release.
