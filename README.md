# Mechanistic Interpretability of Pythia-410M

An exploratory notebook investigating internal representations, causal activation patching, attention-head importance and functional head ablation in `EleutherAI/pythia-410m`.

## Research scope

- Model architecture and parameter distribution
- Layerwise Logit Lens and residual-stream inspection
- Clean/corrupted prompt activation patching
- Attention-head causal recovery and head-output ablation
- Gradient-based importance and causal/gradient/random comparisons

## Experiments and figures

| Section | Notebook cells | Saved figures | Documentation |
|---|---:|---:|---|
| Model Architecture Analysis | 0–9 | 0 | [Read report](docs/01_model_architecture.md) |
| Logit Lens and Residual Stream | 10–19 | 6 | [Read report](docs/02_logit_lens.md) |
| Activation Patching and Causal Tracing | 20–45 | 5 | [Read report](docs/03_causal_tracing.md) |
| Causal-Importance Head Ablation | 46–83 | 15 | [Read report](docs/04_causal_pruning.md) |
| Gradient-Based Head Importance | 84–103 | 0 | [Read report](docs/05_gradient_importance.md) |
| Later Revised Experimental Functions | 104–110 | 0 | [Read report](docs/06_comparative_evaluation.md) |

## Selected saved visualizations

![Archived output from notebook cell 12](assets/figures/02_logit_lens_cell_012_output_00.png)

![Archived output from notebook cell 14](assets/figures/02_logit_lens_cell_014_output_00.png)

![Archived output from notebook cell 15](assets/figures/02_logit_lens_cell_015_output_00.png)

![Archived output from notebook cell 17](assets/figures/02_logit_lens_cell_017_output_00.png)

## Notebook

[Open the original notebook](notebooks/Interpretability_of_Pythia_410-01-03.ipynb). The original file is preserved byte-for-byte.

## Reproduction notes

The notebook contains installation commands and assumes access to the Pythia-410M model and compatible PyTorch/TransformerLens/Hugging Face versions. Cells redefine some functions and depend on prior state. The repository does **not** guarantee a clean top-to-bottom run. See [environment notes](docs/reproducibility.md).

## Scientific limitations

- Several causal-recovery scores depend on small clean–corrupt gaps and may be unstable.
- Prompt construction, token alignment and scoring choices need careful review.
- Perplexity comparisons use different text samples and implementations across notebook sections.
- Zeroing head activations is functional ablation, not physical parameter removal or model compression.
- The later revised functions are documented but not presented as verified rerun results.

These caveats are discussed further in [Limitations](docs/limitations.md).

## Repository layout

```text
notebooks/         Original notebook, unchanged
docs/              Per-section experiment inventories and notes
assets/figures/    PNGs extracted from saved notebook outputs
results/           Figure-to-cell manifest
requirements.txt   Direct dependency list (versions unpinned)
```

## Attribution

Model: [EleutherAI Pythia](https://huggingface.co/EleutherAI/pythia-410m). This repository documents an independent exploratory analysis and is not an official EleutherAI release.
