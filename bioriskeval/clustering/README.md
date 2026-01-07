# ESM2 Clustering Workflows

This directory hosts tooling to cluster ESM2 checkpoint embeddings across tiers and checkpoints. The intent is to compare how checkpoints trained on different tier datasets behave when applied to the same tier, and vice versa, while logging the process in `wandb`.

## What the script produces

When you run `run_esm2_clustering.py --model-size 35M` (or another supported size), the script generates **12 figures**:

1. **Tier-vs-model views** (`tier_{tier}_per_model.png`): for each tier (1–6) the embeddings for that tier's sequences are extracted from all checkpoints of the selected size, clustered with HDBSCAN, and visualized with both cluster colorings and checkpoint colorings.
2. **Model-vs-tier views** (`{model_short}_per_tier.png`): for each checkpoint the embeddings from all six tiers are pooled, clustered, and plotted with tier/color semantics.

Each figure is saved under `bioriskeval/clustering/output/clustering_{timestamp}` and uploaded to the W&B run in the `esm2-eval-clustering` project.

## Running the script

Install the optional dependencies before running the script:

```sh
pip install hdbscan matplotlib scikit-learn
```

Then invoke the script with the desired arguments. The defaults work out of the box:

```sh
python bioriskeval/clustering/run_esm2_clustering.py \
  --model-size 35M \
  --max-sequences 512 \
  --batch-size 32 \
  --min-cluster-size 30 \
  --policy filtering \
  --layer-indices -2 -1
```

(`--policy filtering` remains the default, but `--policy corruption` is also an option.)

The script loads sequences from `/workspace/BioRiskEval_pLM/tier-list/tier{N}.fasta`, processes the requested number of sequences per tier, and uses `bioriskeval/common.py` helpers to load each checkpoint via `load_esm2_model`.

### Useful arguments

- `--tiers` (default `1 2 3 4 5 6`): Limit the tiers that are read.
- `--max-sequences`: Cap the number of sequences per tier (set `0` to run on every fasta entry).
- `--use-compile`: Apply `torch.compile()` for faster inference when available.
- `--output-dir`: Override where the plots and artifacts are written.
- `--wandb-project`: Defaults to `esm2-eval-clustering`, so the run automatically logs the generated figures and cluster metrics.

## Monitoring

The script initializes a `wandb` run with the configured arguments and logs:

- Sequence counts per tier before extraction.
- Number of HDBSCAN clusters, noise ratio, and total points for every figure.
- Each figure as a `wandb.Image` with a descriptive caption.

Ensure you have `WANDB_API_KEY` configured so the run can upload to the `esm2-eval-clustering` project.

## Output expectations

- Figures include a 2-panel scatter showing both the HDBSCAN cluster assignments and the dataset label (model or tier).
- Each title clearly states the combination (e.g., `35M_T1 across tiers`).
- Output paths and run metadata are captured under `bioriskeval/clustering/output`.

With the defaults, expect 6 tier-vs-model figures and 6 model-vs-tier figures, matching the request for twelve cluster visualizations.

