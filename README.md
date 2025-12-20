# Built on top of [BioRiskEval](https://github.com/scaleapi/BioRiskEval)


## Installation

#### Download the repository
```bash
git clone --recursive git@github.com:boyiwei/BioRiskEval.git
cd BioRiskEval
```

#### Download Dataset

Use the following script to download the BioRiskEval dataset:
```bash
cd bioriskeval
bash download_data.sh
```
You may need to first get access to the huggingface dataset before downloading. The script will download BioRiskEval-Gen into `bioriskeval/gen/data/`, BioRiskEval-Mut into `bioriskeval/mut/data/`, and BioRiskEval-Vir into `bioriskeval/vir/data/`. For BioRiskEval-Mut, we have two sets of data: `DMS_ProteinGym_substitutions` and `DMS_Probe`. `DMS_ProteinGym_substitutions` contains 16 DMS datasets collected from ProteinGym and is used for log-likelihood based evaluation. `DMS_Probe` is the dataset used for probe based evaluation. You can also generate `DMS_Probe` by running `dms/probe/create_dms_probe_dataset.py`.




## BioRiskEval
The hierarchy of BioRiskEval is:
- BioRiskEval-Gen (`bioriskeval/gen`): Sequnece modeling evaluation. Metric: Perplexity
- BioRiskEval-Mut (`bioriskeval/mut`): Mutational effect prediction evaluation. Metric: |Spearman correlation $\rho$|
- BioRiskEval-Vir (`bioriskeval/vir`): Virulence prediction evaluation. Metric: Pearson correlation, $R^2$

### BioRiskEval-Gen
The workflow of BioRiskEval-Gen evaluates protein language models on viral protein sequences using curated taxonomic exclusion tiers:

#### Step 1: Download Viral Protein Sequences
Download protein sequences for specific viral taxonomic groups using pre-curated taxid lists:

```bash
python bioriskeval/gen/download_taxid_proteins.py \
    --tier-file path/to/tier_file_taxids.txt \
    --output-dir viral_proteins_output \
    --batch-size 50
```

This script:
- Reads taxids from viral exclusion tier files (e.g., tier1_all_viruses_taxids.txt)
- Downloads protein sequences from NCBI using E-utilities API
- Creates individual FASTA files per taxid and a merged file for analysis

#### Step 2: Evaluate Perplexity with ESM2
Compute perplexity scores on the downloaded viral proteins:

```bash
python bioriskeval/gen/eval_ppl_esm2.py \
    --fasta viral_proteins_output/merged_viral_proteins.faa \
    --ckpt-path facebook/esm2_t6_8M_UR50D \
    --output results.tsv \
    --batch-size 4 \
    --custom-weights path/to/weights.pt  # Optional: load custom weights
```

#### Output Format
Results are saved as TSV files with configuration metadata:
```
# ESM2 Perplexity Evaluation Results
# timestamp: 2025-12-19 16:50:23
# model: facebook/esm2_t6_8M_UR50D
# custom_weights: None
# total_sequences: 760
#
sequence_id    perplexity
AAP04003.1     17.2836
AAQ63890.1     14.1361
```

### BioRiskEval-Mut

#### Zero-shot

The workflow of BioRiskEval-Mut under the zero-shot/loglikelihood setting is:
`eval_fitness.py` calculates log-likelihood-based scores for auto-regressive genomic models on mutational sequences, and Spearman correlation with the ground truth experimental fitness is reported for each DMS. `eval_fitness_esm2_hf.py` calculates scoring with masked marginals for ESM2 protein models.  

For ESM2 protein models, use `eval_fitness_esm2_hf.py` (HuggingFace version):
```bash
python bioriskeval/mut/eval_fitness_esm2_hf.py \
    --csv-path bioriskeval/mut/data/DMS_ProteinGym_substitutions/DMS_substitutions.csv \
    --model-name facebook/esm2_t6_8M_UR50D \
    --custom-weights path/to/weights.pt  # Optional: load custom weights
```

#### Probing

**Note: The probing modules in BioRiskEval-Mut are currently not compatible with HuggingFace ESM2 models. They require the BioNeMo Framework for representation extraction and are excluded from ESM2 workflows.**

The workflow of BioRiskEval-Mut under the probe setting is:
1. Pick $k$ numbers of mutations from each DMS to fit linear probes. Within the k mutations, 80% are used to fit and 20% are used as the validation split. Rest of the data is used as test split. `create_dms_probe_dataset.py` create the splits and saves representations for train and val splits.
2. Sweep over all layers with `sweep_dms_probe.py` to find the best layer for fitting the linear probe. Best probe based on train RMSE or validation split spearman are saved.
3. Save test representation based on the best layer with `probe_layer_utils.py` and `create_dms_probe_dataset.py`. Evaluate saved probes with `test_dms_probe.py`

We provide an example script `bioriskeval/mut/bioriskeval_mut_probe.sh` for quick start. 




### BioRiskEval-Vir

**Note: BioRiskEval-Vir has been simplified to focus exclusively on ESM2/HuggingFace workflows. BioNeMo-dependent files have been removed.**

The workflow of BioRiskEval-Vir for ESM2 models is:

1. **Create probe dataset** - Extract hidden-layer representations and create train-test split:
```bash
python bioriskeval/vir/create_virulence_probe_dataset_esm2.py \
    --model_name facebook/esm2_t6_8M_UR50D \
    --layer_number ${layer_num} \
    --dataset_path data/influenza_virulence_ld50_cleaned_BALB_C.csv \
    --output_dir probe_datasets \
    --n_samples 625 \
    --batch_size 8 \
    --custom_weights path/to/weights.pt  # Optional: load custom weights
```

**Note:** Run this script for each layer you want to evaluate (e.g., layers 0-5 for esm2_t6_8M_UR50D) to compare performance across layers.

2. **Train probe** - Train linear probe and evaluate performance:
```bash
python bioriskeval/vir/train_probe_continuous_esm2.py \
    --train_dataset_path probe_datasets/virulence_probe_dataset_facebook_esm2_t6_8M_UR50D_layer_${layer_num}_train.h5 \
    --test_dataset_path probe_datasets/virulence_probe_dataset_facebook_esm2_t6_8M_UR50D_layer_${layer_num}_test.h5 \
    --output_csv probe_results_layer_${layer_num}.csv \
    --use_closed_form
```

**Key arguments:**
- `--use_closed_form`: Use analytical solution (recommended for speed and stability)
- `--output_csv`: Results file with performance metrics (RMSE, MAE, R², Pearson correlation)
- `--num_steps` / `--learning_rate`: For iterative training (if not using closed form)

The training script evaluates on the test set and appends results to the specified CSV file.

## ESM2 Model Support

All ESM2 scripts now support loading custom weights. The custom weights feature allows you to:
- Load fine-tuned ESM2 models
- Load models with custom architectures (as long as they are compatible with the base ESM2 structure)
- Resume from training checkpoints

### Custom Weights Format

The scripts support several checkpoint formats:
- Direct state dict (`.pt`, `.pth` files)
- Checkpoints with 'model' key: `{'model': state_dict, ...}`
- Checkpoints with 'state_dict' key: `{'state_dict': state_dict, ...}`

Note: If custom weight loading fails, the scripts will fall back to using the pretrained weights with a warning message.

## Probing

The general workflow of probing is:
1. Extract the hidden-layer representations from the model
2. Train a linear probe on the train set
3. Evaluate the probe on the test set

For specific implementations, refer to the BioRiskEval-Vir section above for ESM2-compatible probing workflows.

## Reproducibility
We documented the results in `attack/analysis/`, which contains the raw results and scripts for analysis and plotting.
