# Built on top of [BioRiskEval](https://github.com/scaleapi/BioRiskEval)


## Installation

### Build the docker image and develop environment

Following the same steps in [Getting Started with BioNeMo Framework](https://github.com/NVIDIA/bionemo-framework?tab=readme-ov-file#getting-started-with-bionemo-framework), you can run the following scrips to clone the reposiotry and build the docker image:

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

The workflow of BioRiskEval-Mut under the probe setting is:
1. Pick $k$ numbers of mutations from each DMS to fit linear probes. Within the k mutations, 80% are used to fit and 20% are used as the validation split. Rest of the data is used as test split. `create_dms_probe_dataset.py` create the splits and saves representations for train and val splits.
2. Sweep over all layers with `sweep_dms_probe.py` to find the best layer for fitting the linear probe. Best probe based on train RMSE or validation split spearman are saved.
3. Save test representation based on the best layer with `probe_layer_utils.py` and `create_dms_probe_dataset.py`. Evaluate saved probes with `test_dms_probe.py`

We provide an example script `bioriskeval/mut/bioriskeval_mut_probe.sh` for quick start. 




### BioRiskEval-Vir
The workflow of BioRiskEval-Vir is:
1. Extract hidden-layer representations, create train-test split (1:9) for probing.
2. Train a linear probe on the train set, and evaluate on the test set. `train_probe_continuous.py` will train the linear probe and evaluate its performance on the test set. It will also uplaod the results to Weights & Biases and dumpe the results to a csv file.

For ESM2 protein models, use the ESM2-specific scripts:

1. Create probe dataset:
```bash
python bioriskeval/vir/create_virulence_probe_dataset_esm2.py \
    --model_name facebook/esm2_t6_8M_UR50D \
    --layer_number 4 \
    --custom_weights path/to/weights.pt  # Optional: load custom weights
```

2. Train probe:
```bash
python bioriskeval/vir/train_probe_continuous_esm2.py \
    --train_dataset_path probe_datasets/train.h5 \
    --test_dataset_path probe_datasets/test.h5 \
    --custom_weights path/to/weights.pt  # Optional: for reference only
```

We have provided an example script `bioriskeval/vir/bioriskeval_vir.sh` for quick start.

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

### Usage Examples

```bash
# Generation evaluation with custom weights
python bioriskeval/gen/eval_ppl_esm2.py \
    --fasta sequences.fasta \
    --ckpt-path facebook/esm2_t6_8M_UR50D \
    --custom-weights /path/to/custom_weights.pt \
    --output results.tsv

# Mutation fitness evaluation with custom weights
python bioriskeval/mut/eval_fitness_esm2_hf.py \
    --csv-path data/mutations.csv \
    --model-name facebook/esm2_t6_8M_UR50D \
    --custom-weights /path/to/custom_weights.pt

# Virulence probe dataset creation with custom weights
python bioriskeval/vir/create_virulence_probe_dataset_esm2.py \
    --model_name facebook/esm2_t6_8M_UR50D \
    --custom_weights /path/to/custom_weights.pt
```

Note: If custom weight loading fails, the scripts will fall back to using the pretrained weights with a warning message.

## Fine-Tuning & Probing

### Fine-tuning
Inside `attack/`, we have the scripts for fine-tuning.

The workflow of fine-tuning is:
1. Have the csv file with accession ids in column `#Accession`
2. Convert the csv file to fna file using `convert_csv_to_fna.py`
3. Create the train-val split, tokenize the data
4. Create dataset config for fine-tuning
5. Fine-tune the model

We provide an example script in `attack/data/preprocess_ft_data.sh` (preprocess data, step 1-4) and `attack/ft/launch_ft_7b_1m.sh` (fine-tuning the model, step 5) for quick start, in which you can modify the csv file path and change the preprocess config.

### Probing

The workflow of probing is:
1. Extract the hidden-layer representations from the model
3. Train a linear probe on the train set
4. Evaluate the probe on the test set

Refer to the example scripts `bioriskeval/vir/bioriskeval_vir.sh` for quick start.

## Reproducibility
We documented the results in `attack/analysis/`, which contains the raw results and scripts for analysis and plotting.
