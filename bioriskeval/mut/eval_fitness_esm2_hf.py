#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, matthews_corrcoef, ndcg_score
from pathlib import Path
from typing import List
import traceback

# HuggingFace imports for ESM2
from transformers import AutoTokenizer, EsmForMaskedLM

# Import our scoring function
from bioriskeval.gen.eval_ppl_esm2 import compute_pseudo_ppl_hf


def get_performance_results(merged_df, DMS_score_column, model_score_column, DMS_binary_score_column):
    """
    Compute performance metrics comparing model scores with experimental DMS scores.
    
    Returns:
        dict: Performance metrics (Spearman correlation, AUC, MCC, NDCG)
    """
    # Remove missing values
    clean_df = merged_df[[DMS_score_column, model_score_column, DMS_binary_score_column]].dropna()
    
    if len(clean_df) == 0:
        return {
            'spearman': np.nan, 'spearman_pvalue': np.nan, 
            'ndcg': np.nan, 'auc': np.nan, 'mcc': np.nan
        }
    
    # Spearman correlation
    spearman_corr, spearman_pval = spearmanr(clean_df[DMS_score_column], clean_df[model_score_column])
    
    # Binary classification metrics
    if len(clean_df[DMS_binary_score_column].unique()) > 1:
        # AUC
        try:
            auc = roc_auc_score(clean_df[DMS_binary_score_column], clean_df[model_score_column])
        except:
            auc = np.nan
            
        # MCC  
        try:
            # Convert scores to binary predictions using median threshold
            threshold = clean_df[model_score_column].median()
            binary_preds = (clean_df[model_score_column] >= threshold).astype(int)
            mcc = matthews_corrcoef(clean_df[DMS_binary_score_column], binary_preds)
        except:
            mcc = np.nan
            
        # NDCG
        try:
            # NDCG expects relevance scores, use DMS_score for ranking
            ndcg = ndcg_score(
                clean_df[DMS_binary_score_column].values.reshape(1, -1),
                clean_df[model_score_column].values.reshape(1, -1)
            )
        except:
            ndcg = np.nan
    else:
        auc = mcc = ndcg = np.nan
    
    return {
        'spearman': spearman_corr,
        'spearman_pvalue': spearman_pval, 
        'ndcg': ndcg,
        'auc': auc,
        'mcc': mcc
    }


def load_esm2_model_hf(model_name: str):
    """Load ESM2 model using HuggingFace transformers."""
    print(f"Loading HuggingFace ESM2 model: {model_name}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model loaded on device: {device}")
    
    return model, tokenizer


def score_dms_dataset(dms_df: pd.DataFrame, model_name: str, batch_size: int = 8):
    """
    Score a DMS dataset using ESM2 pseudo-perplexity.
    
    Returns:
        pd.DataFrame: DMS dataframe with added 'esm2_pseudo_ppl' column
    """
    
    # Check required columns
    required_cols = ['mutated_sequence', 'DMS_score', 'DMS_score_bin']
    missing_cols = [col for col in required_cols if col not in dms_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Load model
    model, tokenizer = load_esm2_model_hf(model_name)
    
    # Score sequences
    print(f"Scoring {len(dms_df)} sequences...")
    sequences = dms_df['mutated_sequence'].tolist()
    
    # Process in batches
    all_scores = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(sequences)-1)//batch_size + 1}")
        
        try:
            batch_scores = compute_pseudo_ppl_hf(batch_seqs, model, tokenizer, aggregate="sum")
            all_scores.extend(batch_scores)
        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
            # Fill with NaN for failed batch
            all_scores.extend([float('nan')] * len(batch_seqs))
    
    # Add scores to dataframe
    result_df = dms_df.copy()
    result_df['esm2_pseudo_ppl'] = all_scores
    
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate protein fitness using ESM2 (HuggingFace version)"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to DMS CSV file"
    )
    parser.add_argument(
        "--model-name", 
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="HuggingFace ESM2 model name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_hf",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    try:
        # Load DMS data
        print(f"Loading DMS data from: {args.csv_path}")
        dms_df = pd.read_csv(args.csv_path)
        print(f"Loaded {len(dms_df)} mutations")
        
        # Score sequences
        scored_df = score_dms_dataset(dms_df, args.model_name, args.batch_size)
        
        # Compute performance metrics
        performance = get_performance_results(
            scored_df, 'DMS_score', 'esm2_pseudo_ppl', 'DMS_score_bin'
        )
        
        # Print results
        print("\nPerformance Results:")
        for metric, value in performance.items():
            print(f"  {metric}: {value:.4f}" if not pd.isna(value) else f"  {metric}: NaN")
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save detailed results
        dataset_name = Path(args.csv_path).stem
        model_name_safe = args.model_name.replace("/", "_")
        
        results_file = f"{args.output_dir}/{dataset_name}_{model_name_safe}_results.csv"
        scored_df.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")
        
        # Save summary
        summary_file = f"{args.output_dir}/{dataset_name}_{model_name_safe}_summary.csv"
        summary_df = pd.DataFrame([{
            'dataset': dataset_name,
            'model': args.model_name,
            'n_mutations': len(scored_df),
            'n_scored': scored_df['esm2_pseudo_ppl'].notna().sum(),
            **performance
        }])
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())