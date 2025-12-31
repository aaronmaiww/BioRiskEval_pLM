#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, matthews_corrcoef, ndcg_score
from typing import Tuple
import traceback
import time
import wandb

# Import our scoring functions and model loader
from bioriskeval.common import (
    parse_model_tier, 
    parse_model_size, 
    load_esm2_model,
    compute_pseudo_ppl_hf_batch,
    cleanup_gpu_memory,
    setup_model_optimizations,
)

# Performance optimizations
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


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


def score_dms_dataset(
    dms_df: pd.DataFrame,
    ckpt_path: str,
    batch_size: int = 256,
    aggregate: str = "sum",
    max_seq_len: int = 1024,
    mask_chunk_size: int = 512,
    num_workers: int = 4,
    use_compile: bool = False,
    use_flash_attn: bool = True,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Score a DMS dataset using ESM2 pseudo-perplexity with optimized batch processing.
    
    Args:
        dms_df: DataFrame with DMS data
        ckpt_path: HuggingFace model name or path to local weights file (.pt or .pth)
        batch_size: Batch size for processing
        aggregate: "sum" for total log-likelihood, "mean" for average log-likelihood
        max_seq_len: Maximum sequence length
        mask_chunk_size: Number of masked positions evaluated per forward
        num_workers: Number of DataLoader workers for prefetching
        use_compile: Use torch.compile() for optimization
        use_flash_attn: Use Flash Attention 2 if available
    
    Returns:
        Tuple[pd.DataFrame, float, float]: (DMS dataframe with added 'esm2_pseudo_ppl' column, 
                                            model_load_time, scoring_time)
    """
    
    # Check required columns
    required_cols = ['mutated_sequence', 'DMS_score', 'DMS_score_bin']
    missing_cols = [col for col in required_cols if col not in dms_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Load model using optimized loader
    print("Loading model...")
    model_load_start = time.time()
    model, tokenizer = load_esm2_model(
        ckpt_path=ckpt_path,
        use_flash_attn=use_flash_attn
    )
    
    # Apply optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = setup_model_optimizations(model, device, use_compile)
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.2f}s")

    # Score sequences using optimized batch processing
    print(f"Scoring {len(dms_df)} sequences...")
    sequences = dms_df['mutated_sequence'].tolist()
    
    # Calculate sequence statistics for logging
    seq_lengths = [len(seq) for seq in sequences]
    seq_stats = {
        "total_sequences": len(sequences),
        "min_length": min(seq_lengths) if seq_lengths else 0,
        "max_length": max(seq_lengths) if seq_lengths else 0,
        "mean_length": np.mean(seq_lengths) if seq_lengths else 0,
        "median_length": np.median(seq_lengths) if seq_lengths else 0,
    }
    
    # Log sequence statistics to wandb
    wandb.log({
        "model_load_time": model_load_time,
        "sequence_stats/total_sequences": seq_stats["total_sequences"],
        "sequence_stats/min_length": seq_stats["min_length"],
        "sequence_stats/max_length": seq_stats["max_length"],
        "sequence_stats/mean_length": seq_stats["mean_length"],
        "sequence_stats/median_length": seq_stats["median_length"],
    })
    
    # Monitor GPU memory before processing
    cleanup_gpu_memory()
    
    # Use optimized batch processing with DataLoader (BF16)
    scoring_start = time.time()
    all_scores = compute_pseudo_ppl_hf_batch(
        sequences,
        model,
        tokenizer,
        aggregate=aggregate,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        mask_chunk_size=mask_chunk_size,
        num_workers=num_workers,
    )
    scoring_time = time.time() - scoring_start
    print(f"Scoring completed in {scoring_time:.2f}s ({len(sequences)/scoring_time:.2f} sequences/sec)")
    
    # Log scoring metrics
    valid_scores = [s for s in all_scores if not np.isnan(s)]
    if valid_scores:
        wandb.log({
            "scoring_metrics/total_scoring_time": scoring_time,
            "scoring_metrics/sequences_per_second": len(sequences) / scoring_time,
            "scoring_metrics/mean_score": np.mean(valid_scores),
            "scoring_metrics/median_score": np.median(valid_scores),
            "scoring_metrics/std_score": np.std(valid_scores),
            "scoring_metrics/min_score": np.min(valid_scores),
            "scoring_metrics/max_score": np.max(valid_scores),
            "scoring_metrics/valid_scores": len(valid_scores),
            "scoring_metrics/invalid_scores": len(all_scores) - len(valid_scores),
        })
    
    # Add scores to dataframe
    result_df = dms_df.copy()
    result_df['esm2_pseudo_ppl'] = all_scores
    
    return result_df, model_load_time, scoring_time


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate protein fitness using ESM2 (HuggingFace version, optimized)"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="bioriskeval/mut/data/DMS_ProteinGym_substitutions/DMS_substitutions.csv",
        help="Path to DMS CSV file (default: bioriskeval/mut/data/DMS_ProteinGym_substitutions/DMS_substitutions.csv)"
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="HuggingFace model name or path to local weights file (.pt or .pth). Examples: 'facebook/esm2_t6_8M_UR50D', 'given131/8M_T1', 'path/to/weights.pt'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for processing sequences. If not specified, automatically determined based on model size (8M:512, 35M:256, 150M:128)."
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization."
    )
    parser.add_argument(
        "--mask-chunk-size",
        type=int,
        default=512,
        help="Number of masked positions to evaluate per forward pass. Increase for more speed, decrease if memory is tight."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers for batch prefetching. Increase for better CPU-GPU overlap."
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="sum",
        choices=["sum", "mean"],
        help="Aggregation method: 'sum' for total log-likelihood, 'mean' for average log-likelihood."
    )
    parser.add_argument(
        "--use-compile",
        action="store_true",
        default=False,
        help="Use torch.compile() for model optimization (PyTorch 2.0+). ~20-40%% faster after warmup."
    )
    parser.add_argument(
        "--use-flash-attn",
        action="store_true",
        default=True,
        help="Use Flash Attention 2 if available. Requires: pip install flash-attn --no-build-isolation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_hf",
        help="Output directory"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)"
    )
    
    args = parser.parse_args()
    
    # Auto-determine batch size if not specified
    
    # Generate wandb run name
    model_name = args.ckpt_path.split("/")[-1]
    wandb_run_name = f"{model_name}"
    
    # Initialize wandb
    wandb.init(
        project="esm2-fitness-eval",
        name=wandb_run_name,
        config={
            "ckpt_path": args.ckpt_path,
            "csv_path": args.csv_path,
            "batch_size": args.batch_size,
            "max_seq_len": args.max_seq_len,
            "mask_chunk_size": args.mask_chunk_size,
            "num_workers": args.num_workers,
            "n_samples": args.n_samples,
            "output_dir": args.output_dir,
        },
        tags=["fitness_eval", "dms",
              f"trained_on_{parse_model_tier(args.ckpt_path)}",
              f"size_{parse_model_size(args.ckpt_path)}"]
    )
    
    try:
        # Print configuration
        print("=" * 60)
        print("ESM2 Fitness Evaluation (Optimized)")
        print("=" * 60)
        print(f"Model: {args.ckpt_path}")
        print(f"CSV path: {args.csv_path}")
        print(f"Batch size: {args.batch_size}")
        print(f"Max sequence length: {args.max_seq_len}")
        print(f"Mask chunk size: {args.mask_chunk_size}")
        print(f"DataLoader workers: {args.num_workers}")
        print("Precision: BF16 (hardcoded)")
        print(f"Use torch.compile: {args.use_compile}")
        print(f"Use Flash Attention 2: {args.use_flash_attn}")
        print(f"Aggregation method: {args.aggregate}")
        print("=" * 60)
        
        # Load DMS data
        print(f"\nLoading DMS data from: {args.csv_path}")
        load_start = time.time()
        dms_df = pd.read_csv(args.csv_path)
        load_time = time.time() - load_start
        print(f"Loaded {len(dms_df)} mutations in {load_time:.2f}s")
        
        # Log data load time
        wandb.log({"data_load_time": load_time})
        
        # Sample subset if requested
        if args.n_samples and args.n_samples < len(dms_df):
            print(f"Sampling {args.n_samples} mutations for testing")
            dms_df = dms_df.sample(n=args.n_samples, random_state=42).reset_index(drop=True)
            wandb.log({"sampled_n_mutations": args.n_samples})
        
        # Score sequences
        total_start = time.time()
        scored_df, model_load_time, scoring_time = score_dms_dataset(
            dms_df,
            args.ckpt_path,
            batch_size=args.batch_size,
            aggregate=args.aggregate,
            max_seq_len=args.max_seq_len,
            mask_chunk_size=args.mask_chunk_size,
            num_workers=args.num_workers,
            use_compile=args.use_compile,
            use_flash_attn=args.use_flash_attn,
        )
        total_time = time.time() - total_start
        
        # Compute performance metrics
        print("\nComputing performance metrics...")
        performance = get_performance_results(
            scored_df, 'DMS_score', 'esm2_pseudo_ppl', 'DMS_score_bin'
        )
        
        # Log performance metrics to wandb
        wandb.log({
            "performance/spearman": performance['spearman'],
            "performance/spearman_pvalue": performance['spearman_pvalue'],
            "performance/auc": performance['auc'],
            "performance/mcc": performance['mcc'],
            "performance/ndcg": performance['ndcg'],
        })
        
        # Log final metrics
        final_metrics = {
            "final_metrics/total_time": total_time,
            "final_metrics/model_load_time": model_load_time,
            "final_metrics/scoring_time": scoring_time,
            "final_metrics/data_load_time": load_time,
            "final_metrics/total_mutations": len(scored_df),
            "final_metrics/n_scored": scored_df['esm2_pseudo_ppl'].notna().sum(),
            "final_metrics/sequences_per_second": len(scored_df) / total_time,
        }
        wandb.log(final_metrics)
        
        # Print results
        print("\n" + "=" * 60)
        print("Performance Results:")
        print("=" * 60)
        for metric, value in performance.items():
            if not pd.isna(value):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: NaN")
        print("=" * 60)
        print(f"\nTotal evaluation time: {total_time:.2f}s")
        print(f"Sequences per second: {len(scored_df)/total_time:.2f}")
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save detailed results
        results_file = f"{args.output_dir}/{model_name}_results.csv"
        scored_df.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")
        
        # Save summary
        summary_file = f"{args.output_dir}/{model_name}_summary.csv"
        summary_df = pd.DataFrame([{
            'ckpt_path': args.ckpt_path,
            'n_mutations': len(scored_df),
            'n_scored': scored_df['esm2_pseudo_ppl'].notna().sum(),
            'total_time_seconds': total_time,
            'sequences_per_second': len(scored_df) / total_time,
            'batch_size': args.batch_size,
            'max_seq_len': args.max_seq_len,
            'mask_chunk_size': args.mask_chunk_size,
            'num_workers': args.num_workers,
            'use_compile': args.use_compile,
            'use_flash_attn': args.use_flash_attn,
            'aggregate': args.aggregate,
            **performance
        }])
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary saved to: {summary_file}")
        
        # Log output file info to wandb
        wandb.log({
            "output/results_file": results_file,
            "output/summary_file": summary_file,
            "output/total_results": len(scored_df)
        })
        
        # Finish wandb run
        wandb.finish()
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        wandb.finish()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())