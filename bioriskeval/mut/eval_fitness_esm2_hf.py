#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, matthews_corrcoef, ndcg_score
from pathlib import Path
from typing import Optional
import traceback
import time

# HuggingFace imports for ESM2
from transformers import AutoTokenizer, EsmForMaskedLM

# Import our scoring functions and model loader
from bioriskeval.gen.eval_ppl_esm2 import (
    compute_pseudo_ppl_hf_batch,
    load_esm2_model,
    cleanup_gpu_memory
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


def setup_model_optimizations(model, device, use_compile: bool = False):
    """
    Apply performance optimizations to the model.
    
    Args:
        model: The ESM2 model
        device: torch device
        use_compile: Whether to use torch.compile
    Returns:
        Optimized model
    """
    # Move model to GPU and optimize
    model = model.to(device)
    
    # Enable mixed precision and optimizations (BF16 only)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Always use BF16
        model = model.to(dtype=torch.bfloat16)
        print(f"Model loaded on {device} with BF16 precision")
        
        torch.backends.cudnn.benchmark = True
        
        # Enable PyTorch native SDPA optimizations (fallback/complement to Flash Attention 2)
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)  # Disable slow math fallback
            print("PyTorch SDPA optimizations enabled")
    
    # Apply torch.compile for optimized execution (PyTorch 2.0+)
    if use_compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        compile_start = time.time()
        # Use mode="default" instead of "reduce-overhead" to avoid CUDA Graph issues
        # with ESM's rotary embeddings which have dynamic cached tensors
        model = torch.compile(model, mode="default", fullgraph=False, dynamic=True)
        print(f"Model compiled in {time.time() - compile_start:.2f}s")
    
    return model


def score_dms_dataset(
    dms_df: pd.DataFrame,
    model_name: str,
    batch_size: int = 256,
    custom_weights_path: Optional[str] = None,
    aggregate: str = "sum",
    max_seq_len: int = 1024,
    mask_chunk_size: int = 512,
    num_prefetch: int = 2,
    use_compile: bool = False,
    use_flash_attn: bool = True,
):
    """
    Score a DMS dataset using ESM2 pseudo-perplexity with optimized batch processing.
    
    Args:
        dms_df: DataFrame with DMS data
        model_name: HuggingFace model name
        batch_size: Batch size for processing
        custom_weights_path: Path to custom weights file
        aggregate: "sum" for total log-likelihood, "mean" for average log-likelihood
        max_seq_len: Maximum sequence length
        mask_chunk_size: Number of masked positions evaluated per forward
        num_prefetch: Number of batches to prefetch in background
        use_compile: Use torch.compile() for optimization
        use_flash_attn: Use Flash Attention 2 if available
    
    Returns:
        pd.DataFrame: DMS dataframe with added 'esm2_pseudo_ppl' column
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
        ckpt_path=model_name,
        custom_weights_path=custom_weights_path,
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
    
    # Monitor GPU memory before processing
    cleanup_gpu_memory()
    
    # Use optimized batch processing with prefetching (BF16)
    scoring_start = time.time()
    all_scores = compute_pseudo_ppl_hf_batch(
        sequences,
        model,
        tokenizer,
        aggregate=aggregate,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        mask_chunk_size=mask_chunk_size,
        num_prefetch=num_prefetch,
    )
    scoring_time = time.time() - scoring_start
    print(f"Scoring completed in {scoring_time:.2f}s ({len(sequences)/scoring_time:.2f} sequences/sec)")
    
    # Add scores to dataframe
    result_df = dms_df.copy()
    result_df['esm2_pseudo_ppl'] = all_scores
    
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate protein fitness using ESM2 (HuggingFace version, optimized)"
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
        help="HuggingFace ESM2 model name. Examples: 'facebook/esm2_t6_8M_UR50D', 'given131/8M_T1', 'given131/35M_H', 'given131/150M_F'."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for processing sequences. Larger values use more GPU memory but are faster. Try 512-1024 for 32GB GPU."
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
        "--num-prefetch",
        type=int,
        default=2,
        help="Number of batches to prefetch in background. Increase for better GPU utilization."
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
    parser.add_argument(
        "--custom-weights",
        type=str,
        default=None,
        help="Path to custom weights file (.pt or .pth) to load into the model."
    )
    
    args = parser.parse_args()
    
    try:
        # Print configuration
        print("=" * 60)
        print("ESM2 Fitness Evaluation (Optimized)")
        print("=" * 60)
        print(f"Model: {args.model_name}")
        print(f"CSV path: {args.csv_path}")
        print(f"Batch size: {args.batch_size}")
        print(f"Max sequence length: {args.max_seq_len}")
        print(f"Mask chunk size: {args.mask_chunk_size}")
        print(f"Prefetch batches: {args.num_prefetch}")
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
        
        # Sample subset if requested
        if args.n_samples and args.n_samples < len(dms_df):
            print(f"Sampling {args.n_samples} mutations for testing")
            dms_df = dms_df.sample(n=args.n_samples, random_state=42).reset_index(drop=True)
        
        # Score sequences
        total_start = time.time()
        scored_df = score_dms_dataset(
            dms_df,
            args.model_name,
            batch_size=args.batch_size,
            custom_weights_path=args.custom_weights,
            aggregate=args.aggregate,
            max_seq_len=args.max_seq_len,
            mask_chunk_size=args.mask_chunk_size,
            num_prefetch=args.num_prefetch,
            use_compile=args.use_compile,
            use_flash_attn=args.use_flash_attn,
        )
        total_time = time.time() - total_start
        
        # Compute performance metrics
        print("\nComputing performance metrics...")
        performance = get_performance_results(
            scored_df, 'DMS_score', 'esm2_pseudo_ppl', 'DMS_score_bin'
        )
        
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
            'total_time_seconds': total_time,
            'sequences_per_second': len(scored_df) / total_time,
            'batch_size': args.batch_size,
            'max_seq_len': args.max_seq_len,
            'mask_chunk_size': args.mask_chunk_size,
            'num_prefetch': args.num_prefetch,
            'use_compile': args.use_compile,
            'use_flash_attn': args.use_flash_attn,
            'aggregate': args.aggregate,
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