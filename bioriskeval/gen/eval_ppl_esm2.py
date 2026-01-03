# goal get esm2-ppl for fasta sequences 
# Optimized for RTX 5090 32GB

import argparse
import torch
from Bio import SeqIO
from typing import List, Dict
import time
import numpy as np
import os
import wandb

from bioriskeval.common import (
    parse_model_tier, 
    parse_model_size, 
    load_esm2_model,
    cleanup_gpu_memory,
    setup_model_optimizations,
    ProteinSequenceDataset,
    collate_batch_tensors,
    compute_batch_pseudo_ppl_from_tensors,
    process_sequence_group_batch,
    compute_pseudo_ppl_hf_batch,
)

# Performance optimizations
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def generate_fasta_path(tier: str) -> str:
    """
    Generate FASTA file path from tier number.
    
    Args:
        tier (str): Tier number (e.g., '1', '2', '3')
    Returns:
        str: Path to the FASTA file
    """
    return f"/workspace/BioRiskEval_pLM/tier-list/tier{tier}_sequences.fasta"

def generate_output_filename(model_name: str, tier: str) -> str:
    """
    Generate output filename from model name and tier.
    
    Args:
        model_name (str): Model name (e.g., 'given131/8M_T1' or 'facebook/esm2_t6_8M_UR50D')
        tier (str): Tier number
    Returns:
        str: Output filename
    """
    # Clean model name for filename (replace / with _)
    clean_model_name = model_name.replace("/", "_")
    return f"output/{clean_model_name}_eval_on_{tier}.txt"

def load_sequences_from_fasta(fasta_path: str) -> tuple[List[str], List[str]]:
    """
    Load sequences from a FASTA file.

    Args:
        fasta_path (str): Path to the FASTA file.
    Returns:
        tuple: (sequences, seq_ids) - lists of sequences and their IDs
    """
    sequences = []
    seq_ids = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
        seq_ids.append(record.id)
    return sequences, seq_ids


def eval_ppl_esm2(fasta_path: str, model_name: str, 
                  batch_size: int = 256, aggregate: str = "mean",
                  max_seq_len: int = 1024, mask_chunk_size: int = 512,
                  num_workers: int = 4, use_compile: bool = False, 
                  use_flash_attn: bool = True) -> Dict[str, float]:
    """
    Evaluate perplexity of sequences in a FASTA file using ESM2 model.

    Args:
        fasta_path (str): Path to the input FASTA file.
        model_name (str): HuggingFace model name (e.g., "facebook/esm2_t6_8M_UR50D") or path to local weights file (.pt or .pth)
        batch_size (int): Batch size for processing sequences.
        aggregate (str): "sum" for total log-likelihood, "mean" for average log-likelihood
        mask_chunk_size (int): Number of masked positions evaluated per forward.
        num_workers (int): Number of DataLoader workers for prefetching (default 4).
        use_compile (bool): Use torch.compile() for model optimization (PyTorch 2.0+).
        use_flash_attn (bool): Use Flash Attention 2 if available.
    Returns:
        dict: A dictionary mapping sequence IDs to their perplexity scores.
    """
    start_time = time.time()
    
    # Load ESM2 model using HuggingFace
    print("Loading model...")
    model_load_start = time.time()
    model, tokenizer = load_esm2_model(model_name=model_name) # model_name is a HuggingFace model name (e.g., "facebook/esm2_t6_8M_UR50D") or path to local weights file (.pt or .pth)
    
    # Apply optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = setup_model_optimizations(model, device, use_compile)
    model_load_time = time.time() - model_load_start
    
    # Load sequences from FASTA file
    print("Loading sequences...")
    sequences, seq_ids = load_sequences_from_fasta(fasta_path)
    
    # Calculate sequence statistics
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

    results = {}
    all_perplexities = []
    processing_times = []

    # Use optimized batch processing
    print(f"Processing {len(sequences)} sequences with batch size {batch_size}")
    print(f"Using PyTorch DataLoader with {num_workers} workers for prefetching...")
    
    batch_start_time = time.time()
    
    # Monitor GPU memory before processing
    cleanup_gpu_memory()
    
    # Compute all scores using optimized batch processing with DataLoader (BF16)
    batch_scores = compute_pseudo_ppl_hf_batch(
        sequences,
        model,
        tokenizer,
        aggregate=aggregate,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        mask_chunk_size=mask_chunk_size,
        num_workers=num_workers,
    )
    
    batch_time = time.time() - batch_start_time
    processing_times.append(batch_time)
    
    # Convert scores to perplexities using vectorized operations
    # Convert log-likelihood to perplexity: exp(-log_likelihood)
    scores_array = np.array(batch_scores)
    perplexities_array = np.exp(-scores_array)
    
    # Store results in dictionary
    for seq_id, perplexity in zip(seq_ids, perplexities_array):
        results[seq_id] = float(perplexity)
    
    # Collect valid perplexities for statistics
    valid_mask = ~np.isnan(perplexities_array)
    batch_perplexities = perplexities_array[valid_mask].tolist()
    all_perplexities.extend(batch_perplexities)
    
    # Log batch metrics
    if batch_perplexities:
        wandb.log({
            "batch_metrics/total_batch_time": batch_time,
            "batch_metrics/sequences_per_second": len(sequences) / batch_time,
            "batch_metrics/mean_perplexity": np.mean(batch_perplexities),
            "batch_metrics/median_perplexity": np.median(batch_perplexities),
        })

    # Calculate final statistics
    total_time = time.time() - start_time
    
    if all_perplexities:
        final_stats = {
            "final_metrics/total_time": total_time,
            "final_metrics/mean_batch_time": np.mean(processing_times),
            "final_metrics/total_sequences_processed": len(results),
            "final_metrics/sequences_per_second": len(results) / total_time,
            "final_metrics/mean_perplexity": np.mean(all_perplexities),
            "final_metrics/median_perplexity": np.median(all_perplexities),
            "final_metrics/std_perplexity": np.std(all_perplexities),
            "final_metrics/min_perplexity": np.min(all_perplexities),
            "final_metrics/max_perplexity": np.max(all_perplexities),
            "final_metrics/valid_sequences": len(all_perplexities),
            "final_metrics/invalid_sequences": len(results) - len(all_perplexities),
        }
        
        # Log final metrics
        wandb.log(final_stats)
        
        print(f"\nEvaluation completed in {total_time:.2f}s")
        print(f"Processed {len(results)} sequences ({len(all_perplexities)} valid)")
        print(f"Mean perplexity: {np.mean(all_perplexities):.4f}")
        print(f"Median perplexity: {np.median(all_perplexities):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate perplexity of sequences using ESM2 models."
    )
    parser.add_argument(
        "--tier",
        type=str,
        required=True,
        help="Tier number for sequences (e.g., '1', '2', '3'). Will load from /workspace/BioRiskEval_pLM/tier-list/tier{tier}_sequences.fasta",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="HuggingFace model name or path to local weights file (.pt or .pth). Examples: 'facebook/esm2_t6_8M_UR50D', 'given131/8M_T1', 'path/to/weights.pt'.",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="mean",
        choices=["sum", "mean"],
        help="Aggregation method: 'sum' for total log-likelihood, 'mean' for average log-likelihood.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for processing sequences. Larger values use more GPU memory but are faster. Try 512-1024 for 32GB GPU.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--mask-chunk-size",
        type=int,
        default=512,
        help="Number of masked positions to evaluate per forward pass. Increase for more speed, decrease if memory is tight.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers for batch prefetching (default 4). Increase for better CPU-GPU overlap.",
    )
    parser.add_argument(
        "--use-compile",
        action="store_true",
        default=True,
        help="Use torch.compile() for model optimization (PyTorch 2.0+). ~20-40%% faster after warmup.",
    )
    parser.add_argument(
        "--use-flash-attn",
        action="store_true",
        default=True,
        help="Use Flash Attention 2 if available. Requires: pip install flash-attn --no-build-isolation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output file to save perplexity results. If not provided, will auto-generate as {model_name}_eval_on_{tier}.txt",
    )
 
    args = parser.parse_args()

    # Generate FASTA path from tier
    fasta_path = generate_fasta_path(args.tier)
    
    # Generate output filename if not provided
    os.makedirs("output", exist_ok=True)
    if args.output is None:
        output_path = generate_output_filename(args.model_name, args.tier)
    else:
        output_path = args.output

    # Generate wandb run name: {model_name}_eval_on_{eval_tier}
    # Extract model name from path (e.g., "given131/150M_T1" -> "150M_T1")
    model_name = args.model_name.split("/")[-1]
    wandb_run_name = f"{model_name}_eval_on_{args.tier}"

    # Initialize wandb
    wandb.init(
        project="esm2-gen-eval-random",
        name=wandb_run_name,
        config={
            "model_name": args.model_name,
            "eval_tier": args.tier,
            "batch_size": args.batch_size,
            "max_seq_len": args.max_seq_len,
            "mask_chunk_size": args.mask_chunk_size,
            "precision": "bf16",  # Always BF16
            "use_compile": args.use_compile,
            "use_flash_attn": args.use_flash_attn,
            "num_workers": args.num_workers,
            "aggregation": args.aggregate,
            "fasta_path": fasta_path,
            "output_path": output_path,
        },
        tags=[f"tier_{args.tier}",
              f"trained_on_{parse_model_tier(args.model_name)}",
              f"size_{parse_model_size(args.model_name)}"]
    )

    print(f"Evaluating perplexity using ESM2 model: {args.model_name}")
    print(f"Tier: {args.tier}")
    print(f"Input FASTA: {fasta_path}")
    print(f"Output file: {output_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max sequence length: {args.max_seq_len}")
    print(f"Mask chunk size: {args.mask_chunk_size}")
    print(f"DataLoader workers: {args.num_workers}")
    print("Precision: BF16 (hardcoded)")
    print(f"Use torch.compile: {args.use_compile}")
    print(f"Use Flash Attention 2: {args.use_flash_attn}")
    print(f"Aggregation method: {args.aggregate}")

    results = eval_ppl_esm2(
        fasta_path=fasta_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        aggregate=args.aggregate,
        max_seq_len=args.max_seq_len,
        mask_chunk_size=args.mask_chunk_size,
        num_workers=args.num_workers,
        use_compile=args.use_compile,
        use_flash_attn=args.use_flash_attn,
    )
    
    # Save results with config details
    import datetime
    
    # Create config info
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config_info = {
        "timestamp": timestamp,
        "model": args.model_name,
        "tier": args.tier,
        "fasta_file": fasta_path,
        "batch_size": args.batch_size,
        "mask_chunk_size": args.mask_chunk_size,
        "precision": "bf16",
        "aggregation": args.aggregate,
        "total_sequences": len(results)
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        # Write config header as comments
        f.write("# ESM2 Perplexity Evaluation Results\n")
        for key, value in config_info.items():
            f.write(f"# {key}: {value}\n")
        f.write("#\n")
        
        # Write data header and results
        f.write("sequence_id\tperplexity\n")
        for seq_id, ppl in results.items():
            f.write(f"{seq_id}\t{ppl}\n")
    
    print(f"Perplexity results saved to {output_path}")
    print(f"Processed {len(results)} sequences.")
    
    # Log final output file info to wandb
    wandb.log({
        "output_file": output_path,
        "total_results": len(results)
    })
    
    # Finish wandb run
    wandb.finish()
    

if __name__ == "__main__":
    main()

