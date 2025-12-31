# goal get esm2-ppl for fasta sequences 
# Optimized for RTX 5090 32GB

import argparse
import contextlib
import torch
from Bio import SeqIO
from typing import List, Dict, Optional, Tuple
import time
import numpy as np
import gc
import os

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmForMaskedLM
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from bioriskeval.common import parse_model_tier, parse_model_size, load_esm2_model

# Performance optimizations
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

MODELS = [
    # 8M
    "given131/8M_T1", "given131/8M_T2", "given131/8M_T5", "given131/8M_T6",
    "given131/8M_H",  "given131/8M_F",
    # 35M
    "given131/35M_T1", "given131/35M_T2", "given131/35M_T5", "given131/35M_T6",
    "given131/35M_H",  "given131/35M_F",
    # 150M
    "given131/150M_T1", "given131/150M_T2", "given131/150M_T5", "given131/150M_T6",
    "given131/150M_H",  "given131/150M_F",
]



def cleanup_gpu_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


class ProteinSequenceDataset(Dataset):
    """Dataset for protein sequences."""
    
    def __init__(self, sequences: List[str]):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_batch_tensors(
    sequences: List[str],
    tokenizer,
    max_seq_len: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], 
           Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """
    Collate function for DataLoader. Prepare all tensors for a batch on CPU.
    Returns: (expanded_input_ids, expanded_attention, positions_flat, token_targets_flat, seq_indices, batch_size)
    """
    # Tokenize on CPU
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, 
                      max_length=max_seq_len)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    batch_size = input_ids.size(0)
    seq_lengths = attention_mask.sum(dim=1)
    positions_per_seq = (seq_lengths - 2).clamp(min=0).long()
    total_positions = int(positions_per_seq.sum().item())
    
    if total_positions == 0:
        return None, None, None, None, None, batch_size
    
    # Build expanded tensors on CPU (DataLoader will handle pinning if pin_memory=True)
    expanded_input_ids = input_ids.repeat_interleave(positions_per_seq, dim=0)
    expanded_attention = attention_mask.repeat_interleave(positions_per_seq, dim=0)
    
    # VECTORIZED position tensor building (no Python loop!)
    # Create cumulative offsets for each sequence
    cumsum = torch.cat([torch.tensor([0]), positions_per_seq.cumsum(0)[:-1]])
    
    # Create all positions using vectorized operations
    positions_flat = torch.arange(total_positions, dtype=torch.long)
    seq_indices = torch.arange(batch_size).repeat_interleave(positions_per_seq)
    
    # Compute local positions within each sequence (offset by 1 for CLS token)
    local_positions = positions_flat - cumsum[seq_indices] + 1
    positions_flat = local_positions
    
    # Get target tokens using advanced indexing
    token_targets_flat = input_ids[seq_indices, positions_flat]
    
    # Apply mask token
    mask_id = tokenizer.mask_token_id
    expanded_input_ids[torch.arange(total_positions), positions_flat] = mask_id
    
    return expanded_input_ids, expanded_attention, positions_flat, token_targets_flat, seq_indices, batch_size

def compute_pseudo_ppl_hf_batch(
    sequences: List[str],
    model,
    tokenizer,
    aggregate: str = "mean",
    max_batch_size: int = 32,
    max_seq_len: int = 1024,
    mask_chunk_size: int = 512,
    num_workers: int = 4,
) -> List[float]:
    """
    Compute pseudo-perplexity for sequences using HuggingFace ESM2 model with optimized batching.
    Uses PyTorch DataLoader for efficient prefetching and multiprocessing.
    Uses BF16 precision.
    
    Args:
        sequences: List of protein sequences
        model: HuggingFace EsmForMaskedLM model
        tokenizer: HuggingFace ESM2 tokenizer
        aggregate: "sum" for total log-likelihood, "mean" for average log-likelihood
        max_batch_size: Maximum batch size for processing
        max_seq_len: Maximum sequence length
        mask_chunk_size: Number of masked positions evaluated per forward pass
        num_workers: Number of DataLoader workers for prefetching (default 4)
    Returns:
        List of perplexity scores
    """
    device = next(model.parameters()).device
    scores = []
    
    # Group sequences by similar lengths for efficient batching
    seq_groups = []
    current_group = []
    current_length = 0
    
    for seq in sequences:
        seq_len = min(len(seq) + 2, max_seq_len)  # +2 for special tokens
        if not current_group or abs(seq_len - current_length) <= 50:  # Allow 50 token difference
            current_group.append(seq)
            current_length = seq_len
        else:
            if current_group:
                seq_groups.append(current_group)
            current_group = [seq]
            current_length = seq_len
    
    if current_group:
        seq_groups.append(current_group)
    
    print(f"Processing {len(sequences)} sequences in {len(seq_groups)} length-grouped batches")
    
    for group in tqdm(seq_groups, desc="Processing sequence groups"):
        group_scores = process_sequence_group_batch(
            group,
            model,
            tokenizer,
            aggregate,
            max_batch_size,
            max_seq_len,
            device,
            mask_chunk_size,
            num_workers,
        )
        scores.extend(group_scores)
    
    return scores

def process_sequence_group_batch(
    sequences: List[str],
    model,
    tokenizer,
    aggregate: str,
    max_batch_size: int,
    max_seq_len: int,
    device,
    mask_chunk_size: int,
    num_workers: int = 4,
) -> List[float]:
    """
    Process a group of similar-length sequences in batches using DataLoader.
    Uses PyTorch DataLoader for efficient batch prefetching and multiprocessing.
    Uses BF16 precision.
    
    Args:
        num_workers: Number of workers for DataLoader prefetching (default 4)
    """
    scores = []
    
    if not sequences:
        return scores
    
    # Create dataset and dataloader
    dataset = ProteinSequenceDataset(sequences)
    
    # Create collate function with tokenizer and max_seq_len bound
    def collate_fn(batch):
        return collate_batch_tensors(batch, tokenizer, max_seq_len)
    
    # DataLoader handles prefetching automatically with num_workers
    dataloader = DataLoader(
        dataset,
        batch_size=max_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_fn,
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch 2 batches per worker
        persistent_workers=True if num_workers > 0 else False,
    )
    
    # Process batches from DataLoader
    for batch_idx, tensors in enumerate(dataloader):
        expanded_input_ids, expanded_attention, positions_flat, token_targets_flat, seq_indices, batch_size = tensors
        
        if expanded_input_ids is None:
            # Empty batch
            scores.extend([float("nan")] * batch_size)
            continue
        
        batch_scores = compute_batch_pseudo_ppl_from_tensors(
            expanded_input_ids,
            expanded_attention,
            positions_flat,
            token_targets_flat,
            seq_indices,
            batch_size,
            model,
            aggregate,
            device,
            mask_chunk_size,
        )
        scores.extend(batch_scores)
        
        # Clear GPU cache periodically
        if batch_idx % 8 == 0 and batch_idx > 0:
            cleanup_gpu_memory()
    
    return scores

def compute_batch_pseudo_ppl_from_tensors(
    expanded_input_ids: torch.Tensor,
    expanded_attention: torch.Tensor,
    positions_flat: torch.Tensor,
    token_targets_flat: torch.Tensor,
    seq_indices: torch.Tensor,
    batch_size: int,
    model,
    aggregate: str,
    device,
    mask_chunk_size: int,
) -> List[float]:
    """
    Compute pseudo-perplexity from pre-prepared tensors.
    This function only does GPU compute, allowing CPU preparation to happen in parallel.
    All GPU operations complete before any CPU transfer to minimize GPU-CPU overhead.
    Uses BF16 precision.
    """
    # Use CUDA stream for async data transfer if available
    if device.type == 'cuda':
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            expanded_input_ids = expanded_input_ids.to(device, non_blocking=True)
            expanded_attention = expanded_attention.to(device, non_blocking=True)
            positions_flat = positions_flat.to(device, non_blocking=True)
            token_targets_flat = token_targets_flat.to(device, non_blocking=True)
            seq_indices = seq_indices.to(device, non_blocking=True)
        stream.synchronize()
    else:
        expanded_input_ids = expanded_input_ids.to(device)
        expanded_attention = expanded_attention.to(device)
        positions_flat = positions_flat.to(device)
        token_targets_flat = token_targets_flat.to(device)
        seq_indices = seq_indices.to(device)
    
    total_positions = expanded_input_ids.size(0)
    
    # Pre-allocate result tensors on GPU
    log_sums = torch.zeros(batch_size, device=device, dtype=torch.float32)
    counts = torch.zeros(batch_size, device=device, dtype=torch.float32)

    # Use BF16 autocast
    autocast_enabled = device.type == "cuda"
    autocast_context = torch.cuda.amp.autocast(dtype=torch.bfloat16) if autocast_enabled else contextlib.nullcontext()

    # Process all chunks - stay on GPU
    for chunk_start in range(0, total_positions, mask_chunk_size):
        chunk_end = min(chunk_start + mask_chunk_size, total_positions)
        chunk_inputs = expanded_input_ids[chunk_start:chunk_end]
        chunk_attention = expanded_attention[chunk_start:chunk_end]
        chunk_positions = positions_flat[chunk_start:chunk_end]
        chunk_targets = token_targets_flat[chunk_start:chunk_end]
        chunk_seq_indices = seq_indices[chunk_start:chunk_end]
        chunk_batch = chunk_inputs.size(0)

        # Mark step for CUDA graph compatibility with torch.compile
        if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
            torch.compiler.cudagraph_mark_step_begin()

        with torch.inference_mode(), autocast_context:
            logits = model(chunk_inputs, attention_mask=chunk_attention).logits
            log_probs = F.log_softmax(logits.float(), dim=-1)
            token_log_probs = log_probs[
                torch.arange(chunk_batch, device=device),
                chunk_positions,
                chunk_targets
            ]

        log_sums.scatter_add_(0, chunk_seq_indices, token_log_probs)
        counts.scatter_add_(0, chunk_seq_indices, torch.ones_like(token_log_probs))
    
    # Finish ALL GPU computation before transferring to CPU
    with torch.inference_mode():
        if aggregate == "sum":
            results_tensor = log_sums
        elif aggregate == "mean":
            # mean: divide on GPU, handle division by zero
            results_tensor = torch.where(
                counts > 0,
                log_sums / counts,
                torch.tensor(float('nan'), device=device, dtype=log_sums.dtype)
            )
        else:
            raise ValueError(f"aggregate must be 'sum' or 'mean', got {aggregate}")
    
    # Single bulk GPU->CPU transfer
    scores = results_tensor.cpu().numpy().tolist()
    
    return scores


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


def eval_ppl_esm2(fasta_path: str, ckpt_path: str = "facebook/esm2_t6_8M_UR50D", 
                  batch_size: int = 256, aggregate: str = "mean",
                  max_seq_len: int = 1024, mask_chunk_size: int = 512,
                  num_workers: int = 4, use_compile: bool = False, 
                  use_flash_attn: bool = True) -> Dict[str, float]:
    """
    Evaluate perplexity of sequences in a FASTA file using ESM2 model.

    Args:
        fasta_path (str): Path to the input FASTA file.
        ckpt_path (str): HuggingFace model name (e.g., "facebook/esm2_t6_8M_UR50D") or path to local weights file (.pt or .pth)
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
    model, tokenizer = load_esm2_model(
        ckpt_path=ckpt_path, 
        use_flash_attn=use_flash_attn
    )
    
    # Move model to GPU and optimize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        "--ckpt-path",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
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
        output_path = generate_output_filename(args.ckpt_path, args.tier)
    else:
        output_path = args.output

    # Generate wandb run name: {model_name}_eval_on_{eval_tier}
    # Extract model name from path (e.g., "given131/150M_T1" -> "150M_T1")
    model_name = args.ckpt_path.split("/")[-1]
    wandb_run_name = f"{model_name}_eval_on_{args.tier}"

    # Initialize wandb
    wandb.init(
        project="esm2-gen-eval-random",
        name=wandb_run_name,
        config={
            "model_ckpt": args.ckpt_path,
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
              f"trained_on_{parse_model_tier(args.ckpt_path)}",
              f"size_{parse_model_size(args.ckpt_path)}"]
    )

    print(f"Evaluating perplexity using ESM2 model: {args.ckpt_path}")
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
        ckpt_path=args.ckpt_path,
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
        "model": args.ckpt_path,
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

