# goal get esm2-ppl for fasta sequences 

import argparse
import torch
from Bio import SeqIO
from typing import List, Dict, Optional
import time
import numpy as np
import gc

from transformers import AutoTokenizer, EsmForMaskedLM
import torch.nn.functional as F
import wandb
from tqdm import tqdm

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

FACEBOOK_CONFIG = {
    "8M":   "facebook/esm2_t6_8M_UR50D",
    "35M":  "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
}

def print_gpu_memory_info(stage: str = ""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"GPU Memory {stage}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Peak={max_allocated:.2f}GB")
        return allocated, reserved, max_allocated
    return 0, 0, 0

def cleanup_gpu_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def compute_pseudo_ppl_hf_batch(sequences: List[str], model, tokenizer, aggregate: str = "mean", 
                               max_batch_size: int = 32, max_seq_len: int = 1024) -> List[float]:
    """
    Compute pseudo-perplexity for sequences using HuggingFace ESM2 model with optimized batching.
    
    Args:
        sequences: List of protein sequences
        model: HuggingFace EsmForMaskedLM model
        tokenizer: HuggingFace ESM2 tokenizer
        aggregate: "sum" for total log-likelihood, "mean" for average log-likelihood
        max_batch_size: Maximum batch size for processing
        max_seq_len: Maximum sequence length
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
        group_scores = process_sequence_group_batch(group, model, tokenizer, aggregate, 
                                                  max_batch_size, max_seq_len, device)
        scores.extend(group_scores)
    
    return scores

def process_sequence_group_batch(sequences: List[str], model, tokenizer, aggregate: str,
                               max_batch_size: int, max_seq_len: int, device) -> List[float]:
    """Process a group of similar-length sequences in batches."""
    scores = []
    
    for i in range(0, len(sequences), max_batch_size):
        batch_seqs = sequences[i:i + max_batch_size]
        batch_scores = compute_batch_pseudo_ppl(batch_seqs, model, tokenizer, aggregate, 
                                              max_seq_len, device)
        scores.extend(batch_scores)
        
        # Clear GPU cache periodically
        if i % (max_batch_size * 4) == 0:
            cleanup_gpu_memory()
            if i > 0:  # Don't print on first iteration
                print_gpu_memory_info(f"after batch {i//max_batch_size}")
    
    return scores

def compute_batch_pseudo_ppl(sequences: List[str], model, tokenizer, aggregate: str,
                           max_seq_len: int, device) -> List[float]:
    """Compute pseudo-perplexity for a batch of sequences efficiently."""
    if not sequences:
        return []
    
    # Tokenize all sequences in the batch
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, 
                      max_length=max_seq_len)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    batch_size, seq_len = input_ids.shape
    scores = []
    
    # Process each sequence in the batch
    for seq_idx in range(batch_size):
        seq_input_ids = input_ids[seq_idx:seq_idx+1]  # Keep batch dimension
        seq_attention_mask = attention_mask[seq_idx:seq_idx+1]
        
        # Find actual sequence length (excluding padding)
        actual_len = seq_attention_mask.sum().item()
        
        if actual_len <= 2:  # Skip very short sequences
            scores.append(float("nan"))
            continue
        
        # Compute log-likelihood for each position using vectorized operations
        log_likelihoods = compute_position_likelihoods_vectorized(
            seq_input_ids, seq_attention_mask, model, tokenizer, actual_len
        )
        
        # Aggregate log likelihoods
        if log_likelihoods:
            if aggregate == "sum":
                score = sum(log_likelihoods)
            elif aggregate == "mean":
                score = sum(log_likelihoods) / len(log_likelihoods)
            else:
                raise ValueError(f"aggregate must be 'sum' or 'mean', got {aggregate}")
        else:
            score = float("nan")
            
        scores.append(score)
    
    return scores

def compute_position_likelihoods_vectorized(input_ids, attention_mask, model, tokenizer, 
                                          actual_len: int) -> List[float]:
    """Compute log-likelihoods for all positions using vectorized operations."""
    positions_to_mask = list(range(1, min(actual_len - 1, input_ids.size(1) - 1)))  # Skip [CLS] and [SEP]
    
    if not positions_to_mask:
        return []
    
    # Create batch of masked inputs for all positions at once
    num_positions = len(positions_to_mask)
    batch_masked_inputs = input_ids.repeat(num_positions, 1)  # [num_positions, seq_len]
    batch_attention_masks = attention_mask.repeat(num_positions, 1)
    
    # Mask each position
    for i, pos in enumerate(positions_to_mask):
        batch_masked_inputs[i, pos] = tokenizer.mask_token_id
    
    log_likelihoods = []
    
    # Process in smaller chunks to manage memory
    chunk_size = min(32, num_positions)  # Process up to 32 positions at once
    
    for chunk_start in range(0, num_positions, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_positions)
        chunk_masked_inputs = batch_masked_inputs[chunk_start:chunk_end]
        chunk_attention_masks = batch_attention_masks[chunk_start:chunk_end]
        
        with torch.no_grad():
            outputs = model(chunk_masked_inputs, attention_mask=chunk_attention_masks)
            logits = outputs.logits  # [chunk_size, seq_len, vocab_size]
            
            # Get log probabilities for masked positions
            for i, pos in enumerate(positions_to_mask[chunk_start:chunk_end]):
                log_probs = F.log_softmax(logits[i, pos], dim=-1)
                true_token = input_ids[0, pos]
                log_likelihood = log_probs[true_token].item()
                log_likelihoods.append(log_likelihood)
    
    return log_likelihoods

def compute_pseudo_ppl_hf(sequences: List[str], model, tokenizer, aggregate: str = "mean") -> List[float]:
    """
    Legacy function - redirects to optimized batch version.
    """
    return compute_pseudo_ppl_hf_batch(sequences, model, tokenizer, aggregate, 
                                     max_batch_size=16, max_seq_len=1024)

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
    return f"{clean_model_name}_eval_on_{tier}.txt"

def parse_model_size(model_name: str) -> str:
    """
    Parse model size from HuggingFace model name.
    
    Args:
        model_name (str): Model name like "given131/8M_T1" or "facebook/esm2_t6_8M_UR50D"
    Returns:
        str: Model size key ("8M", "35M", "150M")
    """
    if "8M" in model_name:
        return "8M"
    elif "35M" in model_name:
        return "35M"
    elif "150M" in model_name:
        return "150M"
    else:
        raise ValueError(f"Cannot determine model size from: {model_name}")

def load_esm2_model(ckpt_path: str, custom_weights_path: Optional[str] = None) -> tuple:
    """
    Load ESM2 model using HuggingFace transformers.
    
    Args:
        ckpt_path (str): HuggingFace model name (e.g., "given131/8M_T1" or "facebook/esm2_t6_8M_UR50D")
        custom_weights_path (str, optional): Path to custom weights file (.pt or .pth)
    Returns:
        model: HuggingFace EsmForMaskedLM model
        tokenizer: HuggingFace ESM2 tokenizer
    """
    # Determine the base Facebook model architecture
    if ckpt_path.startswith("given131/"):
        # Parse model size and get corresponding Facebook config
        model_size = parse_model_size(ckpt_path)
        facebook_model = FACEBOOK_CONFIG[model_size]
        print(f"Using custom model {ckpt_path} with architecture from {facebook_model}")
        
        # Initialize tokenizer and model from Facebook architecture
        tokenizer = AutoTokenizer.from_pretrained(facebook_model)
        model = EsmForMaskedLM.from_pretrained(facebook_model)
        
        # Load custom weights from HuggingFace model
        try:
            print(f"Loading custom weights from HuggingFace model: {ckpt_path}")
            from huggingface_hub import hf_hub_download
            
            # Download model.bin file
            model_bin_path = hf_hub_download(repo_id=ckpt_path, filename="model.bin")
            custom_state_dict = torch.load(model_bin_path, map_location='cpu')
            
            # Extract model_state_dict from the ordered_dict
            if 'model_state_dict' in custom_state_dict:
                model_weights = custom_state_dict['model_state_dict']
                print("Found 'model_state_dict' in checkpoint")
            else:
                print("Warning: 'model_state_dict' not found, using full state dict")
                model_weights = custom_state_dict
            
            # Load the state dict (strict=False to allow partial loading)
            missing_keys, unexpected_keys = model.load_state_dict(model_weights, strict=False)
            
            if missing_keys:
                print(f"Missing keys when loading custom weights: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"Unexpected keys when loading custom weights: {unexpected_keys[:5]}...")  # Show first 5
                
            print("Custom weights loaded successfully from HuggingFace")
        except (ImportError, OSError, RuntimeError) as e:
            print(f"Warning: Failed to load custom weights from HuggingFace: {e}")
            print("Continuing with pretrained weights...")
    
    else:
        # Standard Facebook model
        print(f"Using standard Facebook model: {ckpt_path}")
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        model = EsmForMaskedLM.from_pretrained(ckpt_path)
        
        # Load additional custom weights if provided
        if custom_weights_path:
            print(f"Loading additional custom weights from: {custom_weights_path}")
            try:
                custom_state_dict = torch.load(custom_weights_path, map_location='cpu')
                
                # Handle different state dict formats
                if 'model_state_dict' in custom_state_dict:
                    custom_state_dict = custom_state_dict['model_state_dict']
                elif 'model' in custom_state_dict:
                    custom_state_dict = custom_state_dict['model']
                elif 'state_dict' in custom_state_dict:
                    custom_state_dict = custom_state_dict['state_dict']
                
                # Load the state dict (strict=False to allow partial loading)
                missing_keys, unexpected_keys = model.load_state_dict(custom_state_dict, strict=False)
                
                if missing_keys:
                    print(f"Missing keys when loading custom weights: {missing_keys[:5]}...")  # Show first 5
                if unexpected_keys:
                    print(f"Unexpected keys when loading custom weights: {unexpected_keys[:5]}...")  # Show first 5
                    
                print("Additional custom weights loaded successfully")
            except (ImportError, OSError, RuntimeError) as e:
                print(f"Warning: Failed to load additional custom weights: {e}")
                print("Continuing with model weights...")
    
    model.eval()
    return model, tokenizer

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
                  batch_size: int = 256, aggregate: str = "mean", custom_weights_path: Optional[str] = None,
                  max_seq_len: int = 1024, use_fp16: bool = True) -> Dict[str, float]:
    """
    Evaluate perplexity of sequences in a FASTA file using ESM2 model.

    Args:
        fasta_path (str): Path to the input FASTA file.
        ckpt_path (str): HuggingFace model name (e.g., "facebook/esm2_t6_8M_UR50D")
        batch_size (int): Batch size for processing sequences.
        aggregate (str): "sum" for total log-likelihood, "mean" for average log-likelihood
    Returns:
        dict: A dictionary mapping sequence IDs to their perplexity scores.
    """
    start_time = time.time()
    
    # Load ESM2 model using HuggingFace
    print("Loading model...")
    model_load_start = time.time()
    model, tokenizer = load_esm2_model(ckpt_path=ckpt_path, custom_weights_path=custom_weights_path)
    
    # Move model to GPU and optimize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Enable mixed precision and optimizations
    if torch.cuda.is_available():
        if use_fp16:
            model = model.half()  # Use FP16 for better memory efficiency
            print(f"Model loaded on {device} with FP16 precision")
        else:
            print(f"Model loaded on {device} with FP32 precision")
        torch.backends.cudnn.benchmark = True
        print_gpu_memory_info("after model loading")
    
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
    print(f"Using optimized GPU batch processing...")
    
    batch_start_time = time.time()
    
    # Monitor GPU memory before processing
    cleanup_gpu_memory()
    print_gpu_memory_info("before processing")
    
    # Compute all scores using optimized batch processing
    batch_scores = compute_pseudo_ppl_hf_batch(
        sequences, model, tokenizer, aggregate=aggregate, 
        max_batch_size=batch_size, max_seq_len=max_seq_len
    )
    
    batch_time = time.time() - batch_start_time
    processing_times.append(batch_time)
    
    # Convert scores to perplexities and store results
    batch_perplexities = []
    for seq_id, score in zip(seq_ids, batch_scores):
        # Convert log-likelihood to perplexity: exp(-log_likelihood)
        if not torch.isnan(torch.tensor(score)):
            perplexity = torch.exp(-torch.tensor(score)).item() 
        else:
            perplexity = float("nan")
        results[seq_id] = perplexity
        
        if not np.isnan(perplexity):
            batch_perplexities.append(perplexity)
            all_perplexities.append(perplexity)
    
    # Monitor GPU memory after processing
    allocated, reserved, peak = print_gpu_memory_info("after processing")
    
    # Log batch metrics
    if batch_perplexities:
        wandb.log({
            "batch_metrics/total_batch_time": batch_time,
            "batch_metrics/sequences_per_second": len(sequences) / batch_time,
            "batch_metrics/mean_perplexity": np.mean(batch_perplexities),
            "batch_metrics/median_perplexity": np.median(batch_perplexities),
            "batch_metrics/gpu_memory_peak_gb": peak,
            "batch_metrics/gpu_memory_allocated_gb": allocated,
            "batch_metrics/gpu_memory_reserved_gb": reserved,
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
        help="HuggingFace model name. Examples: 'facebook/esm2_t6_8M_UR50D', 'given131/8M_T1', 'given131/35M_H', 'given131/150M_F'.",
    )
    parser.add_argument(
        "--custom-weights",
        type=str,
        default=None,
        help="Path to custom weights file (.pt or .pth) to load into the model.",
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
        "--use-fp16",
        action="store_true",
        default=False,
        help="Use FP16 precision for better memory efficiency.",
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
    if args.output is None:
        output_path = generate_output_filename(args.ckpt_path, args.tier)
    else:
        output_path = args.output

    # Initialize wandb
    wandb.init(
        project="esm2-gen-eval",
        config={
            "model_ckpt": args.ckpt_path,
            "eval_tier": args.tier,
            "custom_weights": args.custom_weights,
            "batch_size": args.batch_size,
            "max_seq_len": args.max_seq_len,
            "use_fp16": args.use_fp16,
            "aggregation": args.aggregate,
            "fasta_path": fasta_path,
            "output_path": output_path,
        },
        tags=[f"tier_{args.tier}", 
              parse_model_size(args.ckpt_path) if any(size in args.ckpt_path for size in ["8M", "35M", "150M"]) else "unknown_size"]
    )

    print(f"Evaluating perplexity using ESM2 model: {args.ckpt_path}")
    print(f"Tier: {args.tier}")
    print(f"Input FASTA: {fasta_path}")
    print(f"Output file: {output_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max sequence length: {args.max_seq_len}")
    print(f"Use FP16: {args.use_fp16}")
    print(f"Aggregation method: {args.aggregate}")

    results = eval_ppl_esm2(
        fasta_path=fasta_path,
        ckpt_path=args.ckpt_path,
        batch_size=args.batch_size,
        aggregate=args.aggregate,
        custom_weights_path=args.custom_weights,
        max_seq_len=args.max_seq_len,
        use_fp16=args.use_fp16
    )
    
    # Save results with config details
    import datetime
    
    # Create config info
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config_info = {
        "timestamp": timestamp,
        "model": args.ckpt_path,
        "custom_weights": args.custom_weights or "None",
        "tier": args.tier,
        "fasta_file": fasta_path,
        "batch_size": args.batch_size,
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

