# goal get esm2-ppl for fasta sequences 

import argparse
import torch
from Bio import SeqIO
from typing import List, Dict, Optional
import time
import numpy as np

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

def compute_pseudo_ppl_hf(sequences: List[str], model, tokenizer, aggregate: str = "mean") -> List[float]:
    """
    Compute pseudo-perplexity for sequences using HuggingFace ESM2 model.
    
    Args:
        sequences: List of protein sequences
        model: HuggingFace EsmForMaskedLM model
        tokenizer: HuggingFace ESM2 tokenizer
        aggregate: "sum" for total log-likelihood, "mean" for average log-likelihood
    Returns:
        List of perplexity scores
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    scores = []
    
    for seq in tqdm(sequences, desc="Computing perplexity", leave=False):
        # Tokenize sequence
        inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        seq_len = input_ids.size(1)
        if seq_len <= 2:  # Skip very short sequences
            scores.append(float("nan"))
            continue
            
        # Mask each position (except special tokens) and compute log-likelihood
        log_likelihoods = []
        
        for pos in range(1, seq_len - 1):  # Skip [CLS] and [SEP] tokens
            # Create masked input
            masked_input = input_ids.clone()
            masked_input[0, pos] = tokenizer.mask_token_id
            
            # Get model prediction
            with torch.no_grad():
                outputs = model(masked_input, attention_mask=attention_mask)
                logits = outputs.logits  # [1, seq_len, vocab_size]
                
                # Get log probabilities for the masked position
                log_probs = F.log_softmax(logits[0, pos], dim=-1)
                true_token = input_ids[0, pos]
                log_likelihood = log_probs[true_token].item()
                log_likelihoods.append(log_likelihood)
        
        # Aggregate log likelihoods
        if aggregate == "sum":
            score = sum(log_likelihoods)
        elif aggregate == "mean":
            score = sum(log_likelihoods) / len(log_likelihoods) if log_likelihoods else float("nan")
        else:
            raise ValueError(f"aggregate must be 'sum' or 'mean', got {aggregate}")
            
        scores.append(score)
    
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
                  batch_size: int = 32, aggregate: str = "mean", custom_weights_path: Optional[str] = None) -> Dict[str, float]:
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

    # Process sequences in batches
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing batches", total=num_batches):
        batch_start_time = time.time()
        batch_seqs = sequences[i : i + batch_size]
        batch_ids = seq_ids[i : i + batch_size]

        # Compute pseudo-perplexity scores (log-likelihoods)
        batch_scores = compute_pseudo_ppl_hf(batch_seqs, model, tokenizer, aggregate=aggregate)
        
        batch_perplexities = []
        for seq_id, score in zip(batch_ids, batch_scores):
            # Convert log-likelihood to perplexity: exp(-log_likelihood)
            if not torch.isnan(torch.tensor(score)):
                perplexity = torch.exp(-torch.tensor(score)).item() 
            else:
                perplexity = float("nan")
            results[seq_id] = perplexity
            
            if not np.isnan(perplexity):
                batch_perplexities.append(perplexity)
                all_perplexities.append(perplexity)
        
        batch_time = time.time() - batch_start_time
        processing_times.append(batch_time)
        
        # Log batch metrics
        if batch_perplexities:
            wandb.log({
                "batch_metrics/batch_idx": i // batch_size,
                "batch_metrics/batch_size": len(batch_seqs),
                "batch_metrics/batch_time": batch_time,
                "batch_metrics/sequences_per_second": len(batch_seqs) / batch_time,
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
        help="Batch size for processing sequences.",
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
    print(f"Aggregation method: {args.aggregate}")

    results = eval_ppl_esm2(
        fasta_path=fasta_path,
        ckpt_path=args.ckpt_path,
        batch_size=args.batch_size,
        aggregate=args.aggregate,
        custom_weights_path=args.custom_weights
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

