# goal get esm2-ppl for fasta sequences 

import argparse
import torch
import os
from pathlib import Path
from Bio import SeqIO
from typing import List, Dict

from transformers import AutoTokenizer, EsmForMaskedLM
import torch.nn.functional as F

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
    
    for seq in sequences:
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

def load_esm2_model(ckpt_path: str) -> tuple:
    """
    Load ESM2 model using HuggingFace transformers.
    
    Args:
        ckpt_path (str): HuggingFace model name (e.g., "facebook/esm2_t6_8M_UR50D")
    Returns:
        model: HuggingFace EsmForMaskedLM model
        tokenizer: HuggingFace ESM2 tokenizer
    """
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = EsmForMaskedLM.from_pretrained(ckpt_path)
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
                  batch_size: int = 32, aggregate: str = "mean") -> Dict[str, float]:
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
    # Load ESM2 model using HuggingFace
    model, tokenizer = load_esm2_model(ckpt_path=ckpt_path)
    
    # Load sequences from FASTA file
    sequences, seq_ids = load_sequences_from_fasta(fasta_path)

    results = {}

    # Process sequences in batches
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        batch_ids = seq_ids[i : i + batch_size]

        # Compute pseudo-perplexity scores (log-likelihoods)
        batch_scores = compute_pseudo_ppl_hf(batch_seqs, model, tokenizer, aggregate=aggregate)

        for seq_id, score in zip(batch_ids, batch_scores):
            # Convert log-likelihood to perplexity: exp(-log_likelihood)
            if not torch.isnan(torch.tensor(score)):
                perplexity = torch.exp(-torch.tensor(score)).item() 
            else:
                perplexity = float("nan")
            results[seq_id] = perplexity
            print(f"Sequence ID: {seq_id}, Score: {score:.4f}, Perplexity: {perplexity:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate perplexity of sequences using ESM2 model with BioNeMo."
    )
    parser.add_argument(
        "--fasta",
        type=str,
        required=True,
        help="Path to the input FASTA file containing sequences.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="HuggingFace model name (e.g., 'facebook/esm2_t6_8M_UR50D', 'facebook/esm2_t33_650M_UR50D').",
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
        default=8,
        help="Batch size for processing sequences.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output file to save perplexity results.",
    )
 
    args = parser.parse_args()

    print(f"Evaluating perplexity using ESM2 model: {args.ckpt_path}")
    print(f"Input FASTA: {args.fasta}")
    print(f"Batch size: {args.batch_size}")
    print(f"Aggregation method: {args.aggregate}")

    results = eval_ppl_esm2(
        fasta_path=args.fasta,
        ckpt_path=args.ckpt_path,
        batch_size=args.batch_size,
        aggregate=args.aggregate
    )

    # Save results to output file
    with open(args.output, "w") as f:
        f.write("sequence_id\tperplexity\n")  # Header
        for seq_id, ppl in results.items():
            f.write(f"{seq_id}\t{ppl}\n")
    
    print(f"Perplexity results saved to {args.output}")
    print(f"Processed {len(results)} sequences.")
    

if __name__ == "__main__":
    main()

