#!/usr/bin/env python3
"""
Create linear probe dataset from virulence data with HuggingFace ESM2 representations.

Usage:
    python create_virulence_probe_dataset_esm2.py --model_name facebook/esm2_t30_150M_UR50D \
                                                   --output_dir ./probe_datasets \
                                                   --layer_numbers 12 24 30 \
                                                   --seed 42
"""

import argparse
import logging
import os
import h5py
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
from typing import List, Dict, Any, cast

# HuggingFace imports for ESM2
from transformers import AutoTokenizer, EsmForMaskedLM

# BioPython for DNA translation
from Bio.Seq import Seq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


def translate_dna_to_protein(dna_sequence: str) -> str:
    """Translate DNA sequence to protein using BioPython."""
    seq = Seq(dna_sequence)
    protein = seq.translate(to_stop=True)  # Stop at first stop codon
    return str(protein)


def load_esm2_model_hf(model_name: str):
    """Load ESM2 model using HuggingFace transformers."""
    logger.info(f"Loading HuggingFace ESM2 model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"Model loaded on device: {device}")
    
    return model, tokenizer, device
    

def extract_representations_batch(sequences: List[str], model, tokenizer, device: str, layer_numbers: List[int]) -> Dict[int, np.ndarray]:
    """
    Extract representations from specified layers for a batch of sequences.
    
    Returns:
        Dict mapping layer numbers to representations of shape (batch_size, hidden_dim)
    """
    # Tokenize sequences
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    representations = {}
    
    with torch.no_grad():
        # Get model outputs with hidden states
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of (batch_size, seq_len, hidden_dim)
        
        # Extract representations from specified layers
        for layer_num in layer_numbers:
            if layer_num < len(hidden_states):
                # Use the representation from the last non-padding token
                layer_hidden = hidden_states[layer_num]  # (batch_size, seq_len, hidden_dim)
                
                # Find the last non-padding token for each sequence
                attention_mask = inputs['attention_mask']
                seq_lengths = attention_mask.sum(dim=1) - 1  # -1 to get last token index
                
                # Extract last token representation for each sequence
                batch_size = layer_hidden.size(0)
                last_token_reprs = []
                for i in range(batch_size):
                    last_idx = seq_lengths[i].item()
                    last_token_reprs.append(layer_hidden[i, last_idx, :])
                
                layer_repr = torch.stack(last_token_reprs)  # (batch_size, hidden_dim)
                representations[layer_num] = layer_repr.cpu().numpy()
            else:
                logger.warning(f"Layer {layer_num} not available in model (only {len(hidden_states)} layers)")
    
    return representations


def create_probe_dataset_hf(args):
    """Create probe dataset using HuggingFace ESM2."""
    
    # Load data
    logger.info(f"Loading data from: {args.dataset_path}")
    df = pd.read_csv(args.dataset_path)
    logger.info(f"Found {len(df)} sequences")
    
    # Sample if requested
    if args.n_samples and args.n_samples < len(df):
        logger.info(f"Sampling {args.n_samples} sequences")
        df = df.sample(n=args.n_samples, random_state=args.seed).reset_index(drop=True)
    
    # Prepare sequences and labels
    dna_sequences = df['nucleotide_sequence'].tolist()
    logger.info("Translating DNA sequences to proteins...")
    sequences = [translate_dna_to_protein(dna) for dna in tqdm(dna_sequences, desc="Translation")]
    
    if args.dataset_type == "binary":
        labels = df['Two-class virulence level'].replace({'Avirulent': 0, 'Virulent': 1}).tolist()
    else:
        labels = df['LD50'].tolist()
    
    logger.info(f"Prepared {len(sequences)} sequences with {args.dataset_type} labels")
    
    # Create train/test split
    train_sequences, test_sequences, train_labels, test_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=args.seed
    )
    logger.info(f"Split: {len(train_sequences)} train, {len(test_sequences)} test")
    
    # Load ESM2 model
    model, tokenizer, device = load_esm2_model_hf(args.model_name)
    logger.info(f"Model loaded successfully")
    
    # Process each split
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {os.path.abspath(args.output_dir)}")
    
    for split_name, (split_seqs, split_labels) in [('train', (train_sequences, train_labels)), 
                                                   ('test', (test_sequences, test_labels))]:
        
        # Process single layer
        logger.info(f"Processing {split_name} split, layer {args.layer_number}...")
        
        # Extract representations in batches
        all_reprs = []
        for i in tqdm(range(0, len(split_seqs), args.batch_size), desc=f"Layer {args.layer_number} {split_name}"):
            batch_seqs = split_seqs[i:i + args.batch_size]
            batch_reprs = extract_representations_batch(batch_seqs, model, tokenizer, device, [args.layer_number])
            if args.layer_number in batch_reprs:
                all_reprs.append(batch_reprs[args.layer_number])
        
        # Save to HDF5 if we got representations
        if all_reprs:
            final_reprs = np.concatenate(all_reprs, axis=0)
            
            # Create output filename
            model_name_safe = args.model_name.replace("/", "_")
            output_file = f"virulence_probe_dataset_{model_name_safe}_layer_{args.layer_number}_{split_name}.h5"
            output_path = os.path.join(args.output_dir, output_file)
            
            # Save to HDF5
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('sequences', data=np.array(split_seqs, dtype='S'), compression='gzip')
                f.create_dataset('representations', data=final_reprs, compression='gzip')
                f.create_dataset('labels', data=np.array(split_labels))
                f.attrs['model_name'] = args.model_name
                f.attrs['layer_number'] = args.layer_number
                f.attrs['split'] = split_name
            
            logger.info(f"Saved {output_path}: {final_reprs.shape}")
    
    logger.info("Dataset creation completed!")

    
    
   
def parse_layer_info(layer_name):
    """Parse layer name to extract layer number and type."""
    parts = layer_name.split('.')
    layer_num = None
    layer_type = None
    
    for i, part in enumerate(parts):
        if part.isdigit():
            layer_num = part
            # Get the next part as the layer type if it exists
            if i + 1 < len(parts):
                layer_type = parts[i + 1]
            break
    
    if layer_num and layer_type:
        return f"_layer_{layer_num}_{layer_type}"
    elif layer_num:
        return f"_layer_{layer_num}"
    else:
        # Fallback: use the last part of the layer name
        return f"_{parts[-1]}" if parts else ""
    
        
def initialize_hdf5_file(output_path, metadata, total_sequences, labels_dtype: str, hidden_dim):
    """Initialize HDF5 file with resizable datasets.

    labels_dtype: 'float32' for continuous targets, 'int64' for binary.
    """
    print(f"Initializing HDF5 file: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Create resizable datasets
        f.create_dataset('sequences', (0,), maxshape=(None,), dtype=h5py.string_dtype())
        f.create_dataset('representations', (0, hidden_dim), maxshape=(None, hidden_dim), dtype='float32', compression='gzip')
        # Labels dtype depends on dataset type
        f.create_dataset('labels', (0,), maxshape=(None,), dtype=labels_dtype)
        
        # Store metadata as attributes
        for key, value in metadata.items():
            f.attrs[key] = value
    
    print(f"HDF5 file initialized")


def append_to_hdf5(output_path, sequences, representations, labels, hidden_dim):
    """Append data to existing HDF5 file."""
    import h5py
    with h5py.File(output_path, 'a') as f:
        # Access datasets with explicit typing to satisfy linters
        seq_ds = cast(h5py.Dataset, f['sequences'])
        repr_ds = cast(h5py.Dataset, f['representations'])
        labels_ds = cast(h5py.Dataset, f['labels'])

        # Current size
        current_size = int(seq_ds.shape[0])

        # Resize datasets based on intended append length
        target_size = current_size + len(sequences)
        seq_ds.resize((target_size,))
        repr_ds.resize((target_size, hidden_dim))
        labels_ds.resize((target_size,))

        # Only write the number of rows that we actually have representations for
        actual_count = int(representations.shape[0])
        new_size = current_size + actual_count

        seq_ds[current_size:new_size] = sequences[:actual_count]
        repr_ds[current_size:new_size] = representations
        # Write labels with dtype matching the dataset
        if np.issubdtype(labels_ds.dtype, np.floating):
            labels_np = np.asarray(labels[:actual_count], dtype='float32')
        else:
            labels_np = np.asarray(labels[:actual_count], dtype='int64')
        labels_ds[current_size:new_size] = labels_np

    print(f"Appended {actual_count} sequences to {output_path}")



def main():
    parser = argparse.ArgumentParser(description="Create virulence probe dataset with BioNeMo representations")
    
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D",
                       help="HuggingFace ESM2 model name")
    parser.add_argument("--layer_number", type=int, default=4,
                       help="Layer number to extract representations from")
    parser.add_argument("--dataset_path", type=str, default="data/influenza_virulence_ld50_cleaned_BALB_C.csv",
                       help="Path to reference dataset")
    parser.add_argument("--dataset_type", type=str, default="continuos", choices=["binary", "continuous"],
                       help="Type of dataset")
    parser.add_argument("--output_dir", type=str, default="probe_datasets",
                       help="Output directory for datasets")
    parser.add_argument("--n_samples", type=int, default=625,
                       help="Total samples per dataset (will be split into train/test)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for representation extraction")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    print("Starting virulence probe dataset creation...")
    print(f"Arguments: {vars(args)}")
    
    try:
        create_probe_dataset_hf(args)
        print("Dataset creation completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Error during dataset creation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 