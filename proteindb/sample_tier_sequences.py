#!/usr/bin/env python3
"""
Random sampling script for tierX_sequences.fasta files.
Creates tierX_sequences_random.fasta files with 1000 random samples from each tier.
"""

import random
from pathlib import Path


def read_fasta(filepath):
    """
    Read FASTA file and return list of (header, sequence) tuples.
    """
    entries = []
    current_header = None
    current_seq = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous entry if exists
                if current_header is not None:
                    entries.append((current_header, ''.join(current_seq)))
                # Start new entry
                current_header = line
                current_seq = []
            else:
                current_seq.append(line)
        
        # Don't forget the last entry
        if current_header is not None:
            entries.append((current_header, ''.join(current_seq)))
    
    return entries


def write_fasta(entries, filepath):
    """
    Write FASTA entries to file.
    """
    with open(filepath, 'w') as f:
        for header, sequence in entries:
            f.write(f"{header}\n")
            # Write sequence in lines of 60 characters (standard FASTA format)
            for i in range(0, len(sequence), 60):
                f.write(f"{sequence[i:i+60]}\n")


def sample_tier_sequences(tier_num, sample_size=1000, max_length=1024):
    """
    Sample random sequences from a tier file.
    Only samples sequences with length <= max_length.
    """
    # Try both naming patterns
    input_file = Path(f"data/tier_train/tier{tier_num}_sequences.fasta")
    if not input_file.exists():
        input_file = Path(f"data/tier_train/tier{tier_num}_sequences_train.fasta")
    
    output_file = Path(f"data/tier_sequences_random/tier{tier_num}_sequences_random.fasta")
    
    if not input_file.exists():
        print(f"⚠️  {input_file} not found, skipping...")
        return
    
    print(f"Reading {input_file}...")
    entries = read_fasta(input_file)
    total_entries = len(entries)
    print(f"  Found {total_entries} sequences")
    
    # Filter sequences by length
    filtered_entries = [(header, seq) for header, seq in entries if len(seq) <= max_length]
    filtered_count = len(filtered_entries)
    print(f"  Filtered to {filtered_count} sequences with length <= {max_length} ({filtered_count/total_entries*100:.2f}%)")
    
    if filtered_count == 0:
        print(f"⚠️  No sequences found with length <= {max_length}, skipping...\n")
        return
    
    # Sample (or take all if less than sample_size)
    if filtered_count <= sample_size:
        print(f"  Taking all {filtered_count} sequences (less than {sample_size})")
        sampled_entries = filtered_entries
    else:
        print(f"  Randomly sampling {sample_size} sequences...")
        sampled_entries = random.sample(filtered_entries, sample_size)
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Writing {output_file}...")
    write_fasta(sampled_entries, output_file)
    print(f"✓ Created {output_file} with {len(sampled_entries)} sequences\n")


def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    print("=" * 60)
    print("Random Sampling FASTA Sequences")
    print("=" * 60)
    print()
    
    # Process each tier (1-6)
    for tier_num in range(1, 7):
        sample_tier_sequences(tier_num, sample_size=1000)
    
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

