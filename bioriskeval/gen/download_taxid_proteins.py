#!/usr/bin/env python3
"""
Download protein sequences for viral taxids from the viral exclusion tier files.
"""

import argparse
import os
import subprocess
import time
from typing import List
import re
import glob


def read_taxids_from_file(file_path: str) -> List[str]:
    """
    Read taxids from a tier file, skipping comment lines.
    
    Parameters:
    - file_path: Path to the taxid file
    
    Returns:
    - List of taxid strings
    """
    taxids = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                taxids.append(line)
    
    return taxids


def download_proteins_for_single_taxid(taxid: str, output_dir: str) -> bool:
    """
    Download protein sequences for a single taxid using NCBI E-utilities.
    
    Parameters:
    - taxid: Single taxid string
    - output_dir: Directory to store the downloaded data
    
    Returns:
    - True if successful, False otherwise
    """
    
    # Create search query for this taxid
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=protein&term=txid{taxid}[Organism]&retmax=1000&retmode=xml"
    
    try:
        # Search for protein IDs
        search_result = subprocess.run(
            f"wget -q -O - '{search_url}'", 
            shell=True, capture_output=True, text=True
        )
        
        if search_result.returncode != 0:
            print(f"✗ Failed to search for taxid {taxid}")
            return False
        
        # Extract protein IDs from XML (simple parsing)
        id_matches = re.findall(r'<Id>(\d+)</Id>', search_result.stdout)
        
        if not id_matches:
            print(f"⚠️ No proteins found for taxid {taxid}")
            return True  # Not an error, just no results
        
        print(f"  Found {len(id_matches)} proteins for taxid {taxid}")
        
        # Download sequences for these IDs (limit to first 100 to avoid huge downloads)
        id_string = ",".join(id_matches)
        fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=protein&id={id_string}&rettype=fasta&retmode=text"
        
        # Download FASTA sequences
        output_file = os.path.join(output_dir, f"taxid_{taxid}_proteins.faa")
        fetch_result = subprocess.run(
            f"wget -q -O '{output_file}' '{fetch_url}'", 
            shell=True, capture_output=True, text=True
        )
        
        if fetch_result.returncode == 0:
            print(f"  ✓ Downloaded {len(id_matches)} sequences to {output_file}")
            return True
        else:
            print(f"  ✗ Failed to download sequences for taxid {taxid}")
            return False
            
    except Exception as e:
        print(f"✗ Exception for taxid {taxid}: {str(e)}")
        return False


def download_proteins_by_taxid_batch(taxid_list: List[str], batch_size: int = 5, 
                                   output_dir: str = "ncbi_protein_downloads") -> None:
    """
    Download protein sequences from NCBI using taxids.
    
    Parameters:
    - taxid_list: List of taxid strings
    - batch_size: Number of taxids to process before a pause (for rate limiting)
    - output_dir: Directory to store the downloaded data
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nProcessing {len(taxid_list)} taxids...")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    successful_downloads = 0
    failed_downloads = 0
    
    for i, taxid in enumerate(taxid_list, 1):
        print(f"\n[{i}/{len(taxid_list)}] Processing taxid {taxid}")
        
        if download_proteins_for_single_taxid(taxid, output_dir):
            successful_downloads += 1
        else:
            failed_downloads += 1
        
        # Rate limiting: pause every batch_size downloads
        if i % batch_size == 0 and i < len(taxid_list):
            print(f"  Pausing for rate limiting...")
            time.sleep(3)
    
    print(f"\n" + "="*50)
    print(f"Download process completed!")
    print(f"Successful: {successful_downloads}")
    print(f"Failed: {failed_downloads}")
    print(f"Results in: {os.path.abspath(output_dir)}")
    print("="*50)


def merge_fasta_files(input_dir: str, output_file: str) -> None:
    """
    Merge all FASTA files in a directory into a single FASTA file.
    
    Parameters:
    - input_dir: Directory containing FASTA files
    - output_file: Path to the merged output FASTA file
    """
    
    fasta_files = glob.glob(os.path.join(input_dir, "*.faa"))
    
    with open(output_file, 'w') as outfile:
        for fasta_file in fasta_files:
            with open(fasta_file, 'r') as infile:
                outfile.write(infile.read())
    
    print(f"Merged {len(fasta_files)} FASTA files into {output_file}")



def main():
    parser = argparse.ArgumentParser(
        description="Download protein sequences for viral taxids from tier files"
    )
    
    parser.add_argument(
        "--tier-file", 
        required=True,
        help="Path to the taxid tier file"
    )
    
    parser.add_argument(
        "--output-dir",
        default="ncbi_protein_downloads", 
        help="Output directory for downloads (default: ncbi_protein_downloads)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of taxids per batch (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Validate tier file exists
    if not os.path.exists(args.tier_file):
        print(f"Error: Tier file not found: {args.tier_file}")
        return 1
    
    print(f"Reading taxids from {args.tier_file}...")
    taxids = read_taxids_from_file(args.tier_file)
    print(f"Found {len(taxids)} taxids")
    
    # Show first few taxids as example
    print(f"First 10 taxids: {taxids[:10]}")
    
    # Download protein sequences
    print("Starting download process...")
    download_proteins_by_taxid_batch(taxids, args.batch_size, args.output_dir)

    # Merge all downloaded FASTA files into one with informative name
    # Extract tier name from file path (e.g., "tier6_betacoronavirus_pandemicum" from the filename)
    tier_filename = os.path.basename(args.tier_file)
    tier_name = tier_filename.replace("_taxids.txt", "").replace(".txt", "")
    merged_output_file = os.path.join(args.output_dir, f"merged_{tier_name}_proteins.faa")
    merge_fasta_files(args.output_dir, merged_output_file)
    
    return 0


if __name__ == "__main__":
    exit(main())