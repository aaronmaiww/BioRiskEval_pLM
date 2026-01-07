#!/usr/bin/env python3
"""
Sample each tier so that sequences only appear in the most specific tier.

Reads the training-tier FASTA files under `proteindb/data/tier_train` and
produces non-overlapping sampled FASTA files under
`proteindb/data/tier_sequences_random_nonoverlapping`. Every sequence is
assigned to the highest-numbered tier that contains it, and earlier tiers
cannot reuse that sequence.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of sequences to retain per tier (all are kept if there are fewer).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length to include in the samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("proteindb/data/tier_sequences_random_nonoverlapping"),
        help="Directory to write the non-overlapping random FASTA files.",
    )
    return parser.parse_args()


def read_fasta_entries(path: Path) -> list[Tuple[str, str]]:
    entries: list[Tuple[str, str]] = []
    current_header: str | None = None
    current_seq: list[str] = []
    with open(path, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                if current_header is not None:
                    entries.append((current_header, "".join(current_seq)))
                current_header = stripped
                current_seq = []
            else:
                current_seq.append(stripped)
    if current_header is not None:
        entries.append((current_header, "".join(current_seq)))
    return entries


def write_fasta(entries: Iterable[Tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for header, sequence in entries:
            fh.write(f"{header}\n")
            for i in range(0, len(sequence), 60):
                fh.write(f"{sequence[i:i+60]}\n")


def locate_tier_train_paths(root: Path) -> dict[int, Path]:
    train_dir = root / "tier_train"
    result: dict[int, Path] = {}
    for tier in range(1, 7):
        for suffix in ("_sequences_train.fasta", "_sequences.fasta"):
            candidate = train_dir / f"tier{tier}{suffix}"
            if candidate.exists():
                result[tier] = candidate
                break
        else:
            raise FileNotFoundError(f"Missing training file for tier {tier}")
    return result


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    root = Path("proteindb/data")
    train_paths = locate_tier_train_paths(root)
    tier_entries = {tier: read_fasta_entries(path) for tier, path in sorted(train_paths.items())}

    tier_sequences = {tier: {sequence for _, sequence in entries} for tier, entries in tier_entries.items()}

    non_overlapping_entries: dict[int, list[Tuple[str, str]]] = {}
    assigned_sequences: set[str] = set()

    for tier in reversed(sorted(tier_sequences)):
        available_sequences = tier_sequences[tier] - assigned_sequences
        assigned_sequences |= tier_sequences[tier]
        entries = [
            (header, sequence)
            for header, sequence in tier_entries[tier]
            if sequence in available_sequences
        ]
        non_overlapping_entries[tier] = entries

    for tier in sorted(non_overlapping_entries):
        entries = non_overlapping_entries[tier]
        filtered = [(hdr, seq) for hdr, seq in entries if len(seq) <= args.max_length]
        if not filtered:
            print(f"⚠️  No sequences remained for tier{tier} after filtering length <= {args.max_length}.")
            continue

        if len(filtered) <= args.sample_size:
            sampled = filtered
            print(f"tier{tier}: taking all {len(filtered)} non-overlapping sequences.")
        else:
            sampled = random.sample(filtered, args.sample_size)
            print(f"tier{tier}: sampled {args.sample_size} from {len(filtered)} candidates.")

        output_path = args.output_dir / f"tier{tier}.fasta"
        write_fasta(sampled, output_path)
        print(f"  wrote {len(sampled)} sequences to {output_path}")


if __name__ == "__main__":
    main()

