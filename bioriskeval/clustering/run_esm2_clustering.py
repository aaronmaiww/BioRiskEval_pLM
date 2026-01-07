"""Cluster ESM2 checkpoint embeddings across tiers and checkpoints."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from Bio import SeqIO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from bioriskeval.common import (
    load_esm2_model,
    parse_model_tier,
    setup_model_optimizations,
)

try:
    import hdbscan  # noqa: F401
except ImportError as exc:
    raise ImportError("Please install hdbscan to run the clustering script.") from exc

MODEL_PREFIX = "given131"
MODEL_GROUPS: Dict[str, List[str]] = {
    "8M": ["8M_T1", "8M_T2", "8M_T5", "8M_T6", "8M_H", "8M_F"],
    "35M": ["35M_T1", "35M_T2", "35M_T5", "35M_T6", "35M_H", "35M_F"],
    "150M": ["150M_T1", "150M_T2", "150M_T5", "150M_T6", "150M_H", "150M_F"],
}


def get_full_model_name(model_short: str) -> str:
    """Convert a short model key to the HuggingFace identifier."""
    return f"{MODEL_PREFIX}/{model_short}"


def load_tier_sequences(
    tier: int,
    fasta_root: str,
    max_sequences: int,
) -> Tuple[List[str], List[str]]:
    """Load up to `max_sequences` records from a tier FASTA file."""
    fasta_path = os.path.join(fasta_root, f"tier{tier}.fasta")
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"Missing tier FASTA: {fasta_path}")

    sequences = []
    seq_ids = []
    for idx, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
        if max_sequences and idx >= max_sequences:
            break
        sequences.append(str(record.seq))
        seq_ids.append(record.id)

    return sequences, seq_ids


def mean_pool_last_hidden(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean pool the last layer representations ignoring special tokens."""
    pooled_mask = attention_mask.clone()
    seq_len = pooled_mask.shape[1]
    if seq_len >= 1:
        pooled_mask[:, 0] = 0
        pooled_mask[:, -1] = 0

    expanded_mask = pooled_mask.unsqueeze(-1)
    lengths = pooled_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
    return (hidden_states * expanded_mask).sum(dim=1) / lengths


def embed_sequences(
    model,
    tokenizer,
    sequences: Sequence[str],
    device: torch.device,
    batch_size: int,
    max_seq_len: int,
) -> np.ndarray:
    """Convert sequences into mean-pooled embeddings from the last encoder."""
    if not sequences:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)

    embeddings: List[torch.Tensor] = []

    for batch_start in tqdm(
        range(0, len(sequences), batch_size),
        desc=f"tokenizing (len={len(sequences)})",
        leave=False,
    ):
        batch = sequences[batch_start : batch_start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        autocast_ctx = (
            torch.cuda.amp.autocast(dtype=torch.bfloat16)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )

        with torch.inference_mode(), autocast_ctx:
            outputs = model(
                **encoded, output_hidden_states=True, return_dict=True
            )

        hidden_states = outputs.hidden_states[-1]
        pooled = mean_pool_last_hidden(hidden_states, encoded["attention_mask"])
        embeddings.append(pooled.detach().cpu())

    stacked = torch.cat(embeddings, dim=0)
    return stacked.numpy()


def stack_embeddings_by_tier(
    tier: int,
    models: List[str],
    store: Dict[Tuple[str, int], np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    """Concatenate embeddings for one tier across multiple models."""
    slices = []
    labels: List[str] = []

    for model_short in models:
        tensor = store.get((model_short, tier))
        if tensor is None or tensor.size == 0:
            continue
        slices.append(tensor)
        labels.extend([model_short] * tensor.shape[0])

    if not slices:
        return np.zeros((0, 0)), []

    return np.concatenate(slices, axis=0), labels


def stack_embeddings_by_model(
    model_short: str,
    tiers: Sequence[int],
    store: Dict[Tuple[str, int], np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    """Concatenate embeddings for one model across multiple tiers."""
    slices = []
    labels: List[str] = []

    for tier in tiers:
        tensor = store.get((model_short, tier))
        if tensor is None or tensor.size == 0:
            continue
        slices.append(tensor)
        labels.extend([f"tier{tier}"] * tensor.shape[0])

    if not slices:
        return np.zeros((0, 0)), []

    return np.concatenate(slices, axis=0), labels


def run_hdbscan_clustering(
    embeddings: np.ndarray,
    min_cluster_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale embeddings and run HDBSCAN."""
    if embeddings.shape[0] == 0:
        return np.array([], dtype=int), np.zeros((0, embeddings.shape[1]))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)

    labels = clusterer.fit_predict(scaled)
    return labels, scaled


def project_embeddings(coords: np.ndarray) -> np.ndarray:
    """Project scaled embeddings down to two dimensions for plotting."""
    if coords.shape[0] == 0:
        return np.zeros((0, 2))
    if coords.shape[0] == 1:
        return np.zeros((1, 2))

    projector = PCA(n_components=2)
    return projector.fit_transform(coords)


def plot_cluster_summary(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    hue_labels: Sequence[str],
    hue_title: str,
    title: str,
    output_path: str,
) -> Dict[str, float]:
    """Produce a 2-panel summary image showing clusters and labels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    axes[0].set_title("HDBSCAN clusters")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[1].set_title(f"{hue_title} coloring")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    num_points = coords.shape[0]
    if num_points > 0:
        cluster_ids = sorted(set(cluster_labels))
        cluster_palette = plt.cm.get_cmap("tab20", max(len(cluster_ids), 1))
        color_map = {
            cid: cluster_palette(idx)
            for idx, cid in enumerate(cluster_ids)
            if cid >= 0
        }
        color_map[-1] = (0.5, 0.5, 0.5, 0.6)

        cluster_colors = np.array(
            [color_map.get(label, color_map[-1]) for label in cluster_labels]
        )
        axes[0].scatter(
            coords[:, 0],
            coords[:, 1],
            c=cluster_colors,
            s=10,
            linewidths=0,
            alpha=0.85,
        )

        noise_ratio = float((cluster_labels == -1).sum()) / num_points
        num_clusters = len([cid for cid in cluster_ids if cid >= 0])
        axes[0].text(
            0.02,
            0.98,
            f"Clusters: {num_clusters}\nNoise: {noise_ratio:.2%}",
            transform=axes[0].transAxes,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        unique_hues = sorted(set(hue_labels), key=lambda value: value)
        hue_palette = plt.cm.get_cmap("tab20", max(len(unique_hues), 1))

        for idx, hue in enumerate(unique_hues):
            mask = np.array(hue_labels) == hue
            axes[1].scatter(
                coords[mask, 0],
                coords[mask, 1],
                label=hue,
                color=hue_palette(idx),
                s=10,
                linewidths=0,
                alpha=0.85,
            )
        axes[1].legend(title=hue_title, loc="best", fontsize="small")
    else:
        num_clusters = 0
        noise_ratio = 0.0
        axes[0].text(0.5, 0.5, "No data", ha="center", va="center")

    fig.suptitle(title)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return {
        "num_points": float(num_points),
        "num_clusters": float(num_clusters),
        "noise_ratio": float(noise_ratio),
    }


def ensure_output_path(base_dir: str, run_id: str) -> str:
    """Create the output directory for the current run."""
    path = os.path.abspath(os.path.join(base_dir, f"clustering_{run_id}"))
    os.makedirs(path, exist_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster ESM2 checkpoints on tier FASTA files."
    )
    parser.add_argument(
        "--model-size",
        choices=list(MODEL_GROUPS.keys()),
        required=True,
        help="Model size (8M, 35M, or 150M) for generating checkpoints.",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6],
        help="Tiers to process (default: 1 2 3 4 5 6).",
    )
    parser.add_argument(
        "--fasta-root",
        type=str,
        default="/workspace/BioRiskEval_pLM/tier-list",
        help="Directory that holds tier{N}.fasta files.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=512,
        help="Limit the number of sequences per tier (0 = all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tokenization/model batch size for embedding extraction.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=30,
        help="HDBSCAN min_cluster_size parameter.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for embedding extraction.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="bioriskeval/clustering/output",
        help="Base directory to write plots and artifacts.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="esm2-eval-clustering",
        help="W&B project name for logging.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--use-compile",
        action="store_true",
        help="Compile the model with torch.compile() for faster inference.",
    )

    args = parser.parse_args()
    args.tiers = sorted(set(args.tiers))

    if not args.tiers:
        parser.error("At least one tier must be provided.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available in this environment.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_list = MODEL_GROUPS[args.model_size]

    tier_sequences: Dict[int, Dict[str, List[str]]] = {}
    for tier in args.tiers:
        sequences, seq_ids = load_tier_sequences(
            tier, args.fasta_root, args.max_sequences
        )
        tier_sequences[tier] = {"sequences": sequences, "seq_ids": seq_ids}
        print(f"Tier{tier}: loaded {len(sequences)} sequences.")

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = ensure_output_path(args.output_dir, run_id)
    run_name = f"{args.model_size}_clustering_{run_id}"

    wandb_run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "model_size": args.model_size,
            "tiers": args.tiers,
            "max_sequences": args.max_sequences,
            "batch_size": args.batch_size,
            "max_seq_len": args.max_seq_len,
            "min_cluster_size": args.min_cluster_size,
            "device": str(device),
            "use_compile": args.use_compile,
        },
        tags=[args.model_size],
    )

    for tier, tier_data in tier_sequences.items():
        wandb_run.log(
            {f"tier/{tier}/sequence_count": len(tier_data["sequences"])}, commit=False
        )

    embedding_store: Dict[Tuple[str, int], np.ndarray] = {}

    for model_short in tqdm(model_list, desc="Models"):
        full_name = get_full_model_name(model_short)
        print(f"Loading {full_name} on {device}...")
        model, tokenizer = load_esm2_model(model_name=full_name)
        model = setup_model_optimizations(model, device, args.use_compile)
        model.eval()

        for tier in args.tiers:
            sequences = tier_sequences[tier]["sequences"]
            if not sequences:
                continue

            print(f"Embedding {model_short} on tier{tier} ({len(sequences)} seqs)...")
            embeddings = embed_sequences(
                model,
                tokenizer,
                sequences,
                device,
                batch_size=args.batch_size,
                max_seq_len=args.max_seq_len,
            )

            embedding_store[(model_short, tier)] = embeddings

        del model, tokenizer
        torch.cuda.empty_cache()

    # Tier-oriented figures: same tier across different checkpoints.
    for tier in args.tiers:
        stacked, hue_values = stack_embeddings_by_tier(
            tier, model_list, embedding_store
        )
        if stacked.size == 0:
            print(f"Skipping tier{tier} figure (no embeddings).")
            continue

        cluster_labels, scaled = run_hdbscan_clustering(
            stacked, args.min_cluster_size
        )
        coords = project_embeddings(scaled)
        title = f"{args.model_size} checkpoints on tier{tier}"
        fig_name = f"tier_{tier}_per_model.png"
        fig_path = os.path.join(output_path, fig_name)

        metrics = plot_cluster_summary(
            coords,
            cluster_labels,
            hue_values,
            "Model",
            title,
            fig_path,
        )

        wandb_run.log(
            {
                f"tier/{tier}/num_points": metrics["num_points"],
                f"tier/{tier}/num_clusters": metrics["num_clusters"],
                f"tier/{tier}/noise_ratio": metrics["noise_ratio"],
                f"tier/{tier}/figure": wandb.Image(
                    fig_path, caption=title
                ),
            }
        )

    # Model-oriented figures: same checkpoint across tiers.
    for model_short in model_list:
        stacked, hue_values = stack_embeddings_by_model(
            model_short, args.tiers, embedding_store
        )
        if stacked.size == 0:
            print(f"Skipping {model_short} figure (no embeddings).")
            continue

        cluster_labels, scaled = run_hdbscan_clustering(
            stacked, args.min_cluster_size
        )
        coords = project_embeddings(scaled)
        title = f"{model_short} across tiers"
        fig_name = f"{model_short}_per_tier.png"
        fig_path = os.path.join(output_path, fig_name)

        metrics = plot_cluster_summary(
            coords,
            cluster_labels,
            hue_values,
            "Tier",
            title,
            fig_path,
        )

        wandb_run.log(
            {
                f"model/{model_short}/num_points": metrics["num_points"],
                f"model/{model_short}/num_clusters": metrics["num_clusters"],
                f"model/{model_short}/noise_ratio": metrics["noise_ratio"],
                f"model/{model_short}/figure": wandb.Image(
                    fig_path, caption=title
                ),
            }
        )

    wandb_run.finish()


if __name__ == "__main__":
    main()
