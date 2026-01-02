import torch
import numpy as np
import h5py
import argparse
import wandb
import os
import random
import csv
import glob
import re
from pathlib import Path
from typing import Tuple, cast, Optional

"""
This script trains a linear regression probe (single linear layer) on a dataset
with continuous labels.
"""

def parse_str_to_bool(value: str) -> bool:
    """Parse common string representations of truthy/falsey to bool."""
    return value.lower() in ("1", "true", "t", "yes", "y")

def read_probe_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(dataset_path, 'r') as f:
        ds_seq = cast(h5py.Dataset, f['sequences'])
        ds_rep = cast(h5py.Dataset, f['representations'])
        ds_lab = cast(h5py.Dataset, f['labels'])
        sequences: np.ndarray = ds_seq[:]
        representations: np.ndarray = ds_rep[:]
        # Note: original logic scales when '30' in path; keep for parity
        # if '30' in dataset_path:
        #     print("Scaling representations by 1e-10 (dividing by 1e10) for layer 30 files")
        #     representations = representations / 1e10
        labels: np.ndarray = ds_lab[:]
    return sequences, representations, labels


def solve_linear_probe(representations: np.ndarray, labels: np.ndarray, args: Optional[argparse.Namespace] = None) -> torch.nn.Module:
    """Solve linear regression probe using closed-form solution (normal equation).
    
    This computes the optimal weights directly without iterative training:
    w = (X^T X)^(-1) X^T y
    
    Parameters
    ----------
    representations : np.ndarray
        Array of latent representations with shape (N, D).
    labels : np.ndarray
        Continuous labels with shape (N,).
    args : argparse.Namespace, optional
        Command-line arguments (mainly for wandb logging).
    
    Returns
    -------
    torch.nn.Module
        Linear probe with optimal weights set.
    """
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to tensors
    x_tensor = torch.tensor(representations, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1)
    
    # Add bias term (column of ones) to X
    N, D = x_tensor.shape
    x_with_bias = torch.cat([x_tensor, torch.ones(N, 1, device=device)], dim=1)  # (N, D+1)
    
    # Solve using pseudoinverse: w = X^+ y
    # This is more numerically stable than (X^T X)^(-1) X^T y
    try:
        # Compute pseudoinverse
        x_pinv = torch.linalg.pinv(x_with_bias)  # (D+1, N)
        optimal_weights = x_pinv @ y_tensor  # (D+1, 1)
        
        # Split weights and bias
        w = optimal_weights[:-1, 0]  # (D,)
        b = optimal_weights[-1, 0]   # scalar
        
    except Exception as e:
        print(f"Warning: Pseudoinverse failed ({e}), falling back to lstsq solution")
        # Fallback to least squares solution
        solution = torch.linalg.lstsq(x_with_bias, y_tensor, rcond=None)
        optimal_weights = solution.solution  # (D+1, 1)
        w = optimal_weights[:-1, 0]
        b = optimal_weights[-1, 0]
    
    # Create probe and set optimal weights
    probe = torch.nn.Linear(D, 1).to(device)
    with torch.no_grad():
        probe.weight.copy_(w.unsqueeze(0))  # (1, D)
        probe.bias.copy_(b.unsqueeze(0))    # (1,)
    
    # Compute training loss for logging
    with torch.no_grad():
        preds = probe(x_tensor)
        mse_loss = torch.nn.functional.mse_loss(preds, y_tensor).item()
        mae_loss = torch.nn.functional.l1_loss(preds, y_tensor).item()
    
    print(f"Closed-form solution found:")
    print(f"  Training MSE: {mse_loss:.6f}")
    print(f"  Training MAE: {mae_loss:.6f}")
    
    # Log to wandb if enabled
    if args and hasattr(args, 'wandb') and args.wandb:
        wandb.log({
            "model/input_dim": D,
            "model/output_dim": 1,
            "dataset/num_samples": N,
            "labels/mean": float(labels.mean()),
            "labels/std": float(labels.std()),
            "labels/min": float(labels.min()),
            "labels/max": float(labels.max()),
            "train/method": "closed_form",
            "train/final_mse": mse_loss,
            "train/final_mae": mae_loss,
        })
    
    return probe


def evaluate_probe(probe, test_representations, test_labels, args=None):
    """Evaluate the trained regression probe on a held-out set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(test_representations, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(test_labels, dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1)
        preds = probe(x_tensor)

        # Compute regression metrics
        errors = preds - y_tensor
        mse = torch.mean(errors ** 2).item()
        rmse = float(np.sqrt(mse))
        mae = torch.mean(torch.abs(errors)).item()

        y_true_np = y_tensor.squeeze(1).detach().cpu().numpy()
        y_pred_np = preds.squeeze(1).detach().cpu().numpy()
        y_true_mean = float(y_true_np.mean())
        ss_res = float(np.sum((y_true_np - y_pred_np) ** 2))
        ss_tot = float(np.sum((y_true_np - y_true_mean) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        # Pearson correlation
        if np.std(y_true_np) > 0 and np.std(y_pred_np) > 0:
            pearson = float(np.corrcoef(y_true_np, y_pred_np)[0, 1])
        else:
            pearson = 0.0
        
        # Log test metrics to wandb
        if args and hasattr(args, 'wandb') and args.wandb:
            wandb.log({
                "test/mse": mse,
                "test/rmse": rmse,
                "test/mae": mae,
                "test/r2": r2,
                "test/pearson": pearson,
            })
    
    return rmse, mae, r2, pearson





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing virulence_probe_dataset_layer_*_train.h5 and virulence_probe_dataset_layer_*_test.h5 files")
    
    # Training duration options (mutually exclusive)
    training_group = parser.add_mutually_exclusive_group(required=False)
    training_group.add_argument("--num_steps", type=int, default=100, help="Number of training steps (default: 100)")
    training_group.add_argument("--num_epochs", type=int, help="Number of training epochs (alternative to --num_steps)")
    
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "mae", "huber"], help="Regression loss to use")
    parser.add_argument("--huber_delta", type=float, default=1.0, help="Delta parameter for Huber/SmoothL1 loss when --loss huber is selected")
    parser.add_argument("--shuffle_labels", type=str, default="False", help="Shuffle labels, for ablation study (accepts true/false)")
    parser.add_argument("--use_closed_form", action="store_true", help="Use closed-form solution instead of iterative training")
    # Wandb arguments
    parser.add_argument("--output_csv", type=str, default="probe_results_continuous.csv")
    parser.add_argument("--normalize_features", action="store_true", help="Normalize features")
    parser.add_argument("--custom_weights", type=str, default=None, help="Path to custom weights file (.pt or .pth) for ESM2 model (note: this script uses pre-extracted representations)")
    args = parser.parse_args()
    
    # Find all layer files in the directory
    train_files = sorted(glob.glob(os.path.join(args.dataset_dir, "virulence_probe_dataset_layer_*_train.h5")))
    
    if not train_files:
        raise ValueError(f"No training files found in {args.dataset_dir}")
    
    # Extract layer numbers from filenames
    layer_numbers = []
    for train_file in train_files:
        match = re.search(r'layer_(\d+)_train\.h5', os.path.basename(train_file))
        if match:
            layer_numbers.append(int(match.group(1)))
    
    layer_numbers = sorted(layer_numbers)
    print(f"Found {len(layer_numbers)} layers to probe: {layer_numbers}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Initialize CSV file
    result_file = Path(args.output_csv)
    file_exists = result_file.exists()
    with result_file.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["train_dataset_path", "test_dataset_path", "method", "probing_layer","learning_rate", "batch_size", "num_steps", "num_epochs", "loss", "rmse", "mae", "r2", "pearson", "shuffle_labels", "custom_weights"])
    
    # Process each layer
    for layer_num in layer_numbers:
        print(f"\n{'='*80}")
        print(f"Processing Layer {layer_num}")
        print(f"{'='*80}")
        
        train_dataset_path = os.path.join(args.dataset_dir, f"virulence_probe_dataset_layer_{layer_num}_train.h5")
        test_dataset_path = os.path.join(args.dataset_dir, f"virulence_probe_dataset_layer_{layer_num}_test.h5")
        
        if not os.path.exists(train_dataset_path):
            print(f"Warning: Training file not found: {train_dataset_path}")
            continue
        if not os.path.exists(test_dataset_path):
            print(f"Warning: Test file not found: {test_dataset_path}")
            continue
        
        # Load datasets
        train_sequences, train_representations, train_labels = read_probe_dataset(train_dataset_path)
        test_sequences, test_representations, test_labels = read_probe_dataset(test_dataset_path)

        # Optionally shuffle training labels for ablation
        if args.shuffle_labels == "True":
            print("Shuffling training labels")
            rng = np.random.default_rng(args.seed)
            rng.shuffle(train_labels)

        # Optional feature standardization
        if args.normalize_features:
            # Normalize each vector to have L2 norm = 1
            print("Normalizing features to unit vectors...")
            
            # # Compute L2 norms for each vector (row) in train_representations
            train_norms = np.linalg.norm(train_representations, axis=1, keepdims=True)
            # Avoid division by zero - replace zero norms with 1
            train_norms = np.where(train_norms == 0, 1, train_norms)
            # Normalize train representations
            train_representations = train_representations / train_norms
            
            # Compute L2 norms for each vector (row) in test_representations
            test_norms = np.linalg.norm(test_representations, axis=1, keepdims=True)
            # Avoid division by zero - replace zero norms with 1
            test_norms = np.where(test_norms == 0, 1, test_norms)
            # Normalize test representations
            test_representations = test_representations / test_norms

        
        print("Using closed-form solution for linear regression...")
        probe = solve_linear_probe(train_representations, train_labels, args)

        rmse, mae, r2, pearson = evaluate_probe(probe, test_representations, test_labels, args)
        print(f"Test RMSE: {rmse:.6f}")
        print(f"Test MAE: {mae:.6f}")
        print(f"Test R2: {r2:.6f}")
        print(f"Test Pearson: {pearson:.6f}")

        # Create dataset path names for CSV
        dataset_dir_name = os.path.basename(args.dataset_dir)
        train_path_name = f"{dataset_dir_name}/virulence_probe_dataset_layer_{layer_num}_train"
        test_path_name = f"{dataset_dir_name}/virulence_probe_dataset_layer_{layer_num}_test"

        # Append results to CSV
        with result_file.open("a", newline="") as f:
            writer = csv.writer(f)
            
            method = "closed_form" if args.use_closed_form else "iterative"
            learning_rate = "N/A" if args.use_closed_form else args.learning_rate
            batch_size = "N/A" if args.use_closed_form else args.batch_size
            num_steps = "N/A" if args.use_closed_form else args.num_steps
            num_epochs = "N/A" if args.use_closed_form else args.num_epochs
            loss = "N/A" if args.use_closed_form else args.loss
            
            custom_weights_info = "N/A" if not args.custom_weights else os.path.basename(args.custom_weights)
            writer.writerow([train_path_name, test_path_name, method, layer_num, learning_rate, batch_size, num_steps, num_epochs, loss, f"{rmse:.6f}", f"{mae:.6f}", f"{r2:.6f}", f"{pearson:.6f}", args.shuffle_labels, custom_weights_info])
    
    print(f"\n{'='*80}")
    print(f"All layers processed. Results saved to {args.output_csv}")
    print(f"{'='*80}")
    