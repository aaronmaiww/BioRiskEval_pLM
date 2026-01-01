import contextlib
import gc
import time
from typing import List, Optional, Tuple

import flash_attn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForMaskedLM

FACEBOOK_CONFIG = {
    "8M":   "facebook/esm2_t6_8M_UR50D",
    "35M":  "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
}


def parse_model_tier(model_name: str) -> str:
    """
    Parse model tier from HuggingFace model name.
    
    Args:
        model_name (str): Model name like "given131/8M_T1" or "facebook/esm2_t6_8M_UR50D"
    Returns:
        str: Tier number (e.g., '1', '2', '3', 'H', 'F')
    """
    if "T1" in model_name:
        return "1"
    elif "T2" in model_name:
        return "2"
    elif "T5" in model_name:
        return "5"
    elif "T6" in model_name:
        return "6"
    elif "H" in model_name:
        return "H"
    elif "F" in model_name:
        return "F"
    else:
        # Facebook 모델의 경우 tier를 추출
        if "t6" in model_name.lower():
            return "6"
        elif "t12" in model_name.lower():
            return "12"
        elif "t30" in model_name.lower():
            return "30"
        # 기본값으로 "unknown" 반환 (에러 대신)
        return "unknown"


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



def load_esm2_model(ckpt_path: str) -> tuple:
    """
    Load ESM2 model using HuggingFace transformers.
    
    Args:
        ckpt_path (str): HuggingFace model name (e.g., "given131/8M_T1" or "facebook/esm2_t6_8M_UR50D")
    Returns:
        model: HuggingFace EsmForMaskedLM model
        tokenizer: HuggingFace ESM2 tokenizer
    """
    # Parse model size and get corresponding Facebook config
    model_size = parse_model_size(ckpt_path)
    facebook_model = FACEBOOK_CONFIG[model_size]
    print(f"Using custom model {ckpt_path} with architecture from {facebook_model}")
    
    # Initialize tokenizer and model from Facebook architecture
    tokenizer = AutoTokenizer.from_pretrained(facebook_model)
    
    # Load model with attention implementation
    model = EsmForMaskedLM.from_pretrained(
        facebook_model,
        attn_implementation="flash_attention_2",
    )
    
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
    
    model.eval()
    
    return model, tokenizer


def cleanup_gpu_memory():
    """Clean up GPU memory."""
    torch.cuda.empty_cache()
    gc.collect()


def setup_model_optimizations(model, device, use_compile: bool = False):
    model = model.to(device)
    
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
    
    return model


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
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        expanded_input_ids = expanded_input_ids.to(device, non_blocking=True)
        expanded_attention = expanded_attention.to(device, non_blocking=True)
        positions_flat = positions_flat.to(device, non_blocking=True)
        token_targets_flat = token_targets_flat.to(device, non_blocking=True)
        seq_indices = seq_indices.to(device, non_blocking=True)
    stream.synchronize()
    
    total_positions = expanded_input_ids.size(0)
    
    # Pre-allocate result tensors on GPU
    log_sums = torch.zeros(batch_size, device=device, dtype=torch.float32)
    counts = torch.zeros(batch_size, device=device, dtype=torch.float32)

    # Use BF16 autocast
    autocast_enabled = device.type == "cuda"
    autocast_context = torch.cuda.amp.autocast(dtype=torch.bfloat16) if autocast_enabled else contextlib.nullcontext()

    # Process all chunks - stay on GPU
    for chunk_start in tqdm(range(0, total_positions, mask_chunk_size)):
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
            print("batch size: ", chunk_inputs.size(0))
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
