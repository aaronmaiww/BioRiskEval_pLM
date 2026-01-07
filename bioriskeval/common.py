"""
Common utilities for BioRiskEval including model loading, sequence processing, and profiling.

Profiling Usage:
    # Enable profiling to identify performance bottlenecks
    scores = compute_pseudo_ppl_hf_batch(
        sequences=my_sequences,
        model=model,
        tokenizer=tokenizer,
        enable_profiling=True,  # Add this line
    )
    
    # Output will show time breakdown like:
    # 4d_model_forward         78.5%  <- Model inference (usually the bottleneck)
    # 4e_log_prob_computation  13.3%  <- Post-processing
    # 4a_cpu_to_gpu_transfer    5.3%  <- Data transfer
    # 3_dataloader_iteration    2.0%  <- CPU preprocessing
    # ...

Manual profiling:
    from bioriskeval.common import PerformanceProfiler, CUDATimer
    
    profiler = PerformanceProfiler()
    
    with profiler.profile("my_section"):
        # code to profile
        pass
    
    profiler.print_summary()
    
    # Or use CUDA timer for GPU operations
    with CUDATimer("gpu_operation"):
        result = my_gpu_function()
"""

import contextlib
import gc
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# import flash_attn  # noqa: F401 - Import to ensure Flash Attention is available
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


# ============================================================================
# Profiling Utilities
# ============================================================================

class CUDATimer:
    """
    High-precision CUDA timer using CUDA events.
    More accurate than time.time() for GPU operations.
    """
    
    def __init__(self, name: str = "", enabled: bool = True):
        self.name = name
        self.enabled = enabled and torch.cuda.is_available()
        self.start_event = None
        self.end_event = None
        
    def __enter__(self):
        if self.enabled:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        return self
    
    def __exit__(self, *args):
        if self.enabled:
            self.end_event.record()
            torch.cuda.synchronize()
            elapsed = self.start_event.elapsed_time(self.end_event)
            if self.name:
                print(f"[CUDA Timer] {self.name}: {elapsed:.2f}ms")
    
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.enabled and self.start_event and self.end_event:
            torch.cuda.synchronize()
            return self.start_event.elapsed_time(self.end_event)
        return 0.0


class PerformanceProfiler:
    """
    Lightweight profiler that tracks timing for different code sections.
    Supports both CPU and GPU timing with CUDA events.
    
    Usage:
        profiler = PerformanceProfiler()
        
        with profiler.profile("section_name"):
            # code to profile
            pass
        
        profiler.print_summary()
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.use_cuda = torch.cuda.is_available()
        
    @contextlib.contextmanager
    def profile(self, name: str):
        """Context manager for profiling a code section."""
        if not self.enabled:
            yield
            return
            
        if self.use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            
            try:
                yield
            finally:
                end_event.record()
                torch.cuda.synchronize()
                elapsed = start_event.elapsed_time(end_event)
                self.timings[name].append(elapsed)
        else:
            start = time.perf_counter()
            try:
                yield
            finally:
                elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
                self.timings[name].append(elapsed)
    
    def get_summary(self) -> str:
        """Generate a formatted summary of all profiled sections."""
        if not self.timings:
            return "No profiling data collected."
        
        lines = ["=" * 80, "Performance Profiling Summary", "=" * 80]
        
        # Calculate statistics for each section
        results = []
        total_time = 0.0
        
        for name, times in self.timings.items():
            total = sum(times)
            count = len(times)
            avg = total / count
            
            results.append({
                'name': name,
                'total': total,
                'count': count,
                'avg': avg,
            })
            total_time += total
        
        # Sort by total time (descending)
        results.sort(key=lambda x: x['total'], reverse=True)
        
        # Print table header
        lines.append(f"{'Section':<40} {'Total (ms)':<12} {'Avg (ms)':<12} {'Count':<8} {'% Total':<8}")
        lines.append("-" * 80)
        
        # Print each section
        for r in results:
            pct = (r['total'] / total_time * 100) if total_time > 0 else 0
            lines.append(
                f"{r['name']:<40} {r['total']:>11.2f} {r['avg']:>11.2f} {r['count']:>7} {pct:>7.1f}%"
            )
        
        lines.append("-" * 80)
        lines.append(f"{'TOTAL':<40} {total_time:>11.2f}")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def reset(self):
        """Clear all collected timing data."""
        self.timings.clear()
    
    def print_summary(self):
        """Print the profiling summary."""
        print(self.get_summary())


# ============================================================================
# Model Loading and Setup
# ============================================================================


def parse_model_tier(model_name: str) -> str:
    """
    Parse model tier from HuggingFace model name.

    Args:
        model_name (str): Model name like "given131/8M_T1", "facebook/esm2_t6_8M_UR50D",
                          or new format like "given131/35M_R_80P"
    Returns:
        str: Tier number (e.g., '1', '2', '3', 'H', 'F') or 'unknown'
    """
    import re

    # Check for traditional tier format (T1, T2, etc.) with word boundary
    tier_match = re.search(r'_T(\d+)(?:_|$)', model_name)
    if tier_match:
        return tier_match.group(1)

    # Check for H or F tier with word boundary (e.g., "_H_" or "_H" at end)
    if re.search(r'_H(?:_|$)', model_name):
        return "H"
    if re.search(r'_F(?:_|$)', model_name):
        return "F"

    # Facebook models: extract tier from architecture name
    if "t6" in model_name.lower() and "esm2" in model_name.lower():
        return "6"
    elif "t12" in model_name.lower() and "esm2" in model_name.lower():
        return "12"
    elif "t30" in model_name.lower() and "esm2" in model_name.lower():
        return "30"

    # 기본값으로 "unknown" 반환 (에러 대신)
    return "unknown"


def parse_model_type(model_name: str) -> str:
    """
    Parse model type from new naming convention (e.g., R, C, I).

    Args:
        model_name (str): Model name like "given131/35M_R_80P" or "35M_C_P_80P_D2"
    Returns:
        str: Model type (e.g., 'R', 'C', 'I') or 'unknown'
    """
    import re
    # Pattern: after size (8M, 35M, 150M), look for type letter
    # e.g., "35M_R_80P" -> R, "35M_C_P_80P_D2" -> C
    match = re.search(r'(?:8M|35M|150M)_([A-Z])(?:_|$)', model_name)
    if match:
        return match.group(1)
    return "unknown"


def parse_model_subtype(model_name: str) -> str:
    """
    Parse model subtype from new naming convention (e.g., P, U).

    Args:
        model_name (str): Model name like "35M_C_P_80P_D2" or "35M_I_U_40P"
    Returns:
        str: Model subtype (e.g., 'P', 'U') or None if not present
    """
    import re
    # Pattern: after size and type, look for single letter subtype before percentage
    # e.g., "35M_C_P_80P" -> P, "35M_I_U_40P" -> U, "35M_R_80P" -> None
    match = re.search(r'(?:8M|35M|150M)_[A-Z]_([A-Z])_\d+P', model_name)
    if match:
        return match.group(1)
    return None


def parse_model_percentage(model_name: str) -> str:
    """
    Parse data percentage from model name (e.g., 80P, 40P).

    Args:
        model_name (str): Model name like "35M_R_80P" or "35M_C_P_80P_D2"
    Returns:
        str: Percentage string (e.g., '80P', '40P') or None if not present
    """
    import re
    # Pattern: look for number followed by P (percentage marker)
    match = re.search(r'(\d+P)(?:_|$)', model_name)
    if match:
        return match.group(1)
    return None


def parse_model_duplication(model_name: str) -> str:
    """
    Parse duplication info from model name (e.g., D2).

    Args:
        model_name (str): Model name like "35M_C_P_80P_D2"
    Returns:
        str: Duplication string (e.g., 'D2') or None if not present
    """
    import re
    # Pattern: look for D followed by number
    match = re.search(r'(D\d+)(?:_|$)', model_name)
    if match:
        return match.group(1)
    return None


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



def load_esm2_model(model_name: str) -> tuple:
    """
    Load ESM2 model using HuggingFace transformers.
    
    Args:
        ckpt_path (str): HuggingFace model name (e.g., "given131/8M_T1" or "facebook/esm2_t6_8M_UR50D")
    Returns:
        model: HuggingFace EsmForMaskedLM model
        tokenizer: HuggingFace ESM2 tokenizer
    """
    # Parse model size and get corresponding Facebook config
    model_size = parse_model_size(model_name)
    facebook_model = FACEBOOK_CONFIG[model_size]
    print(f"Using custom model {model_name} with architecture from {facebook_model}")
    
    # Initialize tokenizer and model from Facebook architecture
    tokenizer = AutoTokenizer.from_pretrained(facebook_model)
    
    # Load model with attention implementation
    model = EsmForMaskedLM.from_pretrained(
        facebook_model,
        # attn_implementation="flash_attention_2",
    )
    
    # Load custom weights from HuggingFace model
    try:
        print(f"Loading custom weights from HuggingFace model: {model_name}")
        from huggingface_hub import hf_hub_download

        # Download model.bin file
        model_bin_path = hf_hub_download(repo_id=model_name, filename="model.bin")
        custom_state_dict = torch.load(model_bin_path, map_location='cpu')
        
        # Extract model_state_dict from the ordered_dict
        if 'model_state_dict' in custom_state_dict:
            model_weights = custom_state_dict['model_state_dict']
            # print("Found 'model_state_dict' in checkpoint")
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
    # print(f"Model loaded on {device} with BF16 precision")
    
    torch.backends.cudnn.benchmark = True
    
    # Enable PyTorch native SDPA optimizations (fallback/complement to Flash Attention 2)
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)  # Disable slow math fallback
        print("PyTorch SDPA optimizations enabled")

    # Apply torch.compile for optimized execution (PyTorch 2.0+)
    if use_compile and hasattr(torch, 'compile'):
        # print("Compiling model with torch.compile()...")
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
    profiler: Optional[PerformanceProfiler] = None,
) -> List[float]:
    """
    Compute pseudo-perplexity from pre-prepared tensors.
    This function only does GPU compute, allowing CPU preparation to happen in parallel.
    All GPU operations complete before any CPU transfer to minimize GPU-CPU overhead.
    Uses BF16 precision.
    """
    with (profiler.profile("4a_cpu_to_gpu_transfer") if profiler else contextlib.nullcontext()):
        # Use CUDA stream for async data transfer if available
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            expanded_input_ids = expanded_input_ids.to(device, non_blocking=True)
            expanded_attention = expanded_attention.to(device, non_blocking=True)
            positions_flat = positions_flat.to(device, non_blocking=True)
            token_targets_flat = token_targets_flat.to(device, non_blocking=True)
            seq_indices = seq_indices.to(device, non_blocking=True)
        stream.synchronize()
    
    with (profiler.profile("4b_tensor_allocation") if profiler else contextlib.nullcontext()):
        total_positions = expanded_input_ids.size(0)
        
        # Pre-allocate result tensors on GPU
        log_sums = torch.zeros(batch_size, device=device, dtype=torch.float32)
        counts = torch.zeros(batch_size, device=device, dtype=torch.float32)

    # Use BF16 autocast
    autocast_enabled = device.type == "cuda"
    autocast_context = torch.cuda.amp.autocast(dtype=torch.bfloat16) if autocast_enabled else contextlib.nullcontext()

    # Process all chunks - stay on GPU
    for chunk_start in range(0, total_positions, mask_chunk_size):
        with (profiler.profile("4c_chunk_slicing") if profiler else contextlib.nullcontext()):
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

        with (profiler.profile("4d_model_forward") if profiler else contextlib.nullcontext()):
            with torch.inference_mode(), autocast_context:
                logits = model(chunk_inputs, attention_mask=chunk_attention).logits
        
        with (profiler.profile("4e_log_prob_computation") if profiler else contextlib.nullcontext()):
            with torch.inference_mode(), autocast_context:
                log_probs = F.log_softmax(logits.float(), dim=-1)
                token_log_probs = log_probs[
                    torch.arange(chunk_batch, device=device),
                    chunk_positions,
                    chunk_targets
                ]

        with (profiler.profile("4f_scatter_accumulation") if profiler else contextlib.nullcontext()):
            log_sums.scatter_add_(0, chunk_seq_indices, token_log_probs)
            counts.scatter_add_(0, chunk_seq_indices, torch.ones_like(token_log_probs))
    
    with (profiler.profile("4g_final_aggregation") if profiler else contextlib.nullcontext()):
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
    
    with (profiler.profile("4h_gpu_to_cpu_transfer") if profiler else contextlib.nullcontext()):
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
    enable_profiling: bool = False,
    profiler: Optional[PerformanceProfiler] = None,
) -> List[float]:
    """
    Process a group of similar-length sequences in batches using DataLoader.
    Uses PyTorch DataLoader for efficient batch prefetching and multiprocessing.
    Uses BF16 precision.
    
    Args:
        num_workers: Number of workers for DataLoader prefetching (default 4)
        enable_profiling: Enable detailed performance profiling (default False)
        profiler: Optional PerformanceProfiler instance for collecting metrics
    """
    scores = []
    
    if not sequences:
        return scores
    
    # Initialize profiler if requested
    if enable_profiling and profiler is None:
        profiler = PerformanceProfiler(enabled=True)
    
    with (profiler.profile("1_dataset_creation") if profiler else contextlib.nullcontext()):
        # Create dataset and dataloader
        dataset = ProteinSequenceDataset(sequences)
        
        # Create collate function with tokenizer and max_seq_len bound
        def collate_fn(batch):
            return collate_batch_tensors(batch, tokenizer, max_seq_len)
    
    with (profiler.profile("2_dataloader_creation") if profiler else contextlib.nullcontext()):
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
        with (profiler.profile("3_dataloader_iteration") if profiler else contextlib.nullcontext()):
            expanded_input_ids, expanded_attention, positions_flat, token_targets_flat, seq_indices, batch_size = tensors
        
        if expanded_input_ids is None:
            # Empty batch
            scores.extend([float("nan")] * batch_size)
            continue
        
        with (profiler.profile("4_compute_batch") if profiler else contextlib.nullcontext()):
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
                profiler=profiler,
            )
        scores.extend(batch_scores)
        
        # Clear GPU cache periodically
        if batch_idx % 8 == 0 and batch_idx > 0:
            with (profiler.profile("5_memory_cleanup") if profiler else contextlib.nullcontext()):
                cleanup_gpu_memory()
    
    if profiler and enable_profiling:
        profiler.print_summary()
    
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
    enable_profiling: bool = False,
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
        enable_profiling: Enable detailed performance profiling (default False)
    Returns:
        List of perplexity scores
    """
    device = next(model.parameters()).device
    scores = []
    
    # Initialize profiler if requested
    profiler = None
    if enable_profiling:
        profiler = PerformanceProfiler(enabled=True)
    
    with (profiler.profile("0_sequence_grouping") if profiler else contextlib.nullcontext()):
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
            enable_profiling=enable_profiling,
            profiler=profiler,
        )
        scores.extend(group_scores)
    
    return scores
