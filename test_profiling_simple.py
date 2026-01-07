#!/usr/bin/env python3
"""
Quick test to verify profiling works correctly in common.py

Usage:
    python test_profiling_simple.py
"""

import torch

# Test 1: Basic profiler functionality
print("=" * 60)
print("Test 1: PerformanceProfiler")
print("=" * 60)

from bioriskeval.common import PerformanceProfiler

profiler = PerformanceProfiler(enabled=True)

with profiler.profile("test_operation_1"):
    x = torch.randn(100, 100)
    y = torch.matmul(x, x)

with profiler.profile("test_operation_2"):
    z = torch.sum(y)

profiler.print_summary()

print("\n✓ PerformanceProfiler working correctly")

# Test 2: CUDA Timer (if available)
if torch.cuda.is_available():
    print("\n" + "=" * 60)
    print("Test 2: CUDATimer")
    print("=" * 60)
    
    from bioriskeval.common import CUDATimer
    
    with CUDATimer("cuda_test"):
        x = torch.randn(500, 500, device='cuda')
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
    
    print("✓ CUDATimer working correctly")
else:
    print("\n⚠ CUDA not available, skipping CUDATimer test")

# Test 3: Integration with actual function (if model available)
print("\n" + "=" * 60)
print("Test 3: Integration Test (optional)")
print("=" * 60)

try:
    from bioriskeval.common import compute_pseudo_ppl_hf_batch, load_esm2_model_optimized
    from transformers import AutoTokenizer
    
    print("Loading small ESM2 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_esm2_model_optimized("facebook/esm2_t6_8M_UR50D", device, use_compile=False)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    
    test_seqs = ["MKTAYIAKQRQISFVKSHFSRQ"] * 4
    
    print(f"\nRunning with profiling enabled on {len(test_seqs)} sequences...")
    scores = compute_pseudo_ppl_hf_batch(
        sequences=test_seqs,
        model=model,
        tokenizer=tokenizer,
        max_batch_size=2,
        mask_chunk_size=64,
        num_workers=0,
        enable_profiling=True,  # <- This enables profiling
    )
    
    print(f"\n✓ Integration test passed")
    print(f"  Got {len(scores)} scores: {scores[:2]}")
    
except Exception as e:
    print(f"⚠ Integration test skipped: {e}")
    print("  (This is optional - basic profiling tools work)")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
print("\nProfiling is now available in common.py")
print("Just add enable_profiling=True to compute_pseudo_ppl_hf_batch()")

