#!/bin/bash

# Optimized ESM2 evaluation script for 32GB GPU
# This script maximizes GPU memory usage for faster processing

set -e

# Default parameters for 32GB GPU
DEFAULT_BATCH_SIZE=1024
DEFAULT_MAX_SEQ_LEN=1024
DEFAULT_TIER="1"
DEFAULT_MODEL="facebook/esm2_t6_8M_UR50D"

# Parse command line arguments
BATCH_SIZE=${1:-$DEFAULT_BATCH_SIZE}
TIER=${2:-$DEFAULT_TIER}
MODEL=${3:-$DEFAULT_MODEL}
MAX_SEQ_LEN=${4:-$DEFAULT_MAX_SEQ_LEN}

echo "=== Optimized ESM2 Evaluation ==="
echo "Batch size: $BATCH_SIZE"
echo "Tier: $TIER"
echo "Model: $MODEL"
echo "Max sequence length: $MAX_SEQ_LEN"
echo "================================="

# Check GPU memory
echo "GPU Memory Info:"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits

# Set CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run evaluation with optimized settings
python eval_ppl_esm2.py \
    --tier "$TIER" \
    --ckpt-path "$MODEL" \
    --batch-size "$BATCH_SIZE" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --use-fp16 \
    --aggregate mean

echo "Evaluation completed!"
echo "Final GPU Memory Info:"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits



