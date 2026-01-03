#!/bin/bash

# Sequential execution of all checkpoints (no tmux, one by one)

# Virtual environment path
VENV_PATH="/venv/main"

TIER="6"
BATCH_SIZE="1024"
MASK_CHUNK_SIZE="256"
NUM_PREFETCH="4"

# List of all checkpoints
CHECKPOINTS=(
    "8M_T1" "8M_T2" "8M_T5" "8M_T6" "8M_H" "8M_F"
    "35M_T1" "35M_T2" "35M_T5" "35M_T6" "35M_H" "35M_F"
    "150M_T1" "150M_T2" "150M_T5" "150M_T6" "150M_H" "150M_F"
)

TOTAL=${#CHECKPOINTS[@]}

# Activate virtual environment
echo "Activating virtual environment: $VENV_PATH"
source $VENV_PATH/bin/activate

# Change to workspace directory
cd /workspace/BioRiskEval_pLM

echo "=================================="
echo "Running $TOTAL checkpoints sequentially"
echo "Tier: $TIER"
echo "=================================="

for i in "${!CHECKPOINTS[@]}"; do
    CKPT=${CHECKPOINTS[$i]}
    PROGRESS=$((i + 1))
    
    echo ""
    echo "[$PROGRESS/$TOTAL] Starting checkpoint: $CKPT"
    echo "-----------------------------------"
    
    python bioriskeval/gen/eval_ppl_esm2.py \
        --tier $TIER \
        --ckpt-path given131/$CKPT \
        --batch-size $BATCH_SIZE \
        --mask-chunk-size $MASK_CHUNK_SIZE \
        --num-prefetch $NUM_PREFETCH
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Checkpoint $CKPT completed successfully"
    else
        echo "✗ Checkpoint $CKPT failed with exit code $EXIT_CODE"
        read -p "Continue with next checkpoint? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Stopping execution."
            exit 1
        fi
    fi
done

echo ""
echo "=================================="
echo "All checkpoints completed!"
echo "=================================="

