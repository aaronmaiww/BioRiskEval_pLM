#!/bin/bash

# Tmux script to run checkpoints distributed across multiple GPUs
# Each GPU runs multiple checkpoints sequentially in a separate pane

SESSION_NAME="esm2-eval-tier6"
TIER="6"
BATCH_SIZE="1024"
MASK_CHUNK_SIZE="256"
NUM_PREFETCH="4"

# Virtual environment path
VENV_PATH="/venv/main"

# Number of GPUs available
NUM_GPUS=4  # Adjust this to your setup

# List of all checkpoints
CHECKPOINTS=(
    "8M_T1" "8M_T2" "8M_T5" "8M_T6" "8M_H" "8M_F"
    "35M_T1" "35M_T2" "35M_T5" "35M_T6" "35M_H" "35M_F"
    "150M_T1" "150M_T2" "150M_T5" "150M_T6" "150M_H" "150M_F"
)

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

echo "Creating tmux session: $SESSION_NAME"
echo "Distributing ${#CHECKPOINTS[@]} checkpoints across $NUM_GPUS GPUs"

# Create new tmux session
tmux new-session -d -s $SESSION_NAME -n "eval"

# Distribute checkpoints across GPUs
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    # Get checkpoints for this GPU
    gpu_checkpoints=()
    for i in "${!CHECKPOINTS[@]}"; do
        if [ $((i % NUM_GPUS)) -eq $gpu_id ]; then
            gpu_checkpoints+=("${CHECKPOINTS[$i]}")
        fi
    done
    
    # Build command to run all checkpoints for this GPU sequentially
    cmd="source $VENV_PATH/bin/activate && "
    cmd+="cd /workspace/BioRiskEval_pLM && "
    cmd+="export CUDA_VISIBLE_DEVICES=$gpu_id; "
    cmd+="echo '=== GPU $gpu_id: Running ${#gpu_checkpoints[@]} checkpoints ==='; "
    
    for ckpt in "${gpu_checkpoints[@]}"; do
        cmd+="echo '--- GPU $gpu_id: Starting checkpoint $ckpt ---'; "
        cmd+="python bioriskeval/gen/eval_ppl_esm2.py "
        cmd+="--tier $TIER "
        cmd+="--ckpt-path given131/$ckpt "
        cmd+="--batch-size $BATCH_SIZE "
        cmd+="--mask-chunk-size $MASK_CHUNK_SIZE "
        cmd+="--num-prefetch $NUM_PREFETCH; "
        cmd+="echo '--- GPU $gpu_id: Completed checkpoint $ckpt ---'; "
    done
    
    cmd+="echo '=== GPU $gpu_id: All checkpoints completed ==='; "
    cmd+="echo 'Press any key to exit.'; read"
    
    # Create pane for this GPU
    if [ $gpu_id -eq 0 ]; then
        # First GPU uses the initial pane
        tmux send-keys -t $SESSION_NAME "$cmd" C-m
    else
        # Create new pane for other GPUs
        tmux split-window -t $SESSION_NAME -h
        tmux send-keys -t $SESSION_NAME "$cmd" C-m
        # Balance layout after each split
        tmux select-layout -t $SESSION_NAME tiled
    fi
done

# Final layout adjustment
tmux select-layout -t $SESSION_NAME tiled

# Attach to the session
echo "All checkpoints distributed and started!"
echo "Layout: $NUM_GPUS panes (one per GPU)"
echo "Attaching to session..."
tmux attach-session -t $SESSION_NAME

