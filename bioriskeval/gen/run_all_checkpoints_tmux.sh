#!/bin/bash

# Tmux script to run all checkpoints in parallel
# Each checkpoint runs in a separate pane

SESSION_NAME="esm2-eval-tier6"
TIER="6"
BATCH_SIZE="1024"
MASK_CHUNK_SIZE="256"
NUM_PREFETCH="4"

# Virtual environment path
VENV_PATH="/venv/main"

# List of all checkpoints
CHECKPOINTS=(
    "8M_T1" "8M_T2" "8M_T5" "8M_T6" "8M_H" "8M_F"
    "35M_T1" "35M_T2" "35M_T5" "35M_T6" "35M_H" "35M_F"
    "150M_T1" "150M_T2" "150M_T5" "150M_T6" "150M_H" "150M_F"
)

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session with first checkpoint
echo "Creating tmux session: $SESSION_NAME"
FIRST_CKPT=${CHECKPOINTS[0]}
tmux new-session -d -s $SESSION_NAME -n "eval" \
    "source $VENV_PATH/bin/activate && \
    cd /workspace/BioRiskEval_pLM && \
    python bioriskeval/gen/eval_ppl_esm2.py \
    --tier $TIER \
    --ckpt-path given131/$FIRST_CKPT \
    --batch-size $BATCH_SIZE \
    --mask-chunk-size $MASK_CHUNK_SIZE \
    --num-prefetch $NUM_PREFETCH; \
    echo 'Checkpoint $FIRST_CKPT completed. Press any key to exit.'; read"

# Split window and create panes for remaining checkpoints
for i in "${!CHECKPOINTS[@]}"; do
    if [ $i -eq 0 ]; then
        continue  # Skip first checkpoint (already created)
    fi
    
    CKPT=${CHECKPOINTS[$i]}
    
    # Split window (alternate between horizontal and vertical for better layout)
    if [ $((i % 2)) -eq 1 ]; then
        tmux split-window -t $SESSION_NAME -h
    else
        tmux split-window -t $SESSION_NAME -v
    fi
    
    # Send command to the new pane
    tmux send-keys -t $SESSION_NAME \
        "source $VENV_PATH/bin/activate && \
        cd /workspace/BioRiskEval_pLM && \
        python bioriskeval/gen/eval_ppl_esm2.py \
        --tier $TIER \
        --ckpt-path given131/$CKPT \
        --batch-size $BATCH_SIZE \
        --mask-chunk-size $MASK_CHUNK_SIZE \
        --num-prefetch $NUM_PREFETCH; \
        echo 'Checkpoint $CKPT completed. Press any key to exit.'; read" C-m
    
    # Balance panes for better layout
    tmux select-layout -t $SESSION_NAME tiled
done

# Balance the layout one final time
tmux select-layout -t $SESSION_NAME tiled

# Attach to the session
echo "All checkpoints started in tmux session: $SESSION_NAME"
echo "Attaching to session..."
tmux attach-session -t $SESSION_NAME

