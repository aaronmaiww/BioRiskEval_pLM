#!/bin/bash

# Tmux script to run checkpoints distributed across multiple GPUs
# Each GPU runs up to 4 jobs in parallel

SESSION_NAME="esm2-eval-model-list"
TIER="6"
BATCH_SIZE="1024"
MASK_CHUNK_SIZE="256"
NUM_PREFETCH="4"

# Virtual environment path
VENV_PATH="/venv/main"

# Project root directory
PROJECT_ROOT="/workspace/BioRiskEval_pLM"

# Auto-detect number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: No GPUs detected"
    exit 1
fi
echo "Detected $NUM_GPUS GPUs"

# Max parallel jobs per GPU
MAX_JOBS_PER_GPU=3

# List of all checkpoints from model_list
CHECKPOINTS=(
    "35M_C_P_10P"
    "35M_C_P_20P"
    "35M_C_P_30P"
    "35M_C_P_40P"
    "35M_C_P_60P"
    "35M_C_P_80P"
    "35M_I_U_10P"
    "35M_I_U_20P"
    "35M_I_U_30P"
    "35M_I_U_40P"
    "35M_I_U_60P"
    "35M_I_U_80P"
    "35M_R_10P"
    "35M_R_20P"
    "35M_R_30P"
    "35M_R_40P"
    "35M_R_60P"
    "35M_R_80P"
    "35M_C_P_10P_D2"
    "35M_C_P_20P_D2"
    "35M_C_P_30P_D2"
    "35M_C_P_40P_D2"
    "35M_C_P_80P_D2"
    "35M_I_U_10P_D2"
    "35M_I_U_20P_D2"
    "35M_I_U_30P_D2"
    "35M_I_U_40P_D2"
    "35M_I_U_60P_D2"
    "35M_I_U_80P_D2"
    "35M_R_10P_D2"
    "35M_R_20P_D2"
    "35M_R_30P_D2"
    "35M_R_40P_D2"
    "35M_R_60P_D2"
    "35M_R_80P_D2"
)

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

echo "Creating tmux session: $SESSION_NAME"
echo "Distributing ${#CHECKPOINTS[@]} checkpoints across $NUM_GPUS GPUs"
echo "Max $MAX_JOBS_PER_GPU parallel jobs per GPU"

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

    # Build command to run checkpoints for this GPU with parallelism
    cmd="source $VENV_PATH/bin/activate && "
    cmd+="cd $PROJECT_ROOT && "
    cmd+="export PYTHONPATH=$PROJECT_ROOT:\$PYTHONPATH; "
    cmd+="export CUDA_VISIBLE_DEVICES=$gpu_id; "
    # cmd+="export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache; "
    # cmd+="export TORCH_LOGS='+none'; "
    cmd+="export TORCH_COMPILE_DEBUG=0; "
    cmd+="export TORCHDYNAMO_VERBOSE=0; "
    cmd+="echo '=== GPU $gpu_id: Running ${#gpu_checkpoints[@]} checkpoints (max $MAX_JOBS_PER_GPU parallel) ==='; "

    # Create array of checkpoints in the command
    cmd+="checkpoints=(${gpu_checkpoints[*]}); "
    cmd+="total=\${#checkpoints[@]}; "
    cmd+="for ((i=0; i<total; i+=$MAX_JOBS_PER_GPU)); do "
    cmd+="  batch_end=\$((i + $MAX_JOBS_PER_GPU)); "
    cmd+="  if [ \$batch_end -gt \$total ]; then batch_end=\$total; fi; "
    cmd+="  echo \"--- GPU $gpu_id: Starting batch \$((i/$MAX_JOBS_PER_GPU + 1)) (checkpoints \$((i+1))-\$batch_end of \$total) ---\"; "
    cmd+="  pids=(); "
    cmd+="  for ((j=i; j<batch_end; j++)); do "
    cmd+="    ckpt=\${checkpoints[\$j]}; "
    cmd+="    echo \"GPU $gpu_id: Launching \$ckpt\"; "
    cmd+="    PYTHONPATH=$PROJECT_ROOT python bioriskeval/gen/eval_ppl_esm2.py "
    cmd+="--tier $TIER "
    cmd+="--model-name given131/\$ckpt "
    cmd+="--mask-chunk-size $MASK_CHUNK_SIZE & "
    cmd+="    pids+=(\$!); "
    cmd+="  done; "
    cmd+="  echo \"GPU $gpu_id: Waiting for batch to complete...\"; "
    cmd+="  for pid in \"\${pids[@]}\"; do wait \$pid; done; "
    cmd+="  echo \"--- GPU $gpu_id: Batch completed ---\"; "
    cmd+="done; "
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
