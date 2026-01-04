#!/bin/bash

# Tmux script to run fitness evaluation across multiple GPUs
# Each GPU runs up to 3 jobs in parallel

# Usage: ./run_fitness_eval_parallel.sh [model-name] [csv-path]
# 
# Arguments:
#   model-name: Optional. Specific model to evaluate (e.g., "35M_C_P_10P"). If not provided, runs all models.
#   csv-path: Optional. Path to DMS CSV file. Default: bioriskeval/mut/data/DMS_ProteinGym_substitutions/DMS_substitutions.csv
#
# Examples:
#   ./run_fitness_eval_parallel.sh                                    # Run all models with default CSV
#   ./run_fitness_eval_parallel.sh "" custom_data.csv                 # Run all models with custom CSV
#   ./run_fitness_eval_parallel.sh 35M_C_P_10P                       # Run specific model with default CSV
#   ./run_fitness_eval_parallel.sh 35M_C_P_10P custom_data.csv      # Run specific model with custom CSV

SESSION_NAME="esm2-fitness-eval"

# Parse arguments
SPECIFIC_MODEL="${1:-}"
CSV_PATH="/workspaces/BioRiskEval/merged_sample.csv"

# Default parameters (optimized for fitness evaluation)
BATCH_SIZE="1024"
MASK_CHUNK_SIZE="512"
NUM_WORKERS="4"

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

# List of all checkpoints (same as gen/run_model_list_parallel.sh)
ALL_CHECKPOINTS=(
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

# Filter checkpoints based on specific model argument
if [ -n "$SPECIFIC_MODEL" ]; then
    # Check if the specific model exists in the list
    if [[ " ${ALL_CHECKPOINTS[@]} " =~ " ${SPECIFIC_MODEL} " ]]; then
        CHECKPOINTS=("$SPECIFIC_MODEL")
        echo "Running specific model: $SPECIFIC_MODEL"
    else
        echo "Error: Model '$SPECIFIC_MODEL' not found in checkpoint list"
        echo "Available models:"
        printf '  %s\n' "${ALL_CHECKPOINTS[@]}"
        exit 1
    fi
else
    CHECKPOINTS=("${ALL_CHECKPOINTS[@]}")
    echo "Running all ${#CHECKPOINTS[@]} models"
fi

# Verify CSV path exists
if [ ! -f "$CSV_PATH" ]; then
    echo "Warning: CSV file not found at: $CSV_PATH"
    echo "Make sure the file exists before the evaluation starts."
fi

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

echo "=" | head -c 60; echo
echo "ESM2 Fitness Evaluation - Parallel Execution"
echo "=" | head -c 60; echo
echo "Session: $SESSION_NAME"
echo "CSV Path: $CSV_PATH"
echo "Models to evaluate: ${#CHECKPOINTS[@]}"
echo "GPUs: $NUM_GPUS"
echo "Max parallel jobs per GPU: $MAX_JOBS_PER_GPU"
echo "Batch size: $BATCH_SIZE"
echo "Mask chunk size: $MASK_CHUNK_SIZE"
echo "DataLoader workers: $NUM_WORKERS"
echo "=" | head -c 60; echo

# Create new tmux session
tmux new-session -d -s $SESSION_NAME -n "fitness-eval" "bash -i"

# Distribute checkpoints across GPUs
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    # Get checkpoints for this GPU
    gpu_checkpoints=()
    for i in "${!CHECKPOINTS[@]}"; do
        if [ $((i % NUM_GPUS)) -eq $gpu_id ]; then
            gpu_checkpoints+=("${CHECKPOINTS[$i]}")
        fi
    done

    # Skip if no checkpoints for this GPU
    if [ ${#gpu_checkpoints[@]} -eq 0 ]; then
        continue
    fi

    # Build command to run checkpoints for this GPU with parallelism
    cmd="source $VENV_PATH/bin/activate && "
    cmd+="cd $PROJECT_ROOT && "
    cmd+="export PYTHONPATH=$PROJECT_ROOT:\$PYTHONPATH; "
    cmd+="export CUDA_VISIBLE_DEVICES=$gpu_id; "
    cmd+="export TORCH_COMPILE_DEBUG=0; "
    cmd+="export TORCHDYNAMO_VERBOSE=0; "
    cmd+="export TOKENIZERS_PARALLELISM=true; "
    cmd+="echo '=== GPU $gpu_id: Running ${#gpu_checkpoints[@]} models (max $MAX_JOBS_PER_GPU parallel) ==='; "

    # Create array of checkpoints in the command
    cmd+="checkpoints=(${gpu_checkpoints[*]}); "
    cmd+="total=\${#checkpoints[@]}; "
    cmd+="for ((i=0; i<total; i+=$MAX_JOBS_PER_GPU)); do "
    cmd+="  batch_end=\$((i + $MAX_JOBS_PER_GPU)); "
    cmd+="  if [ \$batch_end -gt \$total ]; then batch_end=\$total; fi; "
    cmd+="  echo \"--- GPU $gpu_id: Starting batch \$((i/$MAX_JOBS_PER_GPU + 1)) (models \$((i+1))-\$batch_end of \$total) ---\"; "
    cmd+="  pids=(); "
    cmd+="  for ((j=i; j<batch_end; j++)); do "
    cmd+="    ckpt=\${checkpoints[\$j]}; "
    cmd+="    echo \"GPU $gpu_id: Launching \$ckpt\"; "
    cmd+="    PYTHONPATH=$PROJECT_ROOT python bioriskeval/mut/eval_fitness_esm2_hf.py "
    cmd+="--model-name given131/\$ckpt "
    cmd+="--csv-path $CSV_PATH "
    cmd+="--batch-size $BATCH_SIZE "
    cmd+="--mask-chunk-size $MASK_CHUNK_SIZE "
    cmd+="--num-workers $NUM_WORKERS "
    cmd+="--use-compile "
    cmd+="--use-flash-attn & "
    cmd+="    pids+=(\$!); "
    cmd+="  done; "
    cmd+="  echo \"GPU $gpu_id: Waiting for batch to complete...\"; "
    cmd+="  for pid in \"\${pids[@]}\"; do wait \$pid; done; "
    cmd+="  echo \"--- GPU $gpu_id: Batch completed ---\"; "
    cmd+="done; "
    cmd+="echo '=== GPU $gpu_id: All models completed ==='; "
    cmd+="echo '=== GPU $gpu_id: All models completed ==='; exec bash"

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
echo "All models distributed and started!"
echo "Layout: $NUM_GPUS panes (one per GPU)"
echo ""
echo "Attaching to tmux session..."
echo "To detach: Press Ctrl+B then D"
echo "To reattach: tmux attach-session -t $SESSION_NAME"
tmux attach-session -t $SESSION_NAME

