#!/bin/bash
#SBATCH --job-name={{RUN_NAME}}
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

# Base directory for protein_science project
BASE_DIR="/raid/amaytin/protein_science"

# Function to check GPU availability
check_gpu() {
    local gpu_id=$1
    nvidia-smi -i "$gpu_id" -q > /dev/null 2>&1
    return $?
}

# Function to wait for GPU to become available
wait_for_gpu() {
    local gpu_id=$1
    local max_attempts=10
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if check_gpu "$gpu_id"; then
            # Also check CUDA device count via docker
            if docker run --rm --gpus "\"device=$gpu_id\"" protein_science nvidia-smi -L > /dev/null 2>&1; then
                return 0
            fi
        fi
        echo "GPU $gpu_id not available, waiting... (attempt $((attempt+1))/$max_attempts)" >> "$LOG_FILE"
        sleep $((attempt + 1))  # Exponential backoff
        attempt=$((attempt + 1))
    done
    return 1
}

# Run the job inside the docker container (GPU ID set by submit script)
NV_GPU={{GPU_ID}}

# Initialize log file
RUN_NAME="{{RUN_NAME}}"
LOG_FILE="${RUN_NAME}.log"
echo "=== Stationary job started: $(date) ===" > "$LOG_FILE"
echo "GPU ID: $NV_GPU" >> "$LOG_FILE"

# Wait for GPU to be available
if ! wait_for_gpu "$NV_GPU"; then
    echo "ERROR: GPU $NV_GPU failed health check after multiple attempts" >> "$LOG_FILE"
    exit 1
fi

# Retry logic for the actual simulation
MAX_RETRIES=5
RETRY_COUNT=0
EXIT_CODE=1

while [ $RETRY_COUNT -lt $MAX_RETRIES ] && [ $EXIT_CODE -ne 0 ]; do
    if [ $RETRY_COUNT -gt 0 ]; then
        echo "=== Retry attempt $RETRY_COUNT of $MAX_RETRIES after CUDA error ===" >> "$LOG_FILE"
        sleep $((RETRY_COUNT * 5))  # Wait before retry
        # Wait for GPU again before retry
        if ! wait_for_gpu "$NV_GPU"; then
            echo "ERROR: GPU $NV_GPU failed health check on retry" >> "$LOG_FILE"
            exit 1
        fi
    fi

    # Run the stationary simulation
    docker run \
        --gpus "\"device=$NV_GPU\"" \
        --name "stationary_${SLURM_JOB_ID}_${RETRY_COUNT}" \
        --rm \
        -v "${BASE_DIR}/runs/${RUN_NAME}:/mnt" \
        protein_science \
        bash -c "cd /mnt/scripts/ && if [ ! -f \"../data/coords/dna_${RUN_NAME}_0.bin\" ]; then bash run_sc_chain_generation.sh; fi && python3 run_btree_chromo_stationary.py {{SEED}} ${RUN_NAME}" >> "$LOG_FILE" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        CUDA_ERROR_PATTERNS="cudaError|cudaErrorNoDevice|cudaErrorIllegalAddress|cudaDeviceSynchronize.*error|no CUDA-capable device|Kokkos.*Cuda|Kokkos_Cuda"
        if grep -qiE "$CUDA_ERROR_PATTERNS" "$LOG_FILE" 2>/dev/null || \
           (grep -q "SIGABRT\|Signal: Aborted" "$LOG_FILE" 2>/dev/null && \
            grep -q "Kokkos\|Cuda\|CUDA" "$LOG_FILE" 2>/dev/null); then
            echo "CUDA/GPU error detected, will retry..." >> "$LOG_FILE"
            RETRY_COUNT=$((RETRY_COUNT + 1))
        else
            echo "Non-CUDA error detected, exiting without retry" >> "$LOG_FILE"
            break
        fi
    else
        echo "Stationary job completed successfully at $(date)" >> "$LOG_FILE"
        break
    fi
done

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Stationary job failed after $MAX_RETRIES retry attempts at $(date)" >> "$LOG_FILE"
    exit $EXIT_CODE
fi

