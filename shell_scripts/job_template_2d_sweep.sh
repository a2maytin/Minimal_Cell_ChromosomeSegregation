#!/bin/bash
#SBATCH --job-name={{RUN_NAME}}
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

# Base directory for protein_science project
BASE_DIR="/raid/amaytin/protein_science"

# Function to check GPU availability
check_gpu() {
    local gpu_id=$1
    nvidia-smi -i $gpu_id -q > /dev/null 2>&1
    return $?
}

# Function to wait for GPU to become available
wait_for_gpu() {
    local gpu_id=$1
    local max_attempts=10
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if check_gpu $gpu_id; then
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

# SLOT is 1-5 mapping to GPUs 3,4,5,6,7 within a batch
NV_GPU=$(({{SLOT}} + 2))

# Initialize log file
LOG_FILE="{{RUN_NAME}}.log"
echo "=== Job started: $(date) ===" > "$LOG_FILE"
echo "GPU ID: $NV_GPU" >> "$LOG_FILE"

# Wait for GPU to be available
if ! wait_for_gpu $NV_GPU; then
    echo "ERROR: GPU $NV_GPU failed health check after multiple attempts" >> "$LOG_FILE"
    exit 1
fi

# Retry logic for the actual simulation
MAX_RETRIES=10
RETRY_COUNT=0
EXIT_CODE=1
START_TIME={{START_TIME}}
END_TIME={{END_TIME}}
IS_RESTART={{IS_RESTART}}

# Check if we're actually restarting (files exist) and adjust IS_RESTART accordingly
# This ensures sc_chain_generation is skipped if we're restarting
RUN_NAME="{{RUN_NAME}}"
DATA_DIR="${BASE_DIR}/runs/${RUN_NAME}/data/coords"
if [ -d "$DATA_DIR" ]; then
    # Check if any DNA files exist
    if ls "${DATA_DIR}/dna_${RUN_NAME}_"*.bin 1> /dev/null 2>&1; then
        echo "Found existing DNA files - treating as restart, skipping sc_chain_generation" >> "$LOG_FILE"
        IS_RESTART="true"
    fi
fi

while [ $RETRY_COUNT -lt $MAX_RETRIES ] && [ $EXIT_CODE -ne 0 ]; do
    if [ $RETRY_COUNT -gt 0 ]; then
        echo "=== Retry attempt $RETRY_COUNT of $MAX_RETRIES after CUDA error ===" >> "$LOG_FILE"
        sleep $((RETRY_COUNT * 5))  # Wait before retry
        # Wait for GPU again before retry
        if ! wait_for_gpu $NV_GPU; then
            echo "ERROR: GPU $NV_GPU failed health check on retry" >> "$LOG_FILE"
            exit 1
        fi
    fi
    
    # Run the simulation (use >> to append to existing log)
    docker run \
        --gpus "\"device=$NV_GPU\"" \
        --name "single_run_${SLURM_JOB_ID}_${RETRY_COUNT}" \
        --rm \
        -v ${BASE_DIR}/runs/{{RUN_NAME}}:/mnt \
        protein_science \
        bash -c "cd /mnt/scripts/ && if [ '$IS_RESTART' != 'true' ]; then bash run_sc_chain_generation.sh; fi && python3 run_btree_chromo_replicate.py {{SEED}} {{RUN_NAME}} $START_TIME $END_TIME $IS_RESTART {{V}}" >> "$LOG_FILE" 2>&1
    
    EXIT_CODE=$?
    
    # Check if error is CUDA-related or GPU-related
    if [ $EXIT_CODE -ne 0 ]; then
        # Check for various CUDA/GPU errors that should trigger retry
        # Patterns include: CUDA errors, GPU device errors, Kokkos CUDA errors, and SIGABRT from CUDA
        CUDA_ERROR_PATTERNS="cudaError|cudaErrorNoDevice|cudaErrorIllegalAddress|cudaDeviceSynchronize.*error|no CUDA-capable device|Kokkos.*Cuda|Kokkos_Cuda"
        if grep -qiE "$CUDA_ERROR_PATTERNS" "$LOG_FILE" 2>/dev/null || \
           (grep -q "SIGABRT\|Signal: Aborted" "$LOG_FILE" 2>/dev/null && \
            grep -q "Kokkos\|Cuda\|CUDA" "$LOG_FILE" 2>/dev/null); then
            echo "CUDA/GPU error detected, will retry..." >> "$LOG_FILE"
            
            # Find the last successfully completed timestep by checking for output files
            # File naming: dna_{run_name}_{N}.bin is written at the END of timestep N-1
            # If dna_15.bin exists but dna_16.bin does not:
            #   - Timestep 14 completed (wrote dna_15.bin)
            #   - Timestep 15 did NOT complete (didn't write dna_16.bin)
            #   - We can restart from timestep 15 (the one that failed)
            RUN_NAME="{{RUN_NAME}}"
            DATA_DIR="${BASE_DIR}/runs/${RUN_NAME}/data/coords"
            
            # Find the highest timestep N where dna_{run_name}_{N}.bin exists
            # This is the last available output file, so we can restart from timestep N
            LAST_AVAILABLE=-1
            for timestep in $(seq 0 200); do
                if [ -f "${DATA_DIR}/dna_${RUN_NAME}_${timestep}.bin" ]; then
                    LAST_AVAILABLE=$timestep
                else
                    break
                fi
            done
            
            if [ $LAST_AVAILABLE -ge 0 ]; then
                # Restart from the timestep corresponding to the last available file
                # If dna_15.bin exists (but not dna_16.bin), restart from timestep 15
                NEW_START_TIME=$LAST_AVAILABLE
                echo "Last available DNA file: dna_${RUN_NAME}_${LAST_AVAILABLE}.bin, resuming from timestep: $NEW_START_TIME" >> "$LOG_FILE"
                START_TIME=$NEW_START_TIME
                IS_RESTART="true"
            else
                # No output files found, retry from original start_time
                echo "No DNA output files found, retrying from original start_time: {{START_TIME}}" >> "$LOG_FILE"
                START_TIME={{START_TIME}}
                IS_RESTART={{IS_RESTART}}
            fi
            
            RETRY_COUNT=$((RETRY_COUNT + 1))
        else
            # Non-CUDA error, don't retry
            echo "Non-CUDA error detected, exiting without retry" >> "$LOG_FILE"
            break
        fi
    else
        # Success
        echo "Job completed successfully at $(date)" >> "$LOG_FILE"
        break
    fi
done

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Job failed after $MAX_RETRIES retry attempts at $(date)" >> "$LOG_FILE"
    exit $EXIT_CODE
fi
