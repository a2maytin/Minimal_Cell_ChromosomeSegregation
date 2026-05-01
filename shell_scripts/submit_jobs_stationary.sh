#!/bin/bash
#
# Submit batches of stationary jobs using run_btree_chromo_stationary.py.
# We launch 4 sets of 10 runs (40 total), with each set sharing parameters
# but using different RNG seeds. Jobs are pinned to GPUs 3–7 and at most
# 5 jobs are kept in the Slurm queue at any given time.
#

set -e

# Base directory for protein_science project
BASE_DIR="/raid/amaytin/protein_science"
cd "${BASE_DIR}/shell_scripts"

# Label for this batch of runs (used in run names and job names)
DATE=Mar10

# Stationary run layout
NUM_SETS=3
RUNS_PER_SET=5

# Per-set parameters (one value per set: 4 entries each)
# Set 1 uses index 0, set 2 index 1, etc.
N_values=(20 50 100)
v_values=(500 500 500)
tau_values=(100 100 4)
BYPASS_VALUES=(1.0 0.002 1.0)

# GPUs to use (cycled across jobs)
GPU_IDS=(3 4 5 6 7)
NUM_GPUS=${#GPU_IDS[@]}

# Maximum number of jobs (for this DATE tag) allowed in the queue at once
MAX_IN_QUEUE=5

# Base seed; each job gets a unique seed derived from this
BASE_SEED=1000

job_index=0

for ((set_idx=1; set_idx<=NUM_SETS; set_idx++)); do
    for ((run_idx=1; run_idx<=RUNS_PER_SET; run_idx++)); do
        RUN_NAME="${DATE}_s${set_idx}_r${run_idx}"
        JOB_FILE="job_${RUN_NAME}.sh"

        # Unique seed per job: BASE_SEED + linear index
        SEED=$((BASE_SEED + job_index))

        # Cycle through available GPUs 3–7
        gpu_slot=$((job_index % NUM_GPUS))
        GPU_ID=${GPU_IDS[$gpu_slot]}

        # Create run directory and copy scripts + DNA model if not already present
        RUN_DIR="${BASE_DIR}/runs/${RUN_NAME}"
        mkdir -p "${RUN_DIR}/scripts"
        mkdir -p "${RUN_DIR}/LAMMPS_DNA_model_kk"
        mkdir -p "${RUN_DIR}/data/coords" "${RUN_DIR}/data/loops" "${RUN_DIR}/data/rep_states"
        if [ ! -f "${RUN_DIR}/scripts/template_stationary.inp" ]; then
            cp -r "${BASE_DIR}/scripts/"* "${RUN_DIR}/scripts/"
            cp -r "${BASE_DIR}/LAMMPS_DNA_model_kk/"* "${RUN_DIR}/LAMMPS_DNA_model_kk/"
        fi

        # Generate sc-chain initialization scripts/inputs for this run
        REPL_SUFFIX="s${set_idx}_r${run_idx}"
        if [ -f "${RUN_DIR}/scripts/run_sc_chain_generation_template.sh" ]; then
            sed "s/{DATE}/${DATE}/g; s/{REPL}/${REPL_SUFFIX}/g" \
                "${RUN_DIR}/scripts/run_sc_chain_generation_template.sh" > "${RUN_DIR}/scripts/run_sc_chain_generation.sh"
        fi
        if [ -f "${RUN_DIR}/scripts/Syn3A_chromosome_init_template.inp" ]; then
            sed "s/{SEED}/${SEED}/g" \
                "${RUN_DIR}/scripts/Syn3A_chromosome_init_template.inp" > "${RUN_DIR}/scripts/Syn3A_chromosome_init.inp"
        fi

        # Choose parameter set for this set index (same params for all 10 runs in a set)
        set_array_idx=$((set_idx - 1))
        N=${N_values[set_array_idx]}
        v=${v_values[set_array_idx]}
        tau=${tau_values[set_array_idx]}
        BYPASS=${BYPASS_VALUES[set_array_idx]}

        BASAL_DEATH_PROB=$(awk "BEGIN {printf \"%.6f\", 20/(${tau}*${v})}")
        STALL_DEATH_PROB=$(awk "BEGIN {printf \"%.6f\", 20/(${tau}*${v})}")
        SMC=$N
        KNOCKOFF=$(awk "BEGIN {printf \"%.6f\", 1-${BYPASS}}")

        # Update loop_params.txt in this run directory with per-set parameters
        if [ -f "${RUN_DIR}/scripts/loop_params.txt" ]; then
            sed -i "s/^basal_death_prob=.*/basal_death_prob=${BASAL_DEATH_PROB}/g; s/^stall_death_prob=.*/stall_death_prob=${STALL_DEATH_PROB}/g; s/^numSmc=.*/numSmc=${SMC}/g; s/^bypass=.*/bypass=${BYPASS}/g; s/^knockoff=.*/knockoff=${KNOCKOFF}/g" "${RUN_DIR}/scripts/loop_params.txt"
        fi

        # Render the stationary job script (inject run name, seed, and GPU)
        sed "s/{{RUN_NAME}}/${RUN_NAME}/g; s/{{SEED}}/${SEED}/g; s/{{GPU_ID}}/${GPU_ID}/g" \
            job_template_stationary.sh > "$JOB_FILE"

        # Throttle submissions: keep at most MAX_IN_QUEUE jobs with this DATE tag
        while true; do
            # Count jobs for this user whose names contain the DATE tag
            NUM_JOBS=$(squeue -u "$USER" 2>/dev/null | awk 'NR>1 {print $3}' | grep -c "${DATE}_s" || true)
            if [ "$NUM_JOBS" -lt "$MAX_IN_QUEUE" ]; then
                break
            fi
            echo "Queue has ${NUM_JOBS} stationary jobs (>= ${MAX_IN_QUEUE}); waiting before submitting ${RUN_NAME}..."
            sleep 10
        done

        # Submit the job
        sbatch "$JOB_FILE"

        echo "Submitted stationary job: $JOB_FILE (Run: ${RUN_NAME}, GPU: ${GPU_ID}, Seed: ${SEED})"

        job_index=$((job_index + 1))
    done
done