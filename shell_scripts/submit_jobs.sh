#!/bin/bash
#
# Submit one job per replicate. For each run you can specify N, v, tau (like 3D sweep);
# basal_death_prob = 20/(tau*v), stall_death_prob = 20/(25*v), numSmc = N, V = v.
# Optionally set GPU_IDS to use specific GPUs (default: 1 2 for two jobs).
#

set -e

# Base directory for protein_science project
BASE_DIR="/raid/amaytin/protein_science"
cd "${BASE_DIR}/shell_scripts"

DATE=Apr06
# Number of replicates (jobs to submit)
NUM_REPLICATES=1

# GPU IDs to use (one per replicate; e.g. 1 2 to use GPUs 1 and 2 instead of 0)
GPU_IDS=(7)

# List of random seeds (must be at least NUM_REPLICATES long)
SEEDS=(69)

# Per-run N, v, tau (same formulas as submit_jobs_3d_sweep: basal=20/(tau*v), stall=20/(25*v), numSmc=N, V=v)
# One entry per replicate
N_values=(20)
v_values=(500)
tau_values=(100)

# List of bypass probabilities (bypass parameter, knockoff = 1-bypass; must be at least NUM_REPLICATES long)
BYPASS_PROBS=(1.0)

# Simulation parameters
# START_TIME: Timepoint to start from (0 for new runs, or restart point for restarts)
# END_TIME: Target end timepoint (default 90)
# IS_RESTART: "true" to skip chain generation and restart from START_TIME, "false" for new runs
START_TIME=0
END_TIME=80
IS_RESTART="false"

# Loop over replicates
for ((i=0; i<NUM_REPLICATES; i++)); do
    REPL=$((i + 1))
    SEED=${SEEDS[i]}
    N=${N_values[i]}
    v=${v_values[i]}
    tau=${tau_values[i]}
    BASAL_DEATH_PROB=$(awk "BEGIN {printf \"%.6f\", 20/(${tau}*${v})}")
    # STALL_DEATH_PROB=$(awk "BEGIN {printf \"%.6f\", 20/(25*${v})}")
    STALL_DEATH_PROB=$(awk "BEGIN {printf \"%.6f\", 20/(${tau}*${v})}")
    SMC=$N
    V=$v
    BYPASS=${BYPASS_PROBS[i]}
    KNOCKOFF=$(awk "BEGIN {printf \"%.6f\", 1-${BYPASS}}")
    GPU_ID=${GPU_IDS[i]}
    JOB_FILE="job_${DATE}_${REPL}.sh"

    # Create run directory and copy scripts + DNA model (same as submit_jobs_3d_sweep)
    RUN_DIR="${BASE_DIR}/runs/${DATE}_${REPL}"
    mkdir -p "${RUN_DIR}/scripts"
    mkdir -p "${RUN_DIR}/LAMMPS_DNA_model_kk"
    mkdir -p "${RUN_DIR}/data/coords" "${RUN_DIR}/data/loops" "${RUN_DIR}/data/rep_states"
    if [ ! -f "${RUN_DIR}/scripts/loop_params.txt" ]; then
        cp -r "${BASE_DIR}/scripts/"* "${RUN_DIR}/scripts/"
        cp -r "${BASE_DIR}/LAMMPS_DNA_model_kk/"* "${RUN_DIR}/LAMMPS_DNA_model_kk/"
    fi

    # Replace placeholders in the template (including GPU_ID and V)
    sed "s/{{DATE}}/${DATE}/g; s/{{REPL}}/${REPL}/g; s/{{SEED}}/${SEED}/g; s/{{START_TIME}}/${START_TIME}/g; s/{{END_TIME}}/${END_TIME}/g; s/{{IS_RESTART}}/${IS_RESTART}/g; s/{{V}}/${V}/g; s/{{GPU_ID}}/${GPU_ID}/g" job_template.sh > "$JOB_FILE"
    sed "s/{SEED}/${SEED}/g" ${RUN_DIR}/scripts/Syn3A_chromosome_init_template.inp > ${RUN_DIR}/scripts/Syn3A_chromosome_init.inp
    sed "s/{DATE}/${DATE}/g; s/{REPL}/${REPL}/g" ${RUN_DIR}/scripts/run_sc_chain_generation_template.sh > ${RUN_DIR}/scripts/run_sc_chain_generation.sh

    # Replace death probabilities, numSmc, bypass, and knockoff in loop_params.txt
    sed -i "s/^basal_death_prob=.*/basal_death_prob=${BASAL_DEATH_PROB}/g; s/^stall_death_prob=.*/stall_death_prob=${STALL_DEATH_PROB}/g; s/^numSmc=.*/numSmc=${SMC}/g; s/^bypass=.*/bypass=${BYPASS}/g; s/^knockoff=.*/knockoff=${KNOCKOFF}/g" ${RUN_DIR}/scripts/loop_params.txt

    # Submit the job
    sbatch "$JOB_FILE"

    echo "Submitted job: $JOB_FILE (Replicate: $REPL, GPU: ${GPU_ID}, N=${N}, v=${v}, tau=${tau}, basal=${BASAL_DEATH_PROB}, stall=${STALL_DEATH_PROB}, numSmc=${SMC}, Seed: ${SEED}, Start: ${START_TIME}, End: ${END_TIME})"
done
