#!/bin/bash
#
# 3D parameter sweep over N, v, and tau (matches analysis/runs.ipynb).
# S = N * v * tau. Parameters are specified as (N, v, tau); we derive:
#   BASAL_DEATH_PROB = 20/(tau*v)   [e.g. tau=30s, v=500 -> 1/(30*25)=0.00133, 25=v/20]
#   STALL_DEATH_PROB = 400/(v^2)    [e.g. v=500 -> 1/(25*25)=0.0016]
#   NUM_SMC = N,  V = v (translocation speed in bp/s)
#
# Submits in batches of 5 (5 GPUs at a time). One run per (N,v,tau).
#

set -e

BASE_DIR="/raid/amaytin/protein_science"
cd "${BASE_DIR}/shell_scripts"

DATE=Mar03

# --- Parameter grid (must match analysis/runs.ipynb) ---
N_values=(20 35 50)
v_values=(200 350 500)
tau_values=(50 75 100)
# Total runs = 3 × 3 × 3 = 27 (product order: N outer, v middle, tau inner)

# Fixed
SEED=69
BYPASS=1.0
KNOCKOFF=$(awk "BEGIN {printf \"%.6f\", 1-${BYPASS}}")
START_TIME=0
END_TIME=80
IS_RESTART="false"

BATCH_SIZE=5
LEN_N=${#N_values[@]}
LEN_V=${#v_values[@]}
LEN_TAU=${#tau_values[@]}
NUM_PARAM_SETS=$((LEN_N * LEN_V * LEN_TAU))

echo "3D sweep (N, v, tau): ${LEN_N}×${LEN_V}×${LEN_TAU} = ${NUM_PARAM_SETS} runs"
echo "  BASAL_DEATH_PROB = 20/(tau*v), STALL_DEATH_PROB = 400/v^2, NUM_SMC = N, V = v"
echo ""

DEPENDENCY=""
for ((batch=0; batch*BATCH_SIZE < NUM_PARAM_SETS; batch++)); do
  BATCH_JOB_IDS=()

  for ((slot=0; slot<BATCH_SIZE; slot++)); do
    PARAM_IDX=$((batch * BATCH_SIZE + slot + 1))
    if [ "$PARAM_IDX" -gt "$NUM_PARAM_SETS" ]; then
      break
    fi

    # Map param index to (i_n, i_v, i_tau) matching itertools.product(N_values, v_values, tau_values)
    k=$((PARAM_IDX - 1))
    i_tau=$((k % LEN_TAU))
    k=$((k / LEN_TAU))
    i_v=$((k % LEN_V))
    i_n=$((k / LEN_V))
    N=${N_values[i_n]}
    v=${v_values[i_v]}
    tau=${tau_values[i_tau]}

    # Derive simulation parameters from (N, v, tau)
    BASAL_DEATH_PROB=$(awk "BEGIN {printf \"%.6f\", 20/(${tau}*${v})}")
    STALL_DEATH_PROB=$(awk "BEGIN {printf \"%.6f\", 20/(25*${v})}")
    SMC=$N
    V=$v

    RUN_NAME="${DATE}_p${PARAM_IDX}"
    RUN_DIR="${BASE_DIR}/runs/${RUN_NAME}"
    JOB_FILE="job_${RUN_NAME}.sh"

    mkdir -p "${RUN_DIR}/scripts"
    mkdir -p "${RUN_DIR}/LAMMPS_DNA_model_kk"
    mkdir -p "${RUN_DIR}/data/coords" "${RUN_DIR}/data/loops" "${RUN_DIR}/data/rep_states"
    if [ ! -f "${RUN_DIR}/scripts/loop_params.txt" ]; then
      cp -r "${BASE_DIR}/scripts/"* "${RUN_DIR}/scripts/"
      cp -r "${BASE_DIR}/LAMMPS_DNA_model_kk/"* "${RUN_DIR}/LAMMPS_DNA_model_kk/"
    fi

    REPL_SUFFIX="p${PARAM_IDX}"
    sed "s/{DATE}/${DATE}/g; s/{REPL}/${REPL_SUFFIX}/g" "${RUN_DIR}/scripts/run_sc_chain_generation_template.sh" > "${RUN_DIR}/scripts/run_sc_chain_generation.sh"
    sed "s/{SEED}/${SEED}/g" "${RUN_DIR}/scripts/Syn3A_chromosome_init_template.inp" > "${RUN_DIR}/scripts/Syn3A_chromosome_init.inp"

    sed -i "s/^basal_death_prob=.*/basal_death_prob=${BASAL_DEATH_PROB}/g; s/^stall_death_prob=.*/stall_death_prob=${STALL_DEATH_PROB}/g; s/^numSmc=.*/numSmc=${SMC}/g; s/^bypass=.*/bypass=${BYPASS}/g; s/^knockoff=.*/knockoff=${KNOCKOFF}/g" "${RUN_DIR}/scripts/loop_params.txt"

    SLOT=$((slot + 1))
    sed "s/{{RUN_NAME}}/${RUN_NAME}/g; s/{{SLOT}}/${SLOT}/g; s/{{SEED}}/${SEED}/g; s/{{START_TIME}}/${START_TIME}/g; s/{{END_TIME}}/${END_TIME}/g; s/{{IS_RESTART}}/${IS_RESTART}/g; s/{{V}}/${V}/g" job_template_2d_sweep.sh > "${JOB_FILE}"

    if [ -n "$DEPENDENCY" ]; then
      SUBMIT_OUT=$(sbatch --dependency="$DEPENDENCY" "$JOB_FILE")
    else
      SUBMIT_OUT=$(sbatch "$JOB_FILE")
    fi
    JID=$(echo "$SUBMIT_OUT" | awk '{print $NF}')
    BATCH_JOB_IDS+=("$JID")

    S=$((N * v * tau))
    echo "  Submitted ${JOB_FILE} (p=${PARAM_IDX}, N=${N}, v=${v}, tau=${tau}, S=${S}, basal=${BASAL_DEATH_PROB}, stall=${STALL_DEATH_PROB}, slot=${SLOT}) -> job ${JID}"
  done

  DEPENDENCY="afterok:$(IFS=:; echo "${BATCH_JOB_IDS[*]}")"
done

echo ""
echo "All ${NUM_PARAM_SETS} jobs submitted (batches of ${BATCH_SIZE}). Run names: ${DATE}_p1 .. ${DATE}_p${NUM_PARAM_SETS}."
