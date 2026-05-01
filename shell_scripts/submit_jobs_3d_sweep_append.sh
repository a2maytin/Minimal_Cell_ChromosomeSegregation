#!/bin/bash
#
# Append runs 49-60 to the Feb16 3D sweep: N=50 only, same v and tau grid.
# Run names: Feb16_p49 .. Feb16_p60. Submit to squeue like the main sweep.
# (Main sweep has runs 1-48: N in 20,25,30,35 × v × tau.)
#
# To run only after the first 48 jobs have finished, set LAST_JOB_ID to the
# job ID of run 48 (Feb16_p48). Example: LAST_JOB_ID=744 ./submit_jobs_3d_sweep_append.sh
#
LAST_JOB_ID=737
set -e

BASE_DIR="/raid/amaytin/protein_science"
cd "${BASE_DIR}/shell_scripts"

DATE=Feb16
START_RUN_ID=49
NUM_APPEND_RUNS=12   # runs 49..60

# --- Append grid: N=50 only, same v and tau as main sweep ---
N_values=(50)
v_values=(200 350 500)
tau_values=(45 65 85 105)
# 1 × 3 × 4 = 12 runs

# Fixed (match main sweep)
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

if [ "$NUM_PARAM_SETS" -ne "$NUM_APPEND_RUNS" ]; then
  echo "Mismatch: grid gives ${NUM_PARAM_SETS} runs, expected ${NUM_APPEND_RUNS}. Aborting."
  exit 1
fi

echo "Append 3D sweep: runs ${START_RUN_ID}..$((START_RUN_ID + NUM_PARAM_SETS - 1)) (N=50, same v/tau)"
echo "  ${LEN_N}×${LEN_V}×${LEN_TAU} = ${NUM_PARAM_SETS} runs"
if [ -n "${LAST_JOB_ID:-}" ]; then
  echo "  Dependency: after run 48 (job ${LAST_JOB_ID}) finishes"
fi
echo ""

# First batch waits for run 48 if LAST_JOB_ID is set
DEPENDENCY=""
if [ -n "${LAST_JOB_ID:-}" ]; then
  DEPENDENCY="afterok:${LAST_JOB_ID}"
fi
for ((batch=0; batch*BATCH_SIZE < NUM_PARAM_SETS; batch++)); do
  BATCH_JOB_IDS=()

  for ((slot=0; slot<BATCH_SIZE; slot++)); do
    LOCAL_IDX=$((batch * BATCH_SIZE + slot))
    if [ "$LOCAL_IDX" -ge "$NUM_PARAM_SETS" ]; then
      break
    fi
    PARAM_IDX=$((START_RUN_ID + LOCAL_IDX))

    # Map local index to (i_n, i_v, i_tau): same product order as main script
    k=$LOCAL_IDX
    i_tau=$((k % LEN_TAU))
    k=$((k / LEN_TAU))
    i_v=$((k % LEN_V))
    i_n=$((k / LEN_V))
    N=${N_values[i_n]}
    v=${v_values[i_v]}
    tau=${tau_values[i_tau]}

    BASAL_DEATH_PROB=$(awk "BEGIN {printf \"%.6f\", 20/(${tau}*${v})}")
    STALL_DEATH_PROB=$(awk "BEGIN {printf \"%.6f\", 400/(${v}*${v})}")
    SMC=$N
    V=$v

    RUN_NAME="${DATE}_p${PARAM_IDX}"
    RUN_DIR="${BASE_DIR}/runs/${RUN_NAME}"
    JOB_FILE="job_${RUN_NAME}.sh"

    mkdir -p "${RUN_DIR}/scripts"
    mkdir -p "${RUN_DIR}/data/coords" "${RUN_DIR}/data/loops" "${RUN_DIR}/data/rep_states"
    if [ ! -f "${RUN_DIR}/scripts/loop_params.txt" ]; then
      cp -r "${BASE_DIR}/scripts/"* "${RUN_DIR}/scripts/"
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
    echo "  Submitted ${JOB_FILE} (p=${PARAM_IDX}, N=${N}, v=${v}, tau=${tau}, S=${S}, slot=${SLOT}) -> job ${JID}"
  done

  DEPENDENCY="afterok:$(IFS=:; echo "${BATCH_JOB_IDS[*]}")"
done

echo ""
echo "Append jobs submitted: ${DATE}_p${START_RUN_ID} .. ${DATE}_p$((START_RUN_ID + NUM_PARAM_SETS - 1))."
