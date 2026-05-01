#!/bin/bash
#
# 2D parameter sweep over BASAL_DEATH_PROB and NUM_SMC.
# 5 x 5 = 25 parameter sets, 3 replicates each = 75 runs.
# Submits in batches of 5 (5 GPUs at a time).
# Order: complete all 25 params for replicate 1, then replicate 2, then replicate 3,
# so the 2D parameter space can be analyzed after each replicate set.
#

set -e

BASE_DIR="/raid/amaytin/protein_science"
cd "${BASE_DIR}/shell_scripts"

DATE=Feb15

# 2D sweep: 5 values of BASAL_DEATH_PROB, 5 values of NUM_SMC -> 25 parameter sets
BASAL_DEATH_PROBS=(0.00133 0.001 0.0008 0.000666 0.000571)
NUM_SMC=(10 15 20 25 30)

# Replicates per parameter set
NUM_REPLICATES=3
SEEDS=(69 70 71)

# Fixed parameters (same for all runs)
STALL_DEATH_PROB=0.0016
BYPASS=1.0
KNOCKOFF=$(awk "BEGIN {printf \"%.6f\", 1-${BYPASS}}")
V=200
START_TIME=0
END_TIME=80
IS_RESTART="false"

BATCH_SIZE=5
NUM_PARAM_SETS=$((${#BASAL_DEATH_PROBS[@]} * ${#NUM_SMC[@]}))

echo "2D sweep: ${#BASAL_DEATH_PROBS[@]} BASAL_DEATH_PROB x ${#NUM_SMC[@]} NUM_SMC = ${NUM_PARAM_SETS} param sets, ${NUM_REPLICATES} replicates = $((NUM_PARAM_SETS * NUM_REPLICATES)) total jobs"
echo "Batches of ${BATCH_SIZE} jobs. Order: all ${NUM_PARAM_SETS} params for rep 1, then rep 2, then rep 3."
echo ""

# Submit in order: for each replicate, run 5 batches of 5 jobs; next batch depends on previous batch.
# Do NOT reset DEPENDENCY between replicates: rep 2's first batch must wait for rep 1's last batch,
# so only 5 jobs (one batch) run at a time and no two jobs share a GPU.
DEPENDENCY=""
for ((rep_idx=0; rep_idx<NUM_REPLICATES; rep_idx++)); do
  REPL=$((rep_idx + 1))
  SEED=${SEEDS[rep_idx]}

  for ((batch=0; batch*BATCH_SIZE < NUM_PARAM_SETS; batch++)); do
    BATCH_JOB_IDS=()

    for ((slot=0; slot<BATCH_SIZE; slot++)); do
      PARAM_IDX=$((batch * BATCH_SIZE + slot + 1))
      if [ "$PARAM_IDX" -gt "$NUM_PARAM_SETS" ]; then
        break
      fi

      # Map param index to (i_basal, i_smc): param_idx 1..25 -> row-major
      i_basal=$(( (PARAM_IDX - 1) / ${#NUM_SMC[@]} ))
      i_smc=$(( (PARAM_IDX - 1) % ${#NUM_SMC[@]} ))
      BASAL_DEATH_PROB=${BASAL_DEATH_PROBS[i_basal]}
      SMC=${NUM_SMC[i_smc]}

      RUN_NAME="${DATE}_p${PARAM_IDX}_r${REPL}"
      RUN_DIR="${BASE_DIR}/runs/${RUN_NAME}"
      JOB_FILE="job_${RUN_NAME}.sh"

      # Create run directory and scripts (only if not already set up)
      mkdir -p "${RUN_DIR}/scripts"
      mkdir -p "${RUN_DIR}/data/coords" "${RUN_DIR}/data/loops" "${RUN_DIR}/data/rep_states"
      if [ ! -f "${RUN_DIR}/scripts/loop_params.txt" ]; then
        cp -r "${BASE_DIR}/scripts/"* "${RUN_DIR}/scripts/"
      fi

      # REPL_SUFFIX for file names inside run (e.g. dna_Feb15_p1_r1_0.bin)
      REPL_SUFFIX="p${PARAM_IDX}_r${REPL}"
      sed "s/{DATE}/${DATE}/g; s/{REPL}/${REPL_SUFFIX}/g" "${RUN_DIR}/scripts/run_sc_chain_generation_template.sh" > "${RUN_DIR}/scripts/run_sc_chain_generation.sh"
      sed "s/{SEED}/${SEED}/g" "${RUN_DIR}/scripts/Syn3A_chromosome_init_template.inp" > "${RUN_DIR}/scripts/Syn3A_chromosome_init.inp"

      sed -i "s/^basal_death_prob=.*/basal_death_prob=${BASAL_DEATH_PROB}/g; s/^stall_death_prob=.*/stall_death_prob=${STALL_DEATH_PROB}/g; s/^numSmc=.*/numSmc=${SMC}/g; s/^bypass=.*/bypass=${BYPASS}/g; s/^knockoff=.*/knockoff=${KNOCKOFF}/g" "${RUN_DIR}/scripts/loop_params.txt"

      # Generate job script from template (SLOT is 1-based for this batch)
      SLOT=$((slot + 1))
      sed "s/{{RUN_NAME}}/${RUN_NAME}/g; s/{{SLOT}}/${SLOT}/g; s/{{SEED}}/${SEED}/g; s/{{START_TIME}}/${START_TIME}/g; s/{{END_TIME}}/${END_TIME}/g; s/{{IS_RESTART}}/${IS_RESTART}/g; s/{{V}}/${V}/g" job_template_2d_sweep.sh > "${JOB_FILE}"

      # Submit with optional dependency on previous batch
      if [ -n "$DEPENDENCY" ]; then
        SUBMIT_OUT=$(sbatch --dependency="$DEPENDENCY" "$JOB_FILE")
      else
        SUBMIT_OUT=$(sbatch "$JOB_FILE")
      fi
      JID=$(echo "$SUBMIT_OUT" | awk '{print $NF}')
      BATCH_JOB_IDS+=("$JID")

      echo "  Submitted ${JOB_FILE} (param_set=${PARAM_IDX}, rep=${REPL}, basal=${BASAL_DEATH_PROB}, numSmc=${SMC}, slot=${SLOT}) -> job ${JID}"
    done

    # Next batch waits for all jobs in this batch to finish
    DEPENDENCY="afterok:$(IFS=:; echo "${BATCH_JOB_IDS[*]}")"
  done
done

echo ""
echo "All jobs submitted. Run order: 25 param sets for replicate 1 (5 batches of 5), then replicate 2, then replicate 3."
