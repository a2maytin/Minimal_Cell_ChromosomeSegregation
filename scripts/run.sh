#!/bin/bash
# Minimal driver: sc_chain_generation -> btree_chromo energy minimization
#
# Usage (from anywhere; runs relative to this script's directory):
#   ./run.sh [SEED]
#
# Optional environment overrides:
#   SC_CHAIN_GEN   path to gen_sc_chain
#   BTREE_CHROMO   path to btree_chromo executable
#   NPROC          threads for sc_chain_generation (default: 8)
#   SKIP_SC_CHAIN  set to 1 to skip coordinate generation

set -euo pipefail
cd "$(dirname "$0")"

SEED="${1:-10}"
RUN_NAME="bare_run"
SC_CHAIN_GEN="${SC_CHAIN_GEN:-/home/andrew/Desktop/Projects/sc_chain_generation/fortran/gen_sc_chain}"
BTREE_CHROMO="${BTREE_CHROMO:-/home/andrew/Desktop/Projects/btree_chromo_gpu/build/apps/btree_chromo}"
NPROC="${NPROC:-8}"
SKIP_SC_CHAIN="${SKIP_SC_CHAIN:-0}"
DATA_DIR="../data"

if [[ ! -x "$SC_CHAIN_GEN" ]]; then
    echo "ERROR: gen_sc_chain not found or not executable: $SC_CHAIN_GEN" >&2
    exit 1
fi
if [[ ! -x "$BTREE_CHROMO" ]]; then
    echo "ERROR: btree_chromo not found or not executable: $BTREE_CHROMO" >&2
    exit 1
fi
if [[ ! -f minimize.inp ]]; then
    echo "ERROR: minimize.inp not found" >&2
    exit 1
fi

mkdir -p "${DATA_DIR}/coords"

if [[ "$SKIP_SC_CHAIN" != "1" ]]; then
    echo "=== Generating initial DNA and ribosome coordinates with sc_chain_generation ==="
    sed "s/^seed = .*/seed = ${SEED}/" Syn3A_chromosome_init.inp > "${DATA_DIR}/coords/Syn3A_chromosome_init.inp"

    INPUT_FNAME="${DATA_DIR}/coords/Syn3A_chromosome_init.inp"
    OUTPUT_DIR="${DATA_DIR}/coords/"
    LOG_FNAME="${OUTPUT_DIR}log_init.log"
    OUT_LABEL="Syn3A_chromosome_init"

    echo "Executing: ${SC_CHAIN_GEN} --i_f=${INPUT_FNAME} --o_d=${OUTPUT_DIR} --o_l=${OUT_LABEL} --s=${SEED} --l=${LOG_FNAME} --n_t=${NPROC} --bin --xyz"
    "${SC_CHAIN_GEN}" \
        --i_f="${INPUT_FNAME}" \
        --o_d="${OUTPUT_DIR}" \
        --o_l="${OUT_LABEL}" \
        --s="${SEED}" \
        --l="${LOG_FNAME}" \
        --n_t="${NPROC}" \
        --bin \
        --xyz

    cp "${OUTPUT_DIR}x_chain_${OUT_LABEL}_rep00001.bin" "${DATA_DIR}/coords/dna_${RUN_NAME}_0.bin"
    cp "${OUTPUT_DIR}x_obst_${OUT_LABEL}_rep00001.bin" "${DATA_DIR}/coords/ribo_${RUN_NAME}_0.bin"
    echo "Wrote ${DATA_DIR}/coords/dna_${RUN_NAME}_0.bin and ${DATA_DIR}/coords/ribo_${RUN_NAME}_0.bin"
else
    echo "=== Skipping sc_chain_generation (SKIP_SC_CHAIN=1) ==="
    if [[ ! -f "${DATA_DIR}/coords/dna_${RUN_NAME}_0.bin" || ! -f "${DATA_DIR}/coords/ribo_${RUN_NAME}_0.bin" ]]; then
        echo "ERROR: expected ${DATA_DIR}/coords/dna_${RUN_NAME}_0.bin and ${DATA_DIR}/coords/ribo_${RUN_NAME}_0.bin" >&2
        exit 1
    fi
fi

echo "=== Running btree_chromo energy minimization ==="
export LD_LIBRARY_PATH="/usr/local/Software/LAMMPS/GPU_Kokkos/lib:/usr/local/Libraries/OpenMPI/4.1.4/lib:${LD_LIBRARY_PATH:-}"
"${BTREE_CHROMO}" minimize.inp

echo "=== Done ==="
echo "Outputs:"
echo "  ${DATA_DIR}/data_pre_minimization.lammps"
echo "  ${DATA_DIR}/data_post_minimization.lammps"
echo "  ${DATA_DIR}/${RUN_NAME}.log"
