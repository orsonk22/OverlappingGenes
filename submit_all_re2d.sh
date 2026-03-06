#!/bin/bash
#SBATCH --job-name=re2d
#SBATCH --array=0-65%20
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/re2d_%a.out
#SBATCH --error=logs/re2d_%a.err

# ── Environment ──────────────────────────────────────────────────────
# Prevent thread oversubscription: each worker is a separate process
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="/home/xzccaogk/OverlappingGenes"
OUTPUT_DIR="${SCRIPT_DIR}/results_re2d"
DATA_DIR="${SCRIPT_DIR}/bmDCA"

mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

# Use node-local scratch for mmap temp files
export SLURM_TMPDIR="/home/xzccaogk/OverlappingGenes/tmp_mmap/${SLURM_ARRAY_TASK_ID}"
mkdir -p "${SLURM_TMPDIR}"

# ── Run ──────────────────────────────────────────────────────────────
echo "Starting pair ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}, Temp: ${SLURM_TMPDIR}"

python "${SCRIPT_DIR}/re_2d_cluster.py" \
    --pair-index "${SLURM_ARRAY_TASK_ID}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --output-dir "${OUTPUT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --resume

echo "Finished pair ${SLURM_ARRAY_TASK_ID} at $(date)"

# Clean up temp
rm -rf "/home/xzccaogk/OverlappingGenes/tmp_mmap/${SLURM_ARRAY_TASK_ID}"
