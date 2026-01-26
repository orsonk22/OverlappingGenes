#!/bin/bash
# =============================================================================
# SLURM Job Script for UCL DIAS Cluster
# Reading Frame Overlap Analysis
# =============================================================================

# --- SLURM CONFIGURATION - EDIT AS NEEDED ---
#SBATCH -p COMPUTE                        # CPU partition
#SBATCH -N1                               # One node
#SBATCH -n36                              # Number of CPUs (adjust based on availability)
#SBATCH --mem-per-cpu=2G                  # Memory per CPU
#SBATCH --time=48:00:00                   # Max runtime (up to 2 days)
#SBATCH --job-name=overlap_analysis
#SBATCH --output=logs/overlap_%j.out      # Standard output
#SBATCH --error=logs/overlap_%j.err       # Standard error

# Optional: Email notifications (uncomment and edit)
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=YOUR_EMAIL@ucl.ac.uk

# =============================================================================
# SETUP
# =============================================================================

# Print job info
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_ON_NODE"
echo "Start time: $(date)"
echo "========================================"

# Change to the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p results

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Option 1: Use system Python (if available with required packages)
# Uncomment if your system Python has numpy, numba, pandas, scipy

# Option 2: Use Miniforge/Conda environment (RECOMMENDED)
# Uncomment and edit the path to your environment
# source ~/miniforge3/bin/activate
# conda activate myenv

# Option 3: Use a virtual environment
# source ~/venvs/overlap/bin/activate

# Print Python info for debugging
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# =============================================================================
# RUN THE ANALYSIS
# =============================================================================

echo ""
echo "Starting overlap analysis..."
echo ""

# Full run
python cluster_overlap_analysis.py --output-dir ./results

# Alternative: Resume from checkpoint (if job timed out previously)
# python cluster_overlap_analysis.py --output-dir ./results --resume ./results/checkpoint.pkl

# Alternative: Test mode (quick verification)
# python cluster_overlap_analysis.py --output-dir ./results --test

# =============================================================================
# FINISH
# =============================================================================

echo ""
echo "========================================"
echo "Job finished: $(date)"
echo "========================================"
