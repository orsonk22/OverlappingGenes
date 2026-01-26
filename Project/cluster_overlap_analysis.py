#!/usr/bin/env python3
"""
Cluster-optimized Reading Frame Analysis Script

Parallelizes overlapping gene stability analysis across multiple CPU cores
using Python multiprocessing.

Usage:
    python cluster_overlap_analysis.py                    # Run full analysis
    python cluster_overlap_analysis.py --test             # Quick test mode
    python cluster_overlap_analysis.py --resume checkpoint.pkl  # Resume from checkpoint

Author: Based on large_scale_OVERLAPS.ipynb
"""

import os
import sys
import json
import pickle
import argparse
import time
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
import pandas as pd

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overlapping_genes_cluster as og

# =============================================================================
# CONFIGURATION - EDIT THESE PARAMETERS AS NEEDED
# =============================================================================

# --- SIMULATION PARAMETERS ---
ITERATIONS = 200_000          # MC iterations per trial
N_TRIALS = 200                 # Number of trials per overlap value
WHENTOSAVE = 0.01             # Save energy history every 1% of iterations

# --- OVERLAP CONFIGURATION ---
# Reading frames: Frame 0 (overlap % 3 == 0), Frame +1 (% 3 == 1), Frame +2 (% 3 == 2)
OVERLAP_START = 12            # Minimum overlap (nucleotides)
OVERLAP_STOP = 90             # Maximum overlap (exclusive)
OVERLAP_STEP = 6              # Step size within each frame

# --- DATA PATHS ---
BASE_DIR = "bmDCA"                              # Directory containing protein families
OPTIMAL_TEMPS_FILE = "optimal_temperatures.json"  # Optimal temperatures file
OUTPUT_FILE = "reading_frame_results_cluster.csv"  # Output results file

# --- PARALLELIZATION ---
# Set to None to auto-detect, or specify a number
N_WORKERS = None              # None = use all available CPUs

# --- CHECKPOINTING ---
CHECKPOINT_FREQ = 500         # Save checkpoint every N simulations
CHECKPOINT_FILE = "checkpoint.pkl"

# --- TEST MODE PARAMETERS ---
TEST_N_PAIRS = 2              # Number of pairs in test mode
TEST_N_OVERLAPS = 3           # Overlaps per frame in test mode
TEST_N_TRIALS = 2             # Trials per overlap in test mode
TEST_ITERATIONS = 10_000      # MC iterations in test mode

# =============================================================================
# END CONFIGURATION
# =============================================================================


def generate_overlaps(start, stop, step, reading_frames=[0, 1, 2]):
    """Generate overlap values for specified reading frames."""
    overlaps = []
    for frame in reading_frames:
        # Start at the first value >= start that matches this frame
        first = start + (frame - start % 3) % 3
        overlaps.extend(range(first, stop, step))
    return sorted(overlaps)


def load_all_data(base_dir, temps_file):
    """
    Load all protein family data upfront.

    Returns:
        pf_list: List of protein family names
        params_cache: Dict mapping PF name to (J, h) tuple
        stats_cache: Dict mapping PF name to (mean, std) tuple
        temps: Dict mapping PF name to optimal temperature
    """
    # Find protein families
    pf_dirs = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("PF")
    ])

    if not pf_dirs:
        raise FileNotFoundError(f"No protein families found in {base_dir}")

    print(f"Found {len(pf_dirs)} protein families: {pf_dirs}")

    # Load parameters and statistics
    params_cache = {}
    stats_cache = {}

    for pf in pf_dirs:
        param_file = os.path.join(base_dir, pf, f"{pf}_params.dat")
        nat_file = os.path.join(base_dir, pf, f"{pf}_naturalenergies.txt")

        J, h = og.extract_params(param_file)
        params_cache[pf] = (J, h)

        energies = og.load_natural_energies(nat_file)
        stats_cache[pf] = (np.mean(energies), np.std(energies))

    # Load optimal temperatures
    temps = {}
    if os.path.exists(temps_file):
        with open(temps_file, 'r') as f:
            temps = json.load(f)
        print(f"Loaded temperatures for {len(temps)} families")
    else:
        print(f"Warning: {temps_file} not found, using T=1.0 for all")

    return pf_dirs, params_cache, stats_cache, temps


def generate_work_units(pf_list, overlaps, n_trials, params_cache):
    """
    Generate all (pair_idx, overlap, trial) work units.

    Filters out overlaps that are invalid for specific pairs.
    """
    # Generate pairs (excluding self-pairs)
    pairs = []
    for i in range(len(pf_list)):
        for j in range(i + 1, len(pf_list)):
            pairs.append((pf_list[i], pf_list[j]))

    work_units = []
    for pair_idx, (pf1, pf2) in enumerate(pairs):
        # Get protein lengths
        len1 = len(params_cache[pf1][1]) // 21
        len2 = len(params_cache[pf2][1]) // 21
        max_overlap = min(len1, len2) * 3 - 6

        # Filter valid overlaps
        valid_overlaps = [ov for ov in overlaps if ov <= max_overlap]

        for overlap in valid_overlaps:
            for trial in range(n_trials):
                work_units.append((pair_idx, pf1, pf2, overlap, trial))

    return pairs, work_units


def run_single_simulation(args, params_cache, stats_cache, temps, iterations, whentosave):
    """
    Worker function for a single simulation.

    Args:
        args: (pair_idx, pf1, pf2, overlap, trial)
        params_cache, stats_cache, temps: Shared data
        iterations, whentosave: Simulation parameters

    Returns:
        dict with results
    """
    pair_idx, pf1, pf2, overlap, trial = args

    params1 = params_cache[pf1]
    params2 = params_cache[pf2]
    nat_mean1, nat_std1 = stats_cache[pf1]
    nat_mean2, nat_std2 = stats_cache[pf2]
    T1 = temps.get(pf1, 1.0)
    T2 = temps.get(pf2, 1.0)

    result = og.run_overlap_simulation(
        params1, params2,
        nat_mean1, nat_std1,
        nat_mean2, nat_std2,
        overlap,
        T1=T1, T2=T2,
        n_iterations=iterations,
        whentosave=whentosave
    )

    return {
        'pair_idx': pair_idx,
        'PF1': pf1,
        'PF2': pf2,
        'Overlap': overlap,
        'Trial': trial,
        'E1_final': result['E1_final'],
        'E2_final': result['E2_final'],
        'min_dist1': result['min_dist1'],
        'min_dist2': result['min_dist2'],
        'iter_conv1': result['iter_conv1'],
        'iter_conv2': result['iter_conv2'],
        'Nat_Mean1': nat_mean1,
        'Nat_Std1': nat_std1,
        'Nat_Mean2': nat_mean2,
        'Nat_Std2': nat_std2
    }


def aggregate_results(raw_results, pairs):
    """
    Aggregate per-trial results into per-(pair, overlap) summaries.

    Matches the output format of the original notebook.
    """
    df = pd.DataFrame(raw_results)

    # Group by pair and overlap
    grouped = df.groupby(['PF1', 'PF2', 'Overlap'])

    summary = []
    for (pf1, pf2, overlap), group in grouped:
        E1_vals = group['E1_final'].values
        E2_vals = group['E2_final'].values
        nat_mean1 = group['Nat_Mean1'].iloc[0]
        nat_mean2 = group['Nat_Mean2'].iloc[0]
        nat_std1 = group['Nat_Std1'].iloc[0]
        nat_std2 = group['Nat_Std2'].iloc[0]

        summary.append({
            'PF1': pf1,
            'PF2': pf2,
            'Overlap': overlap,
            'Reading_Frame': overlap % 3,
            'Mean_E1': np.mean(E1_vals),
            'Std_E1': np.std(E1_vals),
            'Mean_E2': np.mean(E2_vals),
            'Std_E2': np.std(E2_vals),
            'Nat_Mean1': nat_mean1,
            'Nat_Mean2': nat_mean2,
            'Nat_Std1': nat_std1,
            'Nat_Std2': nat_std2,
            'Dist_1': np.mean(np.abs(E1_vals - nat_mean1)),
            'Dist_2': np.mean(np.abs(E2_vals - nat_mean2)),
            'Min_Dist_1': group['min_dist1'].mean(),
            'Min_Dist_2': group['min_dist2'].mean(),
            'Iter_Converged_1': group['iter_conv1'].mean(),
            'Iter_Converged_2': group['iter_conv2'].mean()
        })

    return pd.DataFrame(summary)


def save_checkpoint(completed_results, remaining_units, checkpoint_file):
    """Save progress to checkpoint file."""
    with open(checkpoint_file, 'wb') as f:
        pickle.dump({
            'completed': completed_results,
            'remaining': remaining_units
        }, f)
    print(f"Checkpoint saved: {len(completed_results)} completed, {len(remaining_units)} remaining")


def load_checkpoint(checkpoint_file):
    """Load progress from checkpoint file."""
    with open(checkpoint_file, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded checkpoint: {len(data['completed'])} completed, {len(data['remaining'])} remaining")
    return data['completed'], data['remaining']


def main():
    parser = argparse.ArgumentParser(description='Cluster overlap analysis')
    parser.add_argument('--test', action='store_true', help='Run in test mode (quick run)')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint file')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers (default: auto)')
    args = parser.parse_args()

    # Determine number of workers
    n_workers = args.workers or N_WORKERS or cpu_count()
    print(f"Using {n_workers} worker processes")

    # Set parameters based on mode
    if args.test:
        print("\n=== TEST MODE ===")
        iterations = TEST_ITERATIONS
        n_trials = TEST_N_TRIALS
        overlap_step = (OVERLAP_STOP - OVERLAP_START) // TEST_N_OVERLAPS
    else:
        iterations = ITERATIONS
        n_trials = N_TRIALS
        overlap_step = OVERLAP_STEP

    # Generate overlaps
    overlaps = generate_overlaps(OVERLAP_START, OVERLAP_STOP, overlap_step)
    print(f"Overlap values: {len(overlaps)} ({min(overlaps)}-{max(overlaps)} nt)")

    # Load data
    print("\nLoading data...")
    pf_list, params_cache, stats_cache, temps = load_all_data(BASE_DIR, OPTIMAL_TEMPS_FILE)

    # Limit pairs in test mode
    if args.test:
        pf_list = pf_list[:TEST_N_PAIRS + 1]  # Need at least 2 for pairs
        print(f"Test mode: using {len(pf_list)} protein families")

    # Generate work units
    pairs, work_units = generate_work_units(pf_list, overlaps, n_trials, params_cache)
    print(f"\nTotal pairs: {len(pairs)}")
    print(f"Total simulations: {len(work_units):,}")

    # Resume from checkpoint if specified
    completed_results = []
    if args.resume and os.path.exists(args.resume):
        completed_results, work_units = load_checkpoint(args.resume)

    if not work_units:
        print("No work remaining!")
        return

    # Create worker function with fixed arguments
    worker_fn = partial(
        run_single_simulation,
        params_cache=params_cache,
        stats_cache=stats_cache,
        temps=temps,
        iterations=iterations,
        whentosave=WHENTOSAVE
    )

    # Run parallel processing
    print(f"\nStarting parallel processing...")
    start_time = time.time()

    checkpoint_file = os.path.join(args.output_dir, CHECKPOINT_FILE)

    try:
        with Pool(n_workers) as pool:
            # Use imap for progress tracking and checkpointing
            results_iter = pool.imap(worker_fn, work_units, chunksize=10)

            for i, result in enumerate(results_iter):
                completed_results.append(result)

                # Progress update
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = len(work_units) - (i + 1)
                    eta = remaining / rate if rate > 0 else 0
                    print(f"Progress: {i + 1}/{len(work_units)} ({100*(i+1)/len(work_units):.1f}%) "
                          f"- Rate: {rate:.1f}/s - ETA: {eta/60:.1f} min")

                # Checkpoint
                if (i + 1) % CHECKPOINT_FREQ == 0:
                    remaining_units = work_units[i + 1:]
                    save_checkpoint(completed_results, remaining_units, checkpoint_file)

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        # Find remaining work
        n_completed = len(completed_results)
        remaining_units = work_units[n_completed:]
        save_checkpoint(completed_results, remaining_units, checkpoint_file)
        return

    total_time = time.time() - start_time
    print(f"\nCompleted {len(completed_results):,} simulations in {total_time/60:.1f} minutes")

    # Aggregate results
    print("\nAggregating results...")
    results_df = aggregate_results(completed_results, pairs)

    # Save results
    output_path = os.path.join(args.output_dir, OUTPUT_FILE)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("Checkpoint file removed")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(results_df)}")
    print(f"Reading frames: {sorted(results_df['Reading_Frame'].unique())}")
    print(f"\nMean metrics by frame:")
    frame_summary = results_df.groupby('Reading_Frame')[['Dist_1', 'Dist_2', 'Min_Dist_1', 'Min_Dist_2']].mean()
    print(frame_summary.round(2))


if __name__ == "__main__":
    main()
