#!/usr/bin/env python3
"""
2D Replica Exchange — Cluster Script

Runs 2D RE for a single protein pair (selected by --pair-index) across
three overlap lengths (max_ov, max_ov-1, max_ov-2). Each overlap is
processed with staggered temperature grids and independent repeats,
using multiprocessing for internal parallelism.

Usage:
    python re_2d_cluster.py --pair-index 0                    # Run pair 0
    python re_2d_cluster.py --pair-index 0 --test             # Quick test
    python re_2d_cluster.py --pair-index 0 --resume           # Resume from checkpoint
    python re_2d_cluster.py --pair-index 0 --workers 10       # Specify workers

Output: results_re2d/{PF1}_{PF2}_re2d.npz
"""

import os
import sys

# Set NUMBA_NUM_THREADS=1 BEFORE any numba import to avoid thread oversubscription
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import pickle
import signal
import tempfile
import time
from itertools import combinations
from multiprocessing import Pool, cpu_count

import numpy as np

# Add this directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overlappingGenes import (
    extract_params, load_natural_energies, initial_seq_no_stops,
    seq_str_to_int_array, re_2d_equilibrium_sampler,
)
from re_2d_worker import init_worker, run_single_stagger

# =============================================================================
# CONFIGURATION
# =============================================================================

# Temperature grid
T1_MIN, T1_MAX = 0.2, 1.5
T2_MIN, T2_MAX = 0.2, 1.5

# Grid dimensions (per stagger run)
N_T1 = 10
N_T2 = 10

# Stagger and repeats
N_STAGGER = 3       # 3x3 = 9 runs per repeat
N_REPEATS = 10      # independent repeats

# MC parameters
RE2D_BURNIN = 200_000
RE2D_SAMPLES = 100_000
RE2D_SAMPLE_INTERVAL = 100
RE2D_SWAP_INTERVAL = 50

# nat_std passed to the sampler's Metropolis criterion.
# 1.0 = "raw mode" (no z-score rescaling inside the MC step).
# The notebook uses 1.0; actual nat_std is only used for post-hoc metrics.
MC_NAT_STD1 = 1.0
MC_NAT_STD2 = 1.0

# Derived
N_T1_FINE = N_T1 * N_STAGGER   # 30
N_T2_FINE = N_T2 * N_STAGGER   # 30
N_RUNS_PER_REPEAT = N_STAGGER ** 2  # 9
SEED_BASE = 42

# Test mode overrides
TEST_N_T1 = 4
TEST_N_T2 = 4
TEST_N_STAGGER = 2
TEST_N_REPEATS = 2
TEST_BURNIN = 10_000
TEST_SAMPLES = 1_000

# =============================================================================
# PAIR GENERATION
# =============================================================================

def get_protein_families(data_dir):
    """Find all protein families in bmDCA directory."""
    pf_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("PF")
    ])
    if not pf_dirs:
        raise FileNotFoundError(f"No protein families found in {data_dir}")
    return pf_dirs


def get_pair(pf_list, pair_index):
    """Get the (PF1, PF2) pair for a given index from C(n,2) combinations."""
    pairs = list(combinations(pf_list, 2))
    if pair_index < 0 or pair_index >= len(pairs):
        raise ValueError(f"pair-index must be 0-{len(pairs)-1}, got {pair_index}")
    return pairs[pair_index]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pair_data(pf1, pf2, data_dir):
    """Load DCA params and natural energy stats for both proteins."""
    params = {}
    nat_stats = {}
    nat_energies = {}

    for pf in (pf1, pf2):
        param_file = os.path.join(data_dir, pf, f"{pf}_params.dat")
        nat_file = os.path.join(data_dir, pf, f"{pf}_naturalenergies.txt")

        J, h = extract_params(param_file)
        params[pf] = (np.asarray(J, dtype=np.float64),
                       np.asarray(h, dtype=np.float64))

        energies = np.array(load_natural_energies(nat_file))
        nat_energies[pf] = energies
        nat_stats[pf] = (float(np.mean(energies)), float(np.std(energies)))

    return params, nat_stats, nat_energies


# =============================================================================
# JIT WARMUP
# =============================================================================

def jit_warmup(params, pf1, pf2, overlap_len):
    """Tiny 2x2 RE run to trigger Numba JIT compilation."""
    J1, h1 = params[pf1]
    J2, h2 = params[pf2]
    prot1_len = len(h1) / 21
    prot2_len = len(h2) / 21

    len_seq_1_n = int(3 * prot1_len + 3)
    len_seq_2_n = int(3 * prot2_len + 3)
    seq_length = len_seq_1_n + len_seq_2_n - overlap_len

    n_total = 4  # 2x2
    initial_seqs = np.empty((n_total, seq_length), dtype=np.uint8)
    for idx in range(n_total):
        s = initial_seq_no_stops(prot1_len, prot2_len, overlap_len, quiet=True)
        initial_seqs[idx] = seq_str_to_int_array(s)

    progress = np.zeros(1, dtype=np.int64)
    re_2d_equilibrium_sampler(
        J1, h1, J2, h2,
        initial_seqs,
        np.array([0.5, 1.0]),
        np.array([0.5, 1.0]),
        50, 5, 10, 10,
        1.0, 1.0,
        progress
    )


# =============================================================================
# CHECKPOINTING
# =============================================================================

def get_checkpoint_path(output_dir, pf1, pf2):
    return os.path.join(output_dir, f"{pf1}_{pf2}_checkpoint.pkl")


def save_checkpoint(ckpt_path, data):
    """Save checkpoint atomically."""
    tmp_path = ckpt_path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(data, f)
    os.replace(tmp_path, ckpt_path)


def load_checkpoint(ckpt_path):
    """Load checkpoint if it exists."""
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            return pickle.load(f)
    return None


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_z_score(E1, E2, nat_mean1, nat_std1, nat_mean2, nat_std2):
    """Z-score metric: mean(|z1|) + mean(|z2|) over samples. Vectorized."""
    return (np.mean(np.abs((E1 - nat_mean1) / nat_std1), axis=-1)
          + np.mean(np.abs((E2 - nat_mean2) / nat_std2), axis=-1))


def compute_wasserstein(E1, E2, nat_energies_1, nat_energies_2,
                        n_T2_fine, n_T1_fine):
    """Wasserstein metric per grid point. Returns (n_T2, n_T1) array."""
    from scipy.stats import wasserstein_distance

    nat1_sorted = np.sort(nat_energies_1)
    nat2_sorted = np.sort(nat_energies_2)
    wass = np.full((n_T2_fine, n_T1_fine), np.nan)

    for i in range(n_T2_fine):
        for j in range(n_T1_fine):
            w1 = wasserstein_distance(nat1_sorted, E1[i, j])
            w2 = wasserstein_distance(nat2_sorted, E2[i, j])
            wass[i, j] = (w1 + w2) / 2

    return wass


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_overlap(overlap_len, params, nat_stats, nat_energies,
                    pf1, pf2, n_workers, tmp_dir,
                    n_t1, n_t2, n_stagger, n_repeats,
                    n_burnin, n_samples, sample_interval, swap_interval,
                    checkpoint_data=None):
    """
    Process one overlap length: run all repeats, compute metrics.

    Uses memory-mapped temp files to hold raw energy samples,
    processes one repeat at a time to control peak RAM.
    """
    J1, h1 = params[pf1]
    J2, h2 = params[pf2]
    prot1_len = len(h1) / 21
    prot2_len = len(h2) / 21
    nat_mean1, nat_std1 = nat_stats[pf1]
    nat_mean2, nat_std2 = nat_stats[pf2]

    n_t1_fine = n_t1 * n_stagger
    n_t2_fine = n_t2 * n_stagger

    T1_fine = np.linspace(T1_MIN, T1_MAX, n_t1_fine)
    T2_fine = np.linspace(T2_MIN, T2_MAX, n_t2_fine)

    # Prepare shared data for worker pool
    shared_data = {
        'Jvec1': J1, 'hvec1': h1,
        'Jvec2': J2, 'hvec2': h2,
        'prot1_len': prot1_len,
        'prot2_len': prot2_len,
    }

    # Memory-mapped temp files for energy samples
    mmap_E1_path = os.path.join(tmp_dir, f"E1_ov{overlap_len}.dat")
    mmap_E2_path = os.path.join(tmp_dir, f"E2_ov{overlap_len}.dat")
    mmap_shape = (n_repeats, n_t2_fine, n_t1_fine, n_samples)

    E1_mmap = np.memmap(mmap_E1_path, dtype=np.float64, mode='w+', shape=mmap_shape)
    E2_mmap = np.memmap(mmap_E2_path, dtype=np.float64, mode='w+', shape=mmap_shape)

    # Small metric arrays (kept in RAM)
    z_all = np.full((n_repeats, n_t2_fine, n_t1_fine), np.nan)
    wass_all = np.full((n_repeats, n_t2_fine, n_t1_fine), np.nan)
    acc_all = np.full((n_repeats, n_t2_fine, n_t1_fine), np.nan)
    run_diagnostics = []

    # Determine starting repeat from checkpoint
    start_repeat = 0
    if checkpoint_data and overlap_len in checkpoint_data.get('completed_overlaps', {}):
        ov_ckpt = checkpoint_data['completed_overlaps'][overlap_len]
        start_repeat = ov_ckpt['completed_repeats']
        if start_repeat > 0:
            z_all[:start_repeat] = ov_ckpt['z_all'][:start_repeat]
            wass_all[:start_repeat] = ov_ckpt['wass_all'][:start_repeat]
            acc_all[:start_repeat] = ov_ckpt['acc_all'][:start_repeat]
            run_diagnostics = list(ov_ckpt.get('run_diagnostics', []))
            print(f"  Resuming from repeat {start_repeat}/{n_repeats}")

    pool_size = min(n_workers, n_stagger ** 2)

    for r in range(start_repeat, n_repeats):
        t_repeat = time.time()

        # Build work items for this repeat: all stagger offsets
        work_items = []
        for a in range(n_stagger):
            for b in range(n_stagger):
                run_idx = r * n_stagger**2 + a * n_stagger + b
                seed = SEED_BASE + overlap_len * 10000 + run_idx

                T1_sub = T1_fine[a::n_stagger]  # n_t1 points
                T2_sub = T2_fine[b::n_stagger]  # n_t2 points

                work_items.append((
                    r, a, b, seed, overlap_len, T1_sub, T2_sub,
                    n_burnin, n_samples, sample_interval, swap_interval,
                    MC_NAT_STD1, MC_NAT_STD2
                ))

        # Run stagger workers
        with Pool(pool_size, initializer=init_worker,
                  initargs=(shared_data,)) as pool:
            results = pool.map(run_single_stagger, work_items)

        # Assemble results into fine grid
        for res in results:
            a, b = res['a'], res['b']
            for i in range(n_t2):
                for j in range(n_t1):
                    i_fine = b + i * n_stagger
                    j_fine = a + j * n_stagger
                    E1_mmap[r, i_fine, j_fine, :] = res['E1_grid'][i, j]
                    E2_mmap[r, i_fine, j_fine, :] = res['E2_grid'][i, j]
                    acc_all[r, i_fine, j_fine] = res['acceptance_grid'][i, j]

            mean_swap_h = float(np.mean(res['swap_rates_h']))
            mean_swap_v = float(np.mean(res['swap_rates_v']))
            run_diagnostics.append({
                'repeat': res['repeat'], 'a': res['a'], 'b': res['b'],
                'mean_swap_h': mean_swap_h,
                'mean_swap_v': mean_swap_v,
                'mean_swap': (mean_swap_h + mean_swap_v) / 2,
                'elapsed': res['elapsed_time'],
            })

        # Flush mmap to disk
        E1_mmap.flush()
        E2_mmap.flush()

        # Compute per-repeat metrics from mmap
        z_all[r] = compute_z_score(
            E1_mmap[r], E2_mmap[r],
            nat_mean1, nat_std1, nat_mean2, nat_std2
        )
        wass_all[r] = compute_wasserstein(
            E1_mmap[r], E2_mmap[r],
            nat_energies[pf1], nat_energies[pf2],
            n_t2_fine, n_t1_fine
        )

        elapsed_repeat = time.time() - t_repeat
        print(f"  Repeat {r+1}/{n_repeats} done in {elapsed_repeat:.1f}s "
              f"(z_min={np.nanmin(z_all[r]):.3f}, wass_min={np.nanmin(wass_all[r]):.2f})")

        # Yield checkpoint data
        yield {
            'type': 'checkpoint',
            'overlap_len': overlap_len,
            'completed_repeats': r + 1,
            'z_all': z_all[:r+1].copy(),
            'wass_all': wass_all[:r+1].copy(),
            'acc_all': acc_all[:r+1].copy(),
            'run_diagnostics': list(run_diagnostics),
        }

    # All repeats done — compute final metrics
    z_mean = np.mean(z_all, axis=0)
    z_std = np.std(z_all, axis=0)
    wass_mean = np.mean(wass_all, axis=0)
    wass_std = np.std(wass_all, axis=0)
    acc_mean = np.mean(acc_all, axis=0)
    acc_std = np.std(acc_all, axis=0)

    # Find best 5 points per metric
    def find_best_n(grid, n=5):
        flat_idx = np.argsort(grid.ravel())[:n]
        indices = np.array(np.unravel_index(flat_idx, grid.shape)).T  # (n, 2)
        return indices

    z_best_idx = find_best_n(z_mean, 5)
    wass_best_idx = find_best_n(wass_mean, 5)

    # Extract energy samples at best points.
    # Use the last repeat run THIS session (mmap has zeros for earlier repeats on resume).
    extract_rep = n_repeats - 1  # always valid: either fresh run or last resumed repeat

    def extract_best_energies(mmap, indices, rep):
        E_list = []
        for idx in indices:
            E_list.append(mmap[rep, idx[0], idx[1], :].copy())
        return np.array(E_list)

    z_best_E1 = extract_best_energies(E1_mmap, z_best_idx, extract_rep)
    z_best_E2 = extract_best_energies(E2_mmap, z_best_idx, extract_rep)
    wass_best_E1 = extract_best_energies(E1_mmap, wass_best_idx, extract_rep)
    wass_best_E2 = extract_best_energies(E2_mmap, wass_best_idx, extract_rep)

    # Swap diagnostic summary
    swap_summary = {
        'mean_swap_rates': [d['mean_swap'] for d in run_diagnostics],
        'elapsed_times': [d['elapsed'] for d in run_diagnostics],
    }

    # Clean up mmap files
    del E1_mmap, E2_mmap
    for path in (mmap_E1_path, mmap_E2_path):
        try:
            os.remove(path)
        except OSError:
            pass

    # Yield final results
    yield {
        'type': 'final',
        'overlap_len': overlap_len,
        'T1_fine': T1_fine,
        'T2_fine': T2_fine,
        'z_mean': z_mean,
        'z_std': z_std,
        'wass_mean': wass_mean,
        'wass_std': wass_std,
        'acc_mean': acc_mean,
        'acc_std': acc_std,
        'z_best_indices': z_best_idx,
        'wass_best_indices': wass_best_idx,
        'z_best_E1': z_best_E1,
        'z_best_E2': z_best_E2,
        'wass_best_E1': wass_best_E1,
        'wass_best_E2': wass_best_E2,
        'swap_diagnostics': swap_summary,
    }


# =============================================================================
# SIGTERM HANDLER
# =============================================================================

_checkpoint_state = {}


def sigterm_handler(signum, frame):
    """Save checkpoint on SIGTERM (cluster time limit)."""
    if _checkpoint_state:
        ckpt_path = _checkpoint_state.get('ckpt_path')
        data = _checkpoint_state.get('data')
        if ckpt_path and data:
            print(f"\nSIGTERM received — saving checkpoint to {ckpt_path}")
            save_checkpoint(ckpt_path, data)
            print("Checkpoint saved. Exiting.")
    sys.exit(0)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='2D Replica Exchange — Cluster Script')
    parser.add_argument('--pair-index', type=int, required=True,
                        help='Pair index (0-65 for 12 families)')
    parser.add_argument('--test', action='store_true',
                        help='Quick test mode')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--output-dir', type=str, default='results_re2d',
                        help='Output directory (default: results_re2d)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of workers (default: auto)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to bmDCA directory (default: ../bmDCA)')
    args = parser.parse_args()

    # Register SIGTERM handler
    signal.signal(signal.SIGTERM, sigterm_handler)

    # Resolve data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(script_dir, '..', 'bmDCA')
    data_dir = os.path.normpath(data_dir)

    # Apply test mode overrides
    if args.test:
        n_t1, n_t2 = TEST_N_T1, TEST_N_T2
        n_stagger = TEST_N_STAGGER
        n_repeats = TEST_N_REPEATS
        n_burnin = TEST_BURNIN
        n_samples = TEST_SAMPLES
        sample_interval = RE2D_SAMPLE_INTERVAL
        swap_interval = RE2D_SWAP_INTERVAL
        print("=== TEST MODE ===")
    else:
        n_t1, n_t2 = N_T1, N_T2
        n_stagger = N_STAGGER
        n_repeats = N_REPEATS
        n_burnin = RE2D_BURNIN
        n_samples = RE2D_SAMPLES
        sample_interval = RE2D_SAMPLE_INTERVAL
        swap_interval = RE2D_SWAP_INTERVAL

    n_t1_fine = n_t1 * n_stagger
    n_t2_fine = n_t2 * n_stagger

    # Determine workers
    n_workers = args.workers or min(cpu_count(), n_stagger ** 2)
    print(f"Workers: {n_workers}")

    # Get protein pair
    pf_list = get_protein_families(data_dir)
    print(f"Found {len(pf_list)} protein families: {pf_list}")
    n_pairs = len(pf_list) * (len(pf_list) - 1) // 2
    print(f"Total pairs: {n_pairs}")

    pf1, pf2 = get_pair(pf_list, args.pair_index)
    print(f"\nPair {args.pair_index}: {pf1} x {pf2}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Temp directory for mmap files
    tmp_dir = os.environ.get('SLURM_TMPDIR', tempfile.gettempdir())
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"Temp dir: {tmp_dir}")

    # Load data
    print("\nLoading DCA parameters and natural energies...")
    t0 = time.time()
    params, nat_stats, nat_energies = load_pair_data(pf1, pf2, data_dir)
    print(f"Data loaded in {time.time() - t0:.1f}s")

    prot1_len = len(params[pf1][1]) / 21
    prot2_len = len(params[pf2][1]) / 21
    print(f"  {pf1}: {int(prot1_len)} AA ({int(3*prot1_len)} nt)")
    print(f"  {pf2}: {int(prot2_len)} AA ({int(3*prot2_len)} nt)")
    print(f"  {pf1} natural energy: mean={nat_stats[pf1][0]:.2f}, std={nat_stats[pf1][1]:.2f}")
    print(f"  {pf2} natural energy: mean={nat_stats[pf2][0]:.2f}, std={nat_stats[pf2][1]:.2f}")

    # Compute overlap lengths
    max_ov = int(min(prot1_len, prot2_len) * 3 - 6)
    overlap_lengths = [max_ov, max_ov - 1, max_ov - 2]
    print(f"\nOverlap lengths: {overlap_lengths} (frames {[ov % 3 for ov in overlap_lengths]})")

    # JIT warmup
    print("\nJIT warmup...")
    t0 = time.time()
    jit_warmup(params, pf1, pf2, overlap_lengths[0])
    print(f"JIT warmup done in {time.time() - t0:.1f}s")

    # Load checkpoint if resuming
    ckpt_path = get_checkpoint_path(args.output_dir, pf1, pf2)
    checkpoint_data = None
    if args.resume:
        checkpoint_data = load_checkpoint(ckpt_path)
        if checkpoint_data:
            print(f"Loaded checkpoint: {list(checkpoint_data.get('completed_overlaps', {}).keys())} completed")
        else:
            print("No checkpoint found, starting fresh")

    # Initialize checkpoint state for SIGTERM handler
    if checkpoint_data is None:
        checkpoint_data = {'completed_overlaps': {}}
    _checkpoint_state['ckpt_path'] = ckpt_path
    _checkpoint_state['data'] = checkpoint_data

    # NPZ output data accumulator
    npz_data = {
        'pf1': pf1,
        'pf2': pf2,
        'overlap_lengths': np.array(overlap_lengths),
        'nat_mean_1': nat_stats[pf1][0],
        'nat_std_1': nat_stats[pf1][1],
        'nat_mean_2': nat_stats[pf2][0],
        'nat_std_2': nat_stats[pf2][1],
        'config': np.array(str({
            'T1_range': (T1_MIN, T1_MAX),
            'T2_range': (T2_MIN, T2_MAX),
            'n_t1': n_t1, 'n_t2': n_t2,
            'n_stagger': n_stagger, 'n_repeats': n_repeats,
            'n_burnin': n_burnin, 'n_samples': n_samples,
            'sample_interval': sample_interval,
            'swap_interval': swap_interval,
        })),
    }

    # Process each overlap
    t_total = time.time()
    completed_overlaps = set(checkpoint_data.get('completed_overlaps', {}).keys())

    # Check if overlap was fully completed in a previous run
    skip_overlaps = set()
    for ov in overlap_lengths:
        ov_ckpt = checkpoint_data.get('completed_overlaps', {}).get(ov)
        if ov_ckpt and ov_ckpt.get('completed_repeats', 0) >= n_repeats:
            if ov_ckpt.get('final_results'):
                skip_overlaps.add(ov)

    for ov_idx, overlap_len in enumerate(overlap_lengths):
        print(f"\n{'='*60}")
        print(f"Overlap {overlap_len} nt (frame {overlap_len % 3}) "
              f"[{ov_idx+1}/{len(overlap_lengths)}]")
        print(f"{'='*60}")

        if overlap_len in skip_overlaps:
            # Restore final results from checkpoint
            final_res = checkpoint_data['completed_overlaps'][overlap_len]['final_results']
            for key, val in final_res.items():
                npz_data[f"{key}_{overlap_len}"] = val
            print("  Skipping (already completed in checkpoint)")
            continue

        t_ov = time.time()
        for event in process_overlap(
            overlap_len, params, nat_stats, nat_energies,
            pf1, pf2, n_workers, tmp_dir,
            n_t1, n_t2, n_stagger, n_repeats,
            n_burnin, n_samples, sample_interval, swap_interval,
            checkpoint_data=checkpoint_data,
        ):
            if event['type'] == 'checkpoint':
                # Update checkpoint state
                checkpoint_data.setdefault('completed_overlaps', {})[overlap_len] = event
                _checkpoint_state['data'] = checkpoint_data
                save_checkpoint(ckpt_path, checkpoint_data)

            elif event['type'] == 'final':
                # Store results for NPZ
                final_keys = {}
                for key in ('T1_fine', 'T2_fine', 'z_mean', 'z_std',
                            'wass_mean', 'wass_std', 'acc_mean', 'acc_std',
                            'z_best_indices', 'wass_best_indices',
                            'z_best_E1', 'z_best_E2',
                            'wass_best_E1', 'wass_best_E2',
                            'swap_diagnostics'):
                    val = event[key]
                    if isinstance(val, dict):
                        # Convert swap_diagnostics dict to arrays
                        for sk, sv in val.items():
                            arr = np.array(sv)
                            npz_data[f"swap_{sk}_{overlap_len}"] = arr
                            final_keys[f"swap_{sk}"] = arr
                    else:
                        npz_data[f"{key}_{overlap_len}"] = val
                        final_keys[key] = val

                # Mark overlap as fully completed in checkpoint
                checkpoint_data.setdefault('completed_overlaps', {})[overlap_len] = {
                    'completed_repeats': n_repeats,
                    'final_results': final_keys,
                }
                save_checkpoint(ckpt_path, checkpoint_data)

                elapsed_ov = time.time() - t_ov
                print(f"  Overlap {overlap_len} completed in {elapsed_ov:.1f}s")

                # Summary
                z_min_idx = np.unravel_index(
                    np.argmin(event['z_mean']), event['z_mean'].shape)
                wass_min_idx = np.unravel_index(
                    np.argmin(event['wass_mean']), event['wass_mean'].shape)
                print(f"  Best z-score: {event['z_mean'][z_min_idx]:.3f} "
                      f"at T1={event['T1_fine'][z_min_idx[1]]:.4f}, "
                      f"T2={event['T2_fine'][z_min_idx[0]]:.4f}")
                print(f"  Best Wasserstein: {event['wass_mean'][wass_min_idx]:.2f} "
                      f"at T1={event['T1_fine'][wass_min_idx[1]]:.4f}, "
                      f"T2={event['T2_fine'][wass_min_idx[0]]:.4f}")

    # Save final NPZ
    npz_path = os.path.join(args.output_dir, f"{pf1}_{pf2}_re2d.npz")
    np.savez_compressed(npz_path, **npz_data)
    print(f"\nResults saved to {npz_path}")
    print(f"  NPZ keys: {list(npz_data.keys())}")

    # Clean up checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"Checkpoint removed: {ckpt_path}")

    elapsed_total = time.time() - t_total
    print(f"\nTotal time: {elapsed_total:.1f}s ({elapsed_total/3600:.2f}h)")


if __name__ == '__main__':
    main()
