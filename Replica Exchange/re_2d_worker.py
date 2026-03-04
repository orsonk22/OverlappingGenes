"""
Worker function for 2D Replica Exchange multiprocessing.

Must live in a separate .py file (not notebook cells) due to Windows
spawn-based multiprocessing requiring picklable imports.

Usage: imported by re_2d_cluster.py — not run directly.
"""

import os
import sys
import numpy as np
import time

# Ensure overlappingGenes is importable (needed for spawn-based multiprocessing)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Module-level globals set by init_worker
_Jvec1 = None
_hvec1 = None
_Jvec2 = None
_hvec2 = None
_prot1_len = None
_prot2_len = None


def init_worker(shared_data):
    """Pool initializer: set module-level globals from shared data dict."""
    global _Jvec1, _hvec1, _Jvec2, _hvec2, _prot1_len, _prot2_len
    _Jvec1 = shared_data['Jvec1']
    _hvec1 = shared_data['hvec1']
    _Jvec2 = shared_data['Jvec2']
    _hvec2 = shared_data['hvec2']
    _prot1_len = shared_data['prot1_len']
    _prot2_len = shared_data['prot2_len']


def run_single_stagger(args):
    """
    Run one stagger configuration of the 2D Replica Exchange sampler.

    Args:
        args: tuple of (repeat, a, b, seed, overlap_len, T1_sub, T2_sub,
              n_burnin, n_samples, sample_interval, swap_interval,
              nat_std1, nat_std2)

    Returns:
        dict with E1_grid, E2_grid, acceptance_grid, swap_rates_h/v,
        elapsed_time, repeat, a, b
    """
    from overlappingGenes import (
        re_2d_equilibrium_sampler, initial_seq_no_stops, seq_str_to_int_array
    )

    (repeat, a, b, seed, overlap_len, T1_sub, T2_sub,
     n_burnin, n_samples, sample_interval, swap_interval,
     nat_std1, nat_std2) = args

    n_T2 = len(T2_sub)
    n_T1 = len(T1_sub)
    n_total = n_T2 * n_T1

    # Compute sequence length from DCA params
    len_seq_1_n = int(3 * len(_hvec1) / 21 + 3)
    len_seq_2_n = int(3 * len(_hvec2) / 21 + 3)
    seq_length = len_seq_1_n + len_seq_2_n - overlap_len

    # Generate initial sequences (replicates setup from run_2d_replica_exchange)
    np.random.seed(seed)
    initial_seqs = np.empty((n_total, seq_length), dtype=np.uint8)
    for idx in range(n_total):
        s = initial_seq_no_stops(_prot1_len, _prot2_len, overlap_len, quiet=True)
        initial_seqs[idx] = seq_str_to_int_array(s)

    # Ensure float64
    Jvec1 = np.asarray(_Jvec1, dtype=np.float64)
    hvec1 = np.asarray(_hvec1, dtype=np.float64)
    Jvec2 = np.asarray(_Jvec2, dtype=np.float64)
    hvec2 = np.asarray(_hvec2, dtype=np.float64)
    T1_grid_f = np.asarray(T1_sub, dtype=np.float64)
    T2_grid_f = np.asarray(T2_sub, dtype=np.float64)

    # Dummy progress array (not monitored in workers)
    progress = np.zeros(1, dtype=np.int64)

    t0 = time.time()

    (E1_flat, E2_flat, acc_rates,
     swap_acc_h, swap_att_h,
     swap_acc_v, swap_att_v) = re_2d_equilibrium_sampler(
        Jvec1, hvec1, Jvec2, hvec2,
        initial_seqs,
        T1_grid_f, T2_grid_f,
        n_burnin, n_samples, sample_interval,
        swap_interval,
        float(nat_std1), float(nat_std2),
        progress
    )

    elapsed = time.time() - t0
    actual_samples = E1_flat.shape[1]

    # Reshape flat (n_total, n_samples) -> (n_T2, n_T1, n_samples)
    E1_grid = E1_flat.reshape(n_T2, n_T1, actual_samples)
    E2_grid = E2_flat.reshape(n_T2, n_T1, actual_samples)
    acc_grid = acc_rates.reshape(n_T2, n_T1)

    # Swap rates
    swap_rates_h = np.zeros_like(swap_acc_h)
    mask_h = swap_att_h > 0
    swap_rates_h[mask_h] = swap_acc_h[mask_h] / swap_att_h[mask_h]

    swap_rates_v = np.zeros_like(swap_acc_v)
    mask_v = swap_att_v > 0
    swap_rates_v[mask_v] = swap_acc_v[mask_v] / swap_att_v[mask_v]

    return {
        'E1_grid': E1_grid,
        'E2_grid': E2_grid,
        'acceptance_grid': acc_grid,
        'swap_rates_h': swap_rates_h,
        'swap_rates_v': swap_rates_v,
        'elapsed_time': elapsed,
        'repeat': repeat,
        'a': a,
        'b': b,
    }
