"""
Worker functions for temperature optimization via multiprocessing.
Must be in a separate file for Windows compatibility (spawn method).

For each protein family, finds the Metropolis temperature that minimizes
the mean energy distance from the natural energy distribution mean.

Approach: pair each family with a dummy partner at T=1000 (unconstrained),
then scan temperatures for the target family using a two-phase grid search
(coarse then fine). Uses a JIT-compiled stripped-down MC simulation.
"""
import numpy as np
import sys
import os
from numba import njit

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import overlappingGenes as og
from overlappingGenes import (
    calculate_Energy, calculate_Delta_Energy,
    split_sequence_and_to_numeric_out, seq_str_to_int_array
)


@njit
def mc_trial_min_distance(Jvec1, hvec1, Jvec2, hvec2, seq_int,
                          T1, T2, n_iterations, nat_mean1):
    """
    JIT-compiled stripped-down MC simulation that tracks the minimum
    distance from the natural mean for protein 1 across the entire run.

    Protein 2 is assumed unconstrained (T2 >> 1), so we only care about
    protein 1's energy distance.

    Parameters
    ----------
    Jvec1, hvec1 : DCA coupling and field parameters for protein 1
    Jvec2, hvec2 : DCA coupling and field parameters for protein 2
    seq_int : Initial nucleotide sequence as uint8 array (0=A,1=C,2=G,3=T)
    T1, T2 : Metropolis temperatures for protein 1 and 2
    n_iterations : MC iterations (stop codon rejections not counted)
    nat_mean1 : Natural mean energy for protein 1

    Returns
    -------
    min_dist : Minimum |E1 - nat_mean1| observed at any point during the run
    """
    seq = seq_int.copy()
    sequence_L = len(seq)

    len_seq_1_n = int(3 * len(hvec1) / 21 + 3)
    len_seq_2_n = int(3 * len(hvec2) / 21 + 3)
    len_aa_1 = len_seq_1_n // 3
    len_aa_2 = len_seq_2_n // 3

    aa_seq_1 = np.empty(len_aa_1, dtype=np.int32)
    aa_seq_2 = np.empty(len_aa_2, dtype=np.int32)
    rc_buffer = np.empty(len_seq_2_n, dtype=np.uint8)
    aa_seq_1_new = np.empty(len_aa_1, dtype=np.int32)
    aa_seq_2_new = np.empty(len_aa_2, dtype=np.int32)

    split_sequence_and_to_numeric_out(seq, len_seq_1_n, len_seq_2_n,
                                      aa_seq_1, aa_seq_2, rc_buffer)
    E1 = calculate_Energy(aa_seq_1[:-1], Jvec1, hvec1)
    E2 = calculate_Energy(aa_seq_2[:-1], Jvec2, hvec2)

    # Track minimum distance from natural mean throughout the run
    min_dist = abs(E1 - nat_mean1)

    itera = 0
    while itera < n_iterations:
        new_position = np.random.randint(0, sequence_L)
        old_nt = seq[new_position]
        idx = np.random.randint(0, 3)
        if idx >= old_nt:
            idx += 1
        seq[new_position] = idx

        split_sequence_and_to_numeric_out(seq, len_seq_1_n, len_seq_2_n,
                                          aa_seq_1_new, aa_seq_2_new, rc_buffer)

        # Stop codon check
        stop_err = False
        if aa_seq_1_new[len_aa_1 - 1] != 21 or aa_seq_2_new[len_aa_2 - 1] != 21:
            stop_err = True
        else:
            for i in range(len_aa_1 - 1):
                if aa_seq_1_new[i] == 21:
                    stop_err = True
                    break
            if not stop_err:
                for i in range(len_aa_2 - 1):
                    if aa_seq_2_new[i] == 21:
                        stop_err = True
                        break

        if stop_err:
            seq[new_position] = old_nt
            continue

        # Delta energy
        dE1 = 0.0
        dE2 = 0.0

        aa_pos_1 = -1
        for i in range(len_aa_1 - 1):
            if aa_seq_1[i] != aa_seq_1_new[i]:
                aa_pos_1 = i
                break
        if aa_pos_1 != -1:
            dE1 = calculate_Delta_Energy(aa_seq_1, Jvec1, hvec1,
                                         aa_pos_1, aa_seq_1_new[aa_pos_1])

        aa_pos_2 = -1
        for i in range(len_aa_2 - 1):
            if aa_seq_2[i] != aa_seq_2_new[i]:
                aa_pos_2 = i
                break
        if aa_pos_2 != -1:
            dE2 = calculate_Delta_Energy(aa_seq_2, Jvec2, hvec2,
                                         aa_pos_2, aa_seq_2_new[aa_pos_2])

        # Metropolis acceptance
        delta_H = (dE1 / T1) + (dE2 / T2)

        if delta_H <= 0 or np.random.rand() < np.exp(-delta_H):
            for i in range(len_aa_1):
                aa_seq_1[i] = aa_seq_1_new[i]
            for i in range(len_aa_2):
                aa_seq_2[i] = aa_seq_2_new[i]
            E1 += dE1
            E2 += dE2

            # Update minimum distance on every accepted move
            dist = abs(E1 - nat_mean1)
            if dist < min_dist:
                min_dist = dist
        else:
            seq[new_position] = old_nt

        itera += 1

    return min_dist


def optimize_single_family(args):
    """
    Worker: find optimal MC temperature for one protein family.

    Two-phase grid search:
    1. Coarse scan over wide temperature range
    2. Fine scan centered on the best coarse temperature

    The target family is protein 1, with a dummy partner at T=1000.

    Parameters
    ----------
    args : tuple
        (pf_name, pf_Jvec, pf_hvec, pf_nat_mean, pf_nat_std,
         partner_Jvec, partner_hvec, partner_len, overlap,
         coarse_temps, n_trials_coarse, n_trials_fine,
         fine_n_temps, fine_half_width, mc_iterations)

    Returns
    -------
    dict with optimization results
    """
    (pf_name, pf_Jvec, pf_hvec, pf_nat_mean, pf_nat_std,
     partner_Jvec, partner_hvec, partner_len,
     overlap, coarse_temps, n_trials_coarse, n_trials_fine,
     fine_n_temps, fine_half_width, mc_iterations) = args

    pf_len = int(len(pf_hvec) / 21)
    T_partner = 1000.0

    # === Phase 1: Coarse Scan ===
    coarse_results = []
    for T in coarse_temps:
        min_dists = []
        for _ in range(n_trials_coarse):
            try:
                seq_str = og.initial_seq_no_stops(pf_len, partner_len,
                                                  overlap, quiet=True)
                seq_int = seq_str_to_int_array(seq_str)
                md = mc_trial_min_distance(
                    pf_Jvec, pf_hvec, partner_Jvec, partner_hvec,
                    seq_int, T, T_partner, mc_iterations, pf_nat_mean
                )
                min_dists.append(md)
            except Exception:
                continue

        if len(min_dists) > 0:
            coarse_results.append((
                float(T),
                float(np.mean(min_dists)),
                float(np.std(min_dists))
            ))

    # Best coarse temperature
    best_coarse_idx = int(np.argmin([r[1] for r in coarse_results]))
    best_coarse_T = coarse_results[best_coarse_idx][0]

    # === Phase 2: Fine Scan ===
    fine_lo = max(0.3, best_coarse_T - fine_half_width)
    fine_hi = best_coarse_T + fine_half_width
    fine_temps = np.linspace(fine_lo, fine_hi, fine_n_temps)

    fine_results = []
    for T in fine_temps:
        min_dists = []
        for _ in range(n_trials_fine):
            try:
                seq_str = og.initial_seq_no_stops(pf_len, partner_len,
                                                  overlap, quiet=True)
                seq_int = seq_str_to_int_array(seq_str)
                md = mc_trial_min_distance(
                    pf_Jvec, pf_hvec, partner_Jvec, partner_hvec,
                    seq_int, T, T_partner, mc_iterations, pf_nat_mean
                )
                min_dists.append(md)
            except Exception:
                continue

        if len(min_dists) > 0:
            fine_results.append((
                float(T),
                float(np.mean(min_dists)),
                float(np.std(min_dists))
            ))

    # Best fine temperature
    best_fine_idx = int(np.argmin([r[1] for r in fine_results]))
    optimal_T = fine_results[best_fine_idx][0]
    optimal_dist = fine_results[best_fine_idx][1]

    return {
        'pf_name': pf_name,
        'optimal_T': optimal_T,
        'optimal_dist': optimal_dist,
        'nat_mean': pf_nat_mean,
        'nat_std': pf_nat_std,
        'coarse_results': coarse_results,
        'fine_results': fine_results,
        'best_coarse_T': best_coarse_T,
    }
