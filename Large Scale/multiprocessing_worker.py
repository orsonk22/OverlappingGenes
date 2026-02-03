"""
Worker functions for multiprocessing in Jupyter notebooks.
Must be in a separate file for Windows compatibility (spawn method).

On Windows, multiprocessing uses 'spawn' which starts fresh Python processes.
These processes need to import all functions, but functions defined in
Jupyter notebook cells cannot be pickled/imported. Moving worker functions
to this module solves the issue.
"""
import numpy as np
import sys
import os

# Ensure overlappingGenes can be imported from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import overlappingGenes as og


def process_single_pair(pair_args):
    """
    Worker function that processes an entire protein family pair.
    Runs all overlaps and all trials for one pair.

    This function MUST be in a separate .py file (not notebook cell)
    for Windows multiprocessing compatibility.

    Parameters:
    -----------
    pair_args : tuple
        Contains: (pf1, pf2, params1, params2, nat_mean1, nat_mean2,
                   nat_std1, nat_std2, t1, t2, valid_overlaps, len1, len2,
                   n_trials, iterations, whentosave, z_score)

    Returns:
    --------
    list of dict : Results for all overlaps for this pair
    """
    (pf1, pf2, params1, params2, nat_mean1, nat_mean2, nat_std1, nat_std2,
     t1, t2, valid_overlaps, len1, len2, n_trials, iterations, whentosave, z_score) = pair_args

    pair_results = []

    for ov in valid_overlaps:
        trial_e1_final = []
        trial_e2_final = []
        trial_e1_min_dist = []
        trial_e2_min_dist = []
        trial_iter_conv_1 = []
        trial_iter_conv_2 = []

        for trial_idx in range(n_trials):
            try:
                # Generate initial sequence without stop codons
                init_seq = og.initial_seq_no_stops(len1, len2, ov, quiet=True)

                # Run Monte Carlo simulation
                result = og.overlapped_sequence_generator_int(
                    params1, params2, init_seq,
                    numberofiterations=iterations,
                    whentosave=whentosave,
                    quiet=True,
                    T1=t1, T2=t2,
                    nat_mean1=nat_mean1,
                    nat_mean2=nat_mean2,
                    nat_std1=nat_std1,
                    nat_std2=nat_std2,
                    use_z_score=z_score
                )

                hist1 = result[2]
                hist2 = result[3]
                final_energies = result[4]
                e1, e2 = final_energies[0], final_energies[1]

                trial_e1_final.append(e1)
                trial_e2_final.append(e2)

                # Calculate distance metrics
                if z_score:
                    dists1 = np.abs(hist1 - nat_mean1) / nat_std1
                    dists2 = np.abs(hist2 - nat_mean2) / nat_std2
                else:
                    dists1 = np.abs(hist1 - nat_mean1)
                    dists2 = np.abs(hist2 - nat_mean2)

                # Minimum distance achieved during simulation
                if z_score:
                    min_d1 = np.min(dists1) if len(dists1) > 0 else abs(e1 - nat_mean1) / nat_std1
                    min_d2 = np.min(dists2) if len(dists2) > 0 else abs(e2 - nat_mean2) / nat_std2
                else:
                    min_d1 = np.min(dists1) if len(dists1) > 0 else abs(e1 - nat_mean1)
                    min_d2 = np.min(dists2) if len(dists2) > 0 else abs(e2 - nat_mean2)

                trial_e1_min_dist.append(min_d1)
                trial_e2_min_dist.append(min_d2)

                # Convergence iteration (when within 1 std dev of natural mean)
                step_size = iterations * whentosave

                if z_score:
                    conv_idxs_1 = np.where(dists1 <= 1.0)[0]
                else:
                    conv_idxs_1 = np.where(dists1 <= nat_std1)[0]
                iter_conv_1 = (conv_idxs_1[0] + 1) * step_size if len(conv_idxs_1) > 0 else iterations

                if z_score:
                    conv_idxs_2 = np.where(dists2 <= 1.0)[0]
                else:
                    conv_idxs_2 = np.where(dists2 <= nat_std2)[0]
                iter_conv_2 = (conv_idxs_2[0] + 1) * step_size if len(conv_idxs_2) > 0 else iterations

                trial_iter_conv_1.append(iter_conv_1)
                trial_iter_conv_2.append(iter_conv_2)

            except Exception as ex:
                # Skip failed trials silently
                continue

        # Compute summary statistics for this overlap
        if len(trial_e1_final) > 0:
            if z_score:
                dist_1 = np.mean(np.abs(np.array(trial_e1_final) - nat_mean1) / nat_std1)
                dist_2 = np.mean(np.abs(np.array(trial_e2_final) - nat_mean2) / nat_std2)
            else:
                dist_1 = np.mean(np.abs(np.array(trial_e1_final) - nat_mean1))
                dist_2 = np.mean(np.abs(np.array(trial_e2_final) - nat_mean2))

            pair_results.append({
                'PF1': pf1, 'PF2': pf2, 'Overlap': ov,
                'Reading_Frame': ov % 3,
                'Mean_E1': np.mean(trial_e1_final), 'Std_E1': np.std(trial_e1_final),
                'Mean_E2': np.mean(trial_e2_final), 'Std_E2': np.std(trial_e2_final),
                'Nat_Mean1': nat_mean1, 'Nat_Mean2': nat_mean2,
                'Nat_Std1': nat_std1, 'Nat_Std2': nat_std2,
                'Dist_1': dist_1,
                'Dist_2': dist_2,
                'Min_Dist_1': np.mean(trial_e1_min_dist),
                'Min_Dist_2': np.mean(trial_e2_min_dist),
                'Iter_Converged_1': np.mean(trial_iter_conv_1),
                'Iter_Converged_2': np.mean(trial_iter_conv_2)
            })

    return pair_results
