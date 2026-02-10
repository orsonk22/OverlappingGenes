"""
PCA Worker Module for Multiprocessing

Worker function for parallel overlapping sequence generation.
In a separate .py file because Windows uses 'spawn' for multiprocessing,
which requires functions to be importable from a module.
"""

import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'GA'))
import overlappingGenes as og


def generate_one(args):
    """
    Generate one overlapping sequence pair for a given overlap level.

    Args:
        args: tuple of (overlap, seed, prot1_len, prot2_len, len_1_n, len_2_n,
               Jvec_1, hvec_1, Jvec_2, hvec_2, T1, T2, mc_iter,
               mean_e1, mean_e2, std_e1, std_e2)

    Returns:
        (overlap, aa1_str, aa2_str)
    """
    (overlap, seed, prot1_len, prot2_len, len_1_n, len_2_n,
     Jvec_1, hvec_1, Jvec_2, hvec_2, T1, T2, mc_iter,
     mean_e1, mean_e2, std_e1, std_e2) = args

    np.random.seed(seed)

    # Generate initial sequence
    initial_seq = og.initial_seq_no_stops(prot1_len, prot2_len, overlap, quiet=True)

    # Run MC relaxation
    result = og.overlapped_sequence_generator_int(
        (Jvec_1, hvec_1), (Jvec_2, hvec_2), initial_seq,
        T1=T1, T2=T2,
        numberofiterations=mc_iter,
        quiet=True,
        whentosave=50.0,
        nat_mean1=mean_e1, nat_mean2=mean_e2,
        nat_std1=std_e1, nat_std2=std_e2,
        use_z_score=True
    )

    # Extract best nucleotide sequence
    best_nt_seq = result[6]  # best_seq_str

    # Convert to AA sequences
    aa1, aa2 = og.split_sequence_and_to_aa(list(best_nt_seq), len_1_n, len_2_n)

    # Remove stop codon (last character) and join to string
    aa1_str = ''.join(aa1[:-1])
    aa2_str = ''.join(aa2[:-1])

    return overlap, aa1_str, aa2_str
