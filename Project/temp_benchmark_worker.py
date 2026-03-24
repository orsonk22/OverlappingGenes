"""
Worker for parallelized temperature benchmark sweeps.
Must be in a separate .py file for Windows multiprocessing (spawn method).

Uses an initializer pattern so DCA parameter matrices are sent once per
worker process instead of once per task (avoids ~84 MB of pickling per task).
"""
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import overlappingGenes as og

# Global worker state — set once per worker via Pool initializer
_dca1 = None
_dca2 = None
_p1len = None
_p2len = None


def init_worker(dca_params_1, dca_params_2, prot1_len, prot2_len):
    global _dca1, _dca2, _p1len, _p2len
    _dca1 = dca_params_1
    _dca2 = dca_params_2
    _p1len = prot1_len
    _p2len = prot2_len


def run_single_trial(args):
    """Run one MC trial. Returns (temp_value, final_energy)."""
    overlap, T1, T2, n_iters, temp_value, protein_index = args
    initial_seq = og.initial_seq_no_stops(_p1len, _p2len, overlap, quiet=True)
    result = og.overlapped_sequence_generator_int(
        _dca1, _dca2, initial_seq,
        numberofiterations=n_iters,
        whentosave=100.0,
        quiet=True,
        T1=T1, T2=T2
    )
    final_Es = result[4]
    return (temp_value, final_Es[protein_index])


def run_trace_trial(args):
    """Run one MC trial and return energy histories for trace plotting."""
    overlap, T1, T2, n_iters, whentosave = args
    initial_seq = og.initial_seq_no_stops(_p1len, _p2len, overlap, quiet=True)
    result = og.overlapped_sequence_generator_int(
        _dca1, _dca2, initial_seq,
        numberofiterations=n_iters,
        whentosave=whentosave,
        quiet=True,
        T1=T1, T2=T2
    )
    return (result[2], result[3])
