"""
Cluster-optimized version of overlappingGenes module.

Optimizations:
- Numba JIT compilation for hot paths
- Pre-allocated arrays instead of list.append()
- Integer-based sequence representation throughout
- Minimized Python object creation in MC loop

Author: Based on overlappingGenesoriginal.py by Kabir Husain and Nicole Wood
"""

import numpy as np
from numba import jit, njit, prange
import os

# =============================================================================
# LOOKUP TABLES (Numba-compatible)
# =============================================================================

# Nucleotide encoding: A=0, C=1, G=2, T=3
NUC_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
INT_TO_NUC = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

# Standard codon table
CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

# Build integer codon table: CODON_TABLE_INT[a][b][c] = amino acid index (0-20, 21=stop)
# Amino acid encoding: gap=0, then alphabetical A=1, C=2, D=3, ... Y=20, stop=21
AA_TO_INT = {'-': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
             'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14,
             'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, '*': 21}

CODON_TABLE_INT = np.zeros((4, 4, 4), dtype=np.int8)
for codon, aa in CODON_TABLE.items():
    i, j, k = NUC_TO_INT[codon[0]], NUC_TO_INT[codon[1]], NUC_TO_INT[codon[2]]
    CODON_TABLE_INT[i, j, k] = AA_TO_INT[aa]

# Complement table for reverse complement: A<->T (0<->3), C<->G (1<->2)
COMPLEMENT_INT = np.array([3, 2, 1, 0], dtype=np.uint8)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_natural_energies(filename):
    """Load natural energies from file (one energy per line)."""
    with open(filename, "r") as f:
        energies = [float(line.strip()) for line in f]
    return np.array(energies)


def extract_params(params_file):
    """
    Extract J and h parameters from bmDCA parameter file.

    Returns:
        J_data: numpy array of coupling parameters
        h_data: numpy array of field parameters
    """
    J_lines = []
    h_lines = []
    j_section = True

    with open(params_file, 'r') as file:
        for line in file:
            parts = line.split()
            if not parts:
                continue
            if j_section and len(parts) == 6:
                J_lines.append(float(parts[5]))
            elif len(parts) == 4:
                j_section = False
                h_lines.append(float(parts[3]))
            elif not j_section and len(parts) >= 3:
                h_lines.append(float(parts[3]))

    return np.array(J_lines), np.array(h_lines)


# =============================================================================
# NUMBA-OPTIMIZED CORE FUNCTIONS
# =============================================================================

@njit(cache=True)
def seq_str_to_int_array(seq_str):
    """Convert string sequence to integer array."""
    n = len(seq_str)
    arr = np.empty(n, dtype=np.uint8)
    for i in range(n):
        c = seq_str[i]
        if c == 'A':
            arr[i] = 0
        elif c == 'C':
            arr[i] = 1
        elif c == 'G':
            arr[i] = 2
        else:  # T
            arr[i] = 3
    return arr


@njit(cache=True)
def int_array_to_seq_str(arr):
    """Convert integer array back to string sequence."""
    chars = ['A', 'C', 'G', 'T']
    result = ''
    for i in range(len(arr)):
        result += chars[arr[i]]
    return result


@njit(cache=True)
def reverse_complement_int(seq):
    """Compute reverse complement of integer-encoded sequence."""
    n = len(seq)
    rc = np.empty(n, dtype=np.uint8)
    for i in range(n):
        rc[i] = COMPLEMENT_INT[seq[n - 1 - i]]
    return rc


@njit(cache=True)
def translate_to_aa_int(seq, codon_table):
    """Translate nucleotide sequence to amino acid indices."""
    n_codons = len(seq) // 3
    aa_seq = np.empty(n_codons, dtype=np.int8)
    for i in range(n_codons):
        a, b, c = seq[3*i], seq[3*i+1], seq[3*i+2]
        aa_seq[i] = codon_table[a, b, c]
    return aa_seq


@njit(cache=True)
def has_internal_stop(aa_seq):
    """Check if sequence has internal stop codons (21 = stop)."""
    n = len(aa_seq)
    for i in range(n - 1):  # Exclude last position (should be stop)
        if aa_seq[i] == 21:
            return True
    return False


@njit(cache=True)
def ends_with_stop(aa_seq):
    """Check if sequence ends with stop codon."""
    return aa_seq[-1] == 21


@njit(cache=True)
def calculate_energy(aa_seq, Jvec, hvec):
    """
    Calculate DCA energy for an amino acid sequence.

    Args:
        aa_seq: Integer-encoded amino acid sequence (excluding stop codon)
        Jvec: Coupling parameters (flattened)
        hvec: Field parameters (flattened)

    Returns:
        energy: DCA energy (negative = more stable)
    """
    energy = 0.0
    seqL = len(hvec) // 21
    Jfirstterm = 0

    for i in range(seqL):
        a = aa_seq[i]

        # Add field energy
        hlookup = 21 * i + a
        energy += hvec[hlookup]

        # Add coupling energies
        for j in range(i + 1, seqL):
            b = aa_seq[j]
            Jlookup = Jfirstterm + 21 * 21 * (j - i - 1) + 21 * a + b
            energy += Jvec[Jlookup]

        Jfirstterm += (seqL - i - 1) * 21 * 21

    return -energy


@njit(cache=True)
def mutate_sequence(seq, seq_len):
    """
    Mutate a single random nucleotide.

    Returns:
        new_seq: Mutated sequence
        pos: Position that was mutated
        new_nuc: New nucleotide value
    """
    pos = np.random.randint(0, seq_len)
    current = seq[pos]

    # Pick a different nucleotide
    new_nuc = np.random.randint(0, 3)
    if new_nuc >= current:
        new_nuc += 1

    new_seq = seq.copy()
    new_seq[pos] = new_nuc
    return new_seq, pos, new_nuc


@njit(cache=True)
def run_mc_simulation(
    seq,
    len_seq_1_n,
    len_seq_2_n,
    Jvec1, hvec1,
    Jvec2, hvec2,
    T1, T2,
    n_iterations,
    save_interval,
    codon_table
):
    """
    Run Monte Carlo simulation for overlapping gene optimization.

    This is the core MC loop, fully compiled with Numba for speed.

    Args:
        seq: Initial sequence (integer-encoded nucleotides)
        len_seq_1_n: Length of gene 1 in nucleotides
        len_seq_2_n: Length of gene 2 in nucleotides
        Jvec1, hvec1: DCA parameters for gene 1
        Jvec2, hvec2: DCA parameters for gene 2
        T1, T2: Temperatures for genes 1 and 2
        n_iterations: Number of MC iterations
        save_interval: How often to save energy history
        codon_table: Codon lookup table

    Returns:
        final_seq: Final sequence
        E1_final, E2_final: Final energies
        E1_history, E2_history: Energy trajectories
        accepted, prob_accepted, rejected: Acceptance statistics
    """
    seq_len = len(seq)

    # Pre-allocate history arrays
    n_saves = int(n_iterations / save_interval) + 2
    E1_history = np.empty(n_saves, dtype=np.float64)
    E2_history = np.empty(n_saves, dtype=np.float64)
    history_idx = 0

    # Calculate initial energies
    aa1 = translate_to_aa_int(seq[:len_seq_1_n], codon_table)
    rc_seq = reverse_complement_int(seq[-len_seq_2_n:])
    aa2 = translate_to_aa_int(rc_seq, codon_table)

    E1 = calculate_energy(aa1[:-1], Jvec1, hvec1)  # Exclude stop codon
    E2 = calculate_energy(aa2[:-1], Jvec2, hvec2)

    E1_history[history_idx] = E1
    E2_history[history_idx] = E2
    history_idx += 1

    # Counters
    accepted = 0
    prob_accepted = 0
    rejected = 0

    next_save = save_interval

    for iteration in range(1, n_iterations + 1):
        # Save history at intervals
        if iteration >= next_save:
            E1_history[history_idx] = E1
            E2_history[history_idx] = E2
            history_idx += 1
            next_save += save_interval

        # Propose mutation
        new_seq, _, _ = mutate_sequence(seq, seq_len)

        # Translate new sequences
        new_aa1 = translate_to_aa_int(new_seq[:len_seq_1_n], codon_table)
        new_rc_seq = reverse_complement_int(new_seq[-len_seq_2_n:])
        new_aa2 = translate_to_aa_int(new_rc_seq, codon_table)

        # Check for stop codon violations
        if has_internal_stop(new_aa1) or has_internal_stop(new_aa2):
            continue
        if not ends_with_stop(new_aa1) or not ends_with_stop(new_aa2):
            continue

        # Calculate new energies
        E1_new = calculate_energy(new_aa1[:-1], Jvec1, hvec1)
        E2_new = calculate_energy(new_aa2[:-1], Jvec2, hvec2)

        # Metropolis criterion
        delta_H1 = E1_new - E1
        delta_H2 = E2_new - E2
        delta_H = (delta_H1 / T1) + (delta_H2 / T2)

        if delta_H <= 0:
            # Accept
            seq = new_seq
            E1 = E1_new
            E2 = E2_new
            accepted += 1
        else:
            # Probabilistic acceptance
            if np.random.random() < np.exp(-delta_H):
                seq = new_seq
                E1 = E1_new
                E2 = E2_new
                prob_accepted += 1
            else:
                rejected += 1

    # Save final state
    E1_history[history_idx] = E1
    E2_history[history_idx] = E2
    history_idx += 1

    return seq, E1, E2, E1_history[:history_idx], E2_history[:history_idx], accepted, prob_accepted, rejected


# =============================================================================
# INITIAL SEQUENCE GENERATION (Python, not speed-critical)
# =============================================================================

def initial_seq_no_stops(prot1, prot2, overlap, quiet=True):
    """
    Generate initial sequence with proper stop codon placement.

    Args:
        prot1: Length of protein 1 in amino acids (without stop)
        prot2: Length of protein 2 in amino acids (without stop)
        overlap: Overlap in nucleotides (must be >= 6)
        quiet: Suppress output

    Returns:
        seq: Nucleotide sequence string
    """
    def revcomp(seq):
        return seq.replace("A", "t").replace("T", "a").replace("C", "g").replace("G", "c")[::-1].upper()

    def randnt():
        return rng.choice(nts)

    nts = ["A", "T", "G", "C"]
    stopcodons = ["TAA", "TAG", "TGA"]
    rng = np.random.default_rng()

    codonsNoStop = ["".join([n1, n2, n3])
                    for n1 in nts for n2 in nts for n3 in nts
                    if "".join([n1, n2, n3]) not in stopcodons]

    codonsNoStopEitherFrame = [a for a in codonsNoStop if revcomp(a) not in stopcodons]

    codonsNoStopNoA1 = ["".join([n1, n2, n3])
                        for n1 in ["T", "G", "C"] for n2 in nts for n3 in nts
                        if "".join([n1, n2, n3]) not in stopcodons]

    noTAs = [codon for codon in codonsNoStop if codon[:2] != "TA"]
    noCAsOrTAs = [codon for codon in codonsNoStop if codon[:2] != "CA" and codon[:2] != "TA"]

    codonChain12 = {}
    for n1 in nts:
        for n2 in nts:
            for n3 in nts:
                thiscodon = n1 + n2 + n3
                if n3 == "T":
                    codonChain12[thiscodon] = noCAsOrTAs
                elif n3 == "C":
                    codonChain12[thiscodon] = noTAs
                else:
                    codonChain12[thiscodon] = codonsNoStop

    # Lengths in nucleotides
    l1 = 3 * prot1 + 3
    l2 = 3 * prot2 + 3

    # Determine reading frame
    if (l1 - overlap) % 3 == 0:
        readingframe = "3-0"
    elif (l1 - overlap - 1) % 3 == 0:
        readingframe = "2-1"
    else:
        readingframe = "1-2"

    if not quiet:
        print(f"Reading frame: {readingframe}")

    # Build sequence
    beforeoverlap = int(np.floor((l1 - overlap) / 3))
    seq = "".join(rng.choice(codonsNoStop, beforeoverlap))

    # Next two codons (stop codon region)
    if readingframe == "3-0":
        next2 = revcomp(rng.choice(stopcodons)) + rng.choice(codonsNoStopEitherFrame)
    elif readingframe == "2-1":
        next2 = randnt() + rng.choice(["TT", "CT", "TC"]) + "A" + randnt() + randnt()
    else:  # 1-2
        next2 = randnt() + randnt() + rng.choice(["TT", "CT", "TC"]) + "A"
        if next2[3] == "T":
            next2 += rng.choice(["T", "C"])
        else:
            next2 += randnt()
    seq += next2

    # Middle of overlap
    lengthmidoverlap = int((l1 - 3 * beforeoverlap - 12) / 3)

    if readingframe == "3-0":
        seq += "".join(rng.choice(codonsNoStopEitherFrame, lengthmidoverlap))
    elif readingframe == "2-1":
        for _ in range(lengthmidoverlap):
            if seq[-2:] in ["TC", "CT", "TT"]:
                seq += rng.choice(codonsNoStopNoA1)
            else:
                seq += rng.choice(codonsNoStop)
    else:  # 1-2
        for _ in range(lengthmidoverlap):
            seq += rng.choice(codonChain12[seq[-3:]])

    # Last two codons
    if readingframe == "3-0":
        nexttwocodons = rng.choice(codonsNoStopEitherFrame) + rng.choice(stopcodons)
    elif readingframe == "2-1":
        if seq[-2:] in ["TC", "CT", "TT"]:
            nexttwocodons = rng.choice(codonsNoStopNoA1) + rng.choice(stopcodons)
        else:
            nexttwocodons = rng.choice(codonsNoStop) + rng.choice(stopcodons)
    else:  # 1-2
        nexttwocodons = rng.choice(codonChain12[seq[-3:]])
        if nexttwocodons[-1] in ["C", "T"]:
            nexttwocodons += "TGA"
        else:
            nexttwocodons += rng.choice(stopcodons)
    seq += nexttwocodons

    # Fill remainder
    if readingframe == "2-1":
        seq += randnt()
    elif readingframe == "1-2":
        seq += randnt() + randnt()

    remainingcodons = int(np.floor((l2 - overlap) / 3))
    seq += revcomp("".join(rng.choice(codonsNoStop, remainingcodons)))

    return seq


# =============================================================================
# HIGH-LEVEL SIMULATION WRAPPER
# =============================================================================

def run_overlap_simulation(
    params1, params2,
    nat_mean1, nat_std1,
    nat_mean2, nat_std2,
    overlap,
    T1=1.0, T2=1.0,
    n_iterations=250000,
    whentosave=0.01
):
    """
    Run a single overlap simulation and compute metrics.

    Args:
        params1: (J, h) tuple for protein 1
        params2: (J, h) tuple for protein 2
        nat_mean1, nat_std1: Natural energy statistics for protein 1
        nat_mean2, nat_std2: Natural energy statistics for protein 2
        overlap: Overlap in nucleotides
        T1, T2: Temperatures
        n_iterations: MC iterations
        whentosave: Fraction of iterations for history saves

    Returns:
        dict with energies and convergence metrics
    """
    Jvec1, hvec1 = params1
    Jvec2, hvec2 = params2

    len1 = len(hvec1) // 21  # Protein length in AA
    len2 = len(hvec2) // 21

    len_seq_1_n = 3 * len1 + 3  # Nucleotide length including stop
    len_seq_2_n = 3 * len2 + 3

    # Generate initial sequence
    init_seq = initial_seq_no_stops(len1, len2, overlap, quiet=True)
    seq = seq_str_to_int_array(init_seq)

    # Calculate save interval
    save_interval = int(n_iterations * whentosave)
    if save_interval < 1:
        save_interval = 1

    # Run MC simulation
    final_seq, E1, E2, E1_hist, E2_hist, accepted, prob_accepted, rejected = run_mc_simulation(
        seq, len_seq_1_n, len_seq_2_n,
        Jvec1, hvec1, Jvec2, hvec2,
        T1, T2,
        n_iterations, save_interval,
        CODON_TABLE_INT
    )

    # Compute metrics
    dists1 = np.abs(E1_hist - nat_mean1)
    dists2 = np.abs(E2_hist - nat_mean2)

    min_dist1 = np.min(dists1)
    min_dist2 = np.min(dists2)

    # Iterations to convergence (within 1 std)
    step_size = save_interval
    conv_idx1 = np.where(dists1 <= nat_std1)[0]
    conv_idx2 = np.where(dists2 <= nat_std2)[0]

    iter_conv1 = (conv_idx1[0] + 1) * step_size if len(conv_idx1) > 0 else n_iterations
    iter_conv2 = (conv_idx2[0] + 1) * step_size if len(conv_idx2) > 0 else n_iterations

    return {
        'E1_final': E1,
        'E2_final': E2,
        'min_dist1': min_dist1,
        'min_dist2': min_dist2,
        'iter_conv1': iter_conv1,
        'iter_conv2': iter_conv2,
        'accepted': accepted,
        'prob_accepted': prob_accepted,
        'rejected': rejected
    }
