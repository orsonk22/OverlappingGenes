"""

Functions to implement Nicole Wood's overlapping genes algorithm.

Last edit by Kabir Husain on 23/6/2025:

- Removed distance in sequence space
- Added variable to choose when to save energy history (e.g., whentosave = 0.1 means save every 10% of iterations)

"""

######## Import Functions

# Usual suspects
import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import jit, prange, njit
import time  # Add import for timing

colorTable = {}
colorTable['k'] = [0,0,0]
colorTable['g'] = [27/255,158/255,119/255]
colorTable['o'] = [217/255,95/255,2/255]

######## Data input, and converting between types

#define stop codons
amber = ['T','A','G']
amber_rc = ['C','T','A']
ochre = ['T','A','A']
ochre_rc = ['T','T','A']
opal = ['T','G','A']
opal_rc = ['T','C','A']
stops = [amber, ochre, opal]
stops_rc = [amber_rc, ochre_rc, opal_rc]

def load_natural_energies(filename):
    """
    Load the natural energies for a given protein family from a file.
    File format is expected to be a text file with one energy value per line.
    
    INPUTS:
    pfname: Protein family name (string)
    
    OUTPUTS:
    energies: List of natural energies (float)
    """
    
    with open(filename, "r") as f:
        energies = [float(line.strip()) for line in f]
    return energies

def extract_params(params_data):
    """"
    Formats the parameter data from bmDCA
    
    INPUTS: numerical_data = numerical data
            params_data = parameter data
    
    OUTPUTS: 
    J_data = J values (last column only)
    h_data = h values (last column only)
    """
    # Read file once and separate J and h parameters, extracting only the values
    J_lines = []
    h_lines = []
    j_section = True
    
    with open(params_data, 'r') as file:
        for line in file:
            parts = line.split()
            if not parts:  # Skip empty lines
                continue
                
            if j_section and len(parts) == 6:
                J_lines.append(float(parts[5]))  # Only extract the value
            elif len(parts) == 4:
                j_section = False
                h_lines.append(float(parts[3]))  # Only extract the value
            elif not j_section and len(parts) >= 3:
                h_lines.append(float(parts[3]))  # Only extract the value
    
    # Convert directly to numpy arrays without intermediate steps
    J_final = np.array(J_lines)
    h_final = np.array(h_lines)
    
    return J_final, h_final

def aa_to_n(aa_sequence):
    """
    Takes input numerical amino acid sequence and outputs it 
    into its nucleotide constituents
    """
    amino_acid = [('-','-','-'),('G','C','T'),('T','G','T'),('G','A','T'),('G','A','A'),('T','T','T'),('G','G','T'),('C','A','T'),('A','T','T'),('A','A','A'),('T','T','G'),('A','T','G'),('A','A','T'),('C','C','T'),('C','A','A'),('C','G','T'),('T','C','T'),('A','C','T'),('G','T','T'),('T','G','G'),('T','A','T')]
    n_seq = []
    for i in aa_sequence:
        n_seq.extend(amino_acid[i])
    return n_seq
    #sequence_string = "".join(n_seq)
    #return Seq(sequence_string)

def to_numeric(n_sequence):
    """
    Takes input amino acid sequence and turns it into a numeric sequence,
    n_sequence must be an array, and have no stop codons in it.
    """

    numerical_sequence = {'-': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}

    return np.array([numerical_sequence[aa] for aa in n_sequence])

# Precompute codon table as a dict
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

def fast_translate(seq):
    aa_seq = []
    for i in range(0, len(seq)-2, 3):
        codon = ''.join(seq[i:i+3])
        aa_seq.append(CODON_TABLE.get(codon, 'X'))  # 'X' for unknown
    return aa_seq

def split_sequence_and_to_aa(sequence, len_1, len_2):
    aa_sequence_1 = fast_translate(sequence[:len_1])
    rc_seq = sequence[-len_2:][::-1]  # reverse
    rc_seq = [complement_base(nt) for nt in rc_seq]
    aa_sequence_2 = fast_translate(rc_seq)
    return aa_sequence_1, aa_sequence_2

def complement_base(nt):
    return {'A':'T', 'T':'A', 'G':'C', 'C':'G'}[nt]

def fast_reverse_complement(seq):
    # seq: list or np.array of 'A','T','G','C'
    comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    # Use list comprehension for speed
    return [comp[nt] for nt in reversed(seq)]

def split_sequence_and_to_aa(sequence, len_1, len_2):
    """
    Fast version: avoids BioPython, uses fast_translate and fast_reverse_complement.
    sequence: list or np.array of nucleotides (characters)
    len_1: length in nucleotides for first protein
    len_2: length in nucleotides for second protein
    """
    aa_sequence_1 = fast_translate(sequence[:len_1])
    rc_seq = fast_reverse_complement(sequence[-len_2:])
    aa_sequence_2 = fast_translate(rc_seq)
    return aa_sequence_1, aa_sequence_2




######## Energy calculations

@jit(nopython = True)
def calculate_Energy(thisseq, Jvec, hvec):
    """
    Written by Kabir Husain and Nicole Wood

    Given an amino acid sequence converted to numerical form,
    computes its DCA energy
    """
    thisenergy = 0
    seqL = int(len(hvec)/21) # length of the sequence
    Jfirstterm = 0
    for i in np.arange(seqL):
        a = thisseq[i]

        # add the field energies
        hlookup = int(21*i + a)
        thisenergy += hvec[hlookup]

        for j in np.arange(i+1,seqL):
            b = thisseq[j]

            # add the coupling energies
            Jlookup = int(Jfirstterm + 21*21*(j-i-1) + 21*a + b)
            thisenergy += Jvec[Jlookup]

        Jfirstterm += (seqL-i-1)*21*21

    return -1*thisenergy

def calculate_energies(aa_sequence_1, aa_sequence_2, Jvec_1, hvec_1, Jvec_2, hvec_2):
    """
    Compute the energies of two sequences, and the total energy
    """
    energy_1 = calculate_Energy(to_numeric(aa_sequence_1[:-1]), Jvec_1, hvec_1)     #have to convert to numeric for the energy calculation
    energy_2 = calculate_Energy(to_numeric(aa_sequence_2[:-1]), Jvec_2, hvec_2)     #index to last one as sstop codon has no defined energy in bmDCA 
    energy_total = energy_1 + energy_2
    return energy_1, energy_2, energy_total


######## Sampling and initial conditions


#code to genetrate an initial sequence - cannot do any overlap less than 3 or overlap = 5 for some reason
def initial_seq_no_stops(prot1, prot2, overlap, quiet=False):
    """
    Function written by Kabir on 5/3/2025

    Generates a sequence that has 2 sequences,
    one of prot1 overlapped with reverse complement of prot2, 
    removes stop codons from the sequences and adds them to the end of each sequence. 

    Arguments:
    - prot1: Length of protein 1 in amino acids (without stop codon)
    - prot2: Length of protein 2 in amino acids (without stop codon)
    - overlap: Length of overlap region (in nucleotides). Must be >=6

    Returns:
    - seq: A string representing the nucleotide sequence of the two proteins

    Defintions:
    - Reading frame "3-0" is a perfect overlap
    - Reading frame "2-1" is of the type:
       AAABBB
        XXXYYY
    - Reading frame "1-2" is of the type:
       AAABBB
         YYYXXX

    """

    ## Utility functions
    def revcomp(seq):
        return seq.replace("A", "t").replace("T", "a").replace("C", "g").replace("G","c")[::-1].upper()

    def randnt():
        return rng.choice(nts)

    ### Utility variables
    nts = ["A", "T", "G", "C"]
    stopcodons = ["TAA", "TAG", "TGA"]
    rng = np.random.default_rng()

    codonsNoStop = ["".join([n1,n2,n3]) 
                    for n1 in nts 
                    for n2 in nts 
                    for n3 in nts 
                    if "".join([n1,n2,n3]) not in stopcodons]
    
    codonsNoStopEitherFrame = [a for a in codonsNoStop if revcomp(a) not in stopcodons]

    codonsNoStopNoA1 = ["".join([n1,n2,n3]) 
                    for n1 in ["T", "G", "C"] 
                    for n2 in nts 
                    for n3 in nts 
                    if "".join([n1,n2,n3]) not in stopcodons]
    
    diCodonsStop12 = [a + b for a in codonsNoStop for b in codonsNoStop 
                      if a not in stopcodons and b not in stopcodons and a[2] + b[:2] in ["CTA","TTA","TCA"]]

    ## The 1-2 frame requires some work
    noTAs = [codon for codon in codonsNoStop if codon[:2] != "TA"]
    noCAsOrTAs = [codon for codon in codonsNoStop if codon[:2] != "CA" and codon[:2] != "TA"]

    codonChain12 = {}
    for n1 in nts:
        for n2 in nts:
            for n3 in nts:
                thiscodon = n1 + n2 + n3
                if n3 == "T":
                    # cannot allow TA or CA in the first two 2nt
                    codonChain12[thiscodon] = noCAsOrTAs
                elif n3 == "C":
                    # cannot allow a TA in the first two 2nt
                    codonChain12[thiscodon] = noTAs
                else:
                    codonChain12[thiscodon] = codonsNoStop

    ### Generation code
    # Basic lengths -- in nucleotides, including stop codons
    l1 = 3*prot1 + 3
    l2 = 3*prot2 + 3
    seqL = l1 + l2 - overlap

    # Determine reading frame
    if (l1-overlap)%3 == 0:
        readingframe = "3-0"
    elif (l1-overlap - 1)%3 == 0:
        readingframe = "2-1"
    elif (l1-overlap - 2)%3 == 0:
        readingframe = "1-2"

    if not quiet:
        print("Reading frame: ", readingframe)

    # Step 1: fill in codons before the overlap
    beforeoverlap = int(np.floor((l1-overlap)/3))    # Codons before overlap

    seq = "".join(rng.choice(codonsNoStop, beforeoverlap))

    # Step 2: Pick the next two codons such that the frame opposite has a stop codon
    if readingframe == "3-0":
        next2 = revcomp(rng.choice(stopcodons)) + rng.choice(codonsNoStopEitherFrame)
    if readingframe == "2-1":
        next2 = randnt() + rng.choice(["TT", "CT", "TC"]) + "A" + randnt() + randnt()
    if readingframe == "1-2":
        next2 = randnt() + randnt() + rng.choice(["TT", "CT", "TC"]) + "A"
        if next2[3] == "T":
            next2 += rng.choice(["T","C"])
        else:
            next2 += randnt()

    seq += next2

    # Step 3: Fill out the bulk of the overlapping region
    lengthmidoverlap = int((l1 - 3*beforeoverlap - 12)/3) # not including the dicodons around each stop

    if readingframe == "3-0":
        seq += "".join(rng.choice(codonsNoStopEitherFrame, lengthmidoverlap))
        
    elif readingframe == "2-1":
        # one-by-one
        for codonsadded in np.arange(lengthmidoverlap):
            # Avoid a stop codon
            if seq[-2:] in ["TC", "CT", "TT"]:
                nextcodon = rng.choice(codonsNoStopNoA1)
            else:
                nextcodon = rng.choice(codonsNoStop)

            seq += nextcodon

    elif readingframe == "1-2":
        for codonsadded in np.arange(lengthmidoverlap):
            seq += rng.choice(codonChain12[seq[-3:]])


    # Step 4: Pick the last two codons -- the last one being a stop
    if readingframe == "3-0":
        nexttwocodons = rng.choice(codonsNoStopEitherFrame) + rng.choice(stopcodons)
    elif readingframe == "2-1":
        if seq[-2:] in ["TC", "CT", "TT"]:
            nexttwocodons = rng.choice(codonsNoStopNoA1) + rng.choice(stopcodons)
        else:
            nexttwocodons = rng.choice(codonsNoStop) + rng.choice(stopcodons)
    elif readingframe == "1-2":
        nexttwocodons = rng.choice(codonChain12[seq[-3:]])
        if nexttwocodons[-1] == "C" or nexttwocodons[-1] == "T":
            nexttwocodons += "TGA"
        else:
            nexttwocodons += rng.choice(stopcodons)

    seq += nexttwocodons

    # Step 5: If in a 2-1 or 1-2 frame, fill in nucleotides to complete a codon in frame 2
    if readingframe == "2-1":
        seq += randnt()
    elif readingframe == "1-2":
        seq+= randnt() + randnt()

    # Step 6: generate random codons in remainder of frame 2
    remainingcodons = int( np.floor((l2-overlap)/3) )

    seq += revcomp("".join(rng.choice(codonsNoStop, remainingcodons)))

    return seq


# Integer encoding for nucleotides
NUC_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
INT_TO_NUC = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

# Build integer codon table for fast lookup
CODON_TABLE_INT = np.full((4, 4, 4), ord('X'), dtype=np.uint8)
for codon, aa in CODON_TABLE.items():
    i, j, k = NUC_TO_INT[codon[0]], NUC_TO_INT[codon[1]], NUC_TO_INT[codon[2]]
    CODON_TABLE_INT[i, j, k] = ord(aa)

@njit
def seq_str_to_int_array(seq):
    arr = np.empty(len(seq), dtype=np.uint8)
    for i in range(len(seq)):
        nt = seq[i]
        if nt == 'A':
            arr[i] = 0
        elif nt == 'C':
            arr[i] = 1
        elif nt == 'G':
            arr[i] = 2
        else:  # 'T'
            arr[i] = 3
    return arr

@njit
def int_array_to_seq_str(arr):
    out = []
    for i in range(len(arr)):
        nt = arr[i]
        if nt == 0:
            out.append('A')
        elif nt == 1:
            out.append('C')
        elif nt == 2:
            out.append('G')
        else:
            out.append('T')
    return ''.join(out)

@njit
def fast_reverse_complement_int(seq):
    # seq: np.array of ints (0,1,2,3)
    comp = np.array([3, 2, 1, 0], dtype=np.uint8)  # A<->T, C<->G
    rc = np.empty_like(seq)
    n = len(seq)
    for i in range(n):
        rc[i] = comp[seq[n - 1 - i]]
    return rc

@njit
def fast_translate_int(seq):
    # seq: np.array of ints (0,1,2,3)
    n_codons = len(seq) // 3
    aa_seq = []
    for i in range(n_codons):
        a, b, c = seq[3*i], seq[3*i+1], seq[3*i+2]
        aa = CODON_TABLE_INT[a, b, c]
        aa_seq.append(chr(aa))
    return aa_seq

def split_sequence_and_to_aa_int(sequence, len_1, len_2):
    # sequence: np.array of ints
    aa_sequence_1 = fast_translate_int(sequence[:len_1])
    rc_seq = fast_reverse_complement_int(sequence[-len_2:])
    aa_sequence_2 = fast_translate_int(rc_seq)
    return aa_sequence_1, aa_sequence_2

@njit
def change_random_codon_int(len_sequence, sequence):
    # sequence: np.array of ints (0,1,2,3)
    new_position = np.random.randint(0, len_sequence)
    current_nt = sequence[new_position]
    idx = np.random.randint(0, 3)
    if idx >= current_nt:
        idx += 1
    new_sequence = sequence.copy()
    new_sequence[new_position] = idx
    return new_sequence, new_position, idx


# Example: update your overlapped_sequence_generator to use integer arrays
def overlapped_sequence_generator_int(DCA_params_1, DCA_params_2, initialsequence, T1=1, T2=1, numberofiterations=100000, quiet=False, whentosave=0.1):
    Jvec1, hvec1 = DCA_params_1
    Jvec2, hvec2 = DCA_params_2

    seq = seq_str_to_int_array(initialsequence)
    sequence_L = len(seq)
    len_seq_1_n = int(3*len(hvec1)/21 + 3)
    len_seq_2_n = int(3*len(hvec2)/21 + 3)
    original_seq = seq.copy()

    accepted = 0.0
    prob_accepted = 0
    not_accepted = 0
    energy_history_seq_1 = []
    energy_history_seq_2 = []

    aa_seq_1, aa_seq_2 = split_sequence_and_to_aa_int(seq, len_seq_1_n, len_seq_2_n)
    E1, E2, E = calculate_energies(aa_seq_1, aa_seq_2, Jvec1, hvec1, Jvec2, hvec2)

    energy_history_seq_1.append(E1)
    energy_history_seq_2.append(E2)

    itera = 1
    nextmessage = 100*whentosave # whentosave is a fraction of the total iterations. Convert to percentage
    while itera < numberofiterations:
        if 100*(itera/numberofiterations) > nextmessage:
            nextmessage += 100*whentosave
            energy_history_seq_1.append(E1)
            energy_history_seq_2.append(E2)
            if not quiet:
                print(f"Iteration {itera} of {numberofiterations}, Energy: {E}, E1: {E1}, E2: {E2}")

        new_seq, new_pos, new_nuc = change_random_codon_int(sequence_L, seq)
        aa_seq_1_new, aa_seq_2_new = split_sequence_and_to_aa_int(new_seq, len_seq_1_n, len_seq_2_n)

        # Reject change if it introduces new or removes existing stop codons
        if "*" in aa_seq_1_new[:-1] or "*" in aa_seq_2_new[:-1] or aa_seq_1_new[-1] != "*" or aa_seq_2_new[-1] != "*":
            continue

        E1_new, E2_new, E_new = calculate_energies(aa_seq_1_new, aa_seq_2_new, Jvec1, hvec1, Jvec2, hvec2)
        delta_H_1 = E1_new - E1
        delta_H_2 = E2_new - E2
        delta_H = (delta_H_1/T1) + (delta_H_2/T2)

        if delta_H <= 0:
            seq = new_seq.copy()
            E1 = E1 + delta_H_1
            E2 = E2 + delta_H_2
            E = E + delta_H_1 + delta_H_2
            accepted += 1
        else:
            probability = np.exp(-delta_H)
            random_number = np.random.rand()
            if random_number < probability:
                seq = new_seq.copy()
                E1 = E1 + delta_H_1
                E2 = E2 + delta_H_2
                E = E + delta_H_1 + delta_H_2
                prob_accepted += 1
            else:
                not_accepted += 1

        itera += 1

    finalenergies = np.array([E1, E2])
    acceptedornot = [accepted, prob_accepted, not_accepted]
    
    if not quiet:
        print(f"Accepted: {accepted}, Probabilistically Accepted: {prob_accepted}, Not Accepted: {not_accepted}")
        print(f"Final Energy: {E}, E1: {E1}, E2: {E2}")

    return int_array_to_seq_str(seq), acceptedornot, np.array(energy_history_seq_1), np.array(energy_history_seq_2), finalenergies

def main():
    overlapLen = 62

    #### Read in input data
    dcaparams1 = "../Data/20250601 bmDCA/bmDCA/PF00004/PF00004_params.dat"
    dcaparams2 = "../Data/20250601 bmDCA/bmDCA/PF00041/PF00041_params.dat"

    naturalenergies1_file = "../Data/20250601 bmDCA/bmDCA/PF00004/PF00004_naturalenergies.txt"
    naturalenergies2_file = "../Data/20250601 bmDCA/bmDCA/PF00041/PF00041_naturalenergies.txt"

    Js_1, hs_1 = extract_params(dcaparams1)
    Js_2, hs_2 = extract_params(dcaparams2)

    # Length in amino acids, not including stop codons
    lenprot1 = len(hs_1)/21  # 21 is the number of amino acids incl blank
    lenprot2 = len(hs_2)/21  # 21 is the number of amino acids incl blank

    print(f"Length of protein 1 w/out stop: {int(lenprot1)} amino acids or {int(3*lenprot1)} nucleotides")
    print(f"Length of protein 2 w/out stop: {int(lenprot2)} amino acids or {int(3*lenprot2)} nucleotides")
    print(f"Overlap length (nucleotides): {overlapLen}")

    # Combining parameters
    DCA_params_1 = [Js_1, hs_1]
    DCA_params_2 = [Js_2, hs_2]

    # The energies of natural sequences
    naturalenergies1 = load_natural_energies(naturalenergies1_file)
    naturalenergies2 = load_natural_energies(naturalenergies2_file)
    mean_1 = np.mean(naturalenergies1)
    mean_2 = np.mean(naturalenergies2)
    sd_1 = np.std(naturalenergies1)
    sd_2 = np.std(naturalenergies2)

    print("Data loaded")

    #### Overlap

    # Generate initial sequence
    initialCondition = initial_seq_no_stops(lenprot1, lenprot2, overlapLen)

    print("\n")

    # Time the sequence generator
    start_time = time.time()
    overlapOutput = overlapped_sequence_generator_int(DCA_params_1, DCA_params_2, 
                                                  initialCondition, numberofiterations=10000, 
                                                  T1=0.4, T2=0.5)
    end_time = time.time()

    # Print the execution time
    print(f"\n Overlapped sequence generation completed in {end_time - start_time:.2f} seconds.")

    seq = overlapOutput[0]
    acceptedornot = overlapOutput[1]
    E1_vec = overlapOutput[2]
    E2_vec = overlapOutput[3]
    hamming_vec = overlapOutput[4]

    # Print the final nucleotide sequence
    print("Final nucleotide sequence:")
    print("".join(seq))

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)

    axs[0].plot(E1_vec, "-k")
    axs[0].axhline(y=mean_1, linestyle='-', color=colorTable["o"])
    axs[0].axhline(y=mean_1 + sd_1, linestyle='--', color=colorTable["o"], label=r'$+\sigma$ seq 1')
    axs[0].axhline(y=mean_1 - sd_1, linestyle='--', color=colorTable["o"], label=r'$-\sigma$ seq 1')

    axs[1].plot(E2_vec, "-b")
    axs[1].axhline(y=mean_2, linestyle='-', color=colorTable["g"])
    axs[1].axhline(y=mean_2 + sd_2, linestyle='--', color=colorTable["g"], label=r'$+\sigma$ seq 2')
    axs[1].axhline(y=mean_2 - sd_2, linestyle='--', color=colorTable["g"], label=r'$-\sigma$ seq 2')
    plt.show()

if __name__ == "__main__":
    # Cprofile to see how long it takes to run
    # import cProfile
    # cProfile.run('main()')
    main()
