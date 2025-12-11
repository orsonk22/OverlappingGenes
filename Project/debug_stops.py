
import numpy as np
import overlappingGenes as og
import sys

def debug_initial_sequence():
    print("DEBUG: Starting Initial Sequence Check")
    
    # Parameters from the notebook output
    L1 = 110
    L2 = 74
    overlap = 62
    
    print(f"Parameters: L1={L1}, L2={L2}, Overlap={overlap}")
    
    try:
        # Generate initial sequence
        # Note: initial_seq_no_stops(prot1, prot2, overlap)
        # where prot1, prot2 are lengths in AAs
        initial_seq_str = og.initial_seq_no_stops(L1, L2, overlap, quiet=False)
        print(f"Generated Sequence Length: {len(initial_seq_str)}")
        
        # Convert to numeric
        seq_int = og.seq_str_to_int_array(initial_seq_str)
        
        # Manually perform what overlapped_sequence_generator_selective does
        len_aa_1 = L1
        len_aa_2 = L2
        
        len_seq_1_n = 3 * len_aa_1
        len_seq_2_n = 3 * len_aa_2
        
        print(f"Expected Nucleotide Lengths: Gene1={len_seq_1_n}, Gene2={len_seq_2_n}")
        
        # Buffers
        aa_seq_1 = np.zeros(len_aa_1, dtype=np.int64)
        aa_seq_2 = np.zeros(len_aa_2, dtype=np.int64)
        rc_buffer = np.zeros(len_seq_2_n, dtype=np.int64)
        
        # Translate
        # Note: og.split_sequence_and_to_numeric_out is njit-ed
        og.split_sequence_and_to_numeric_out(seq_int, len_seq_1_n, len_seq_2_n, aa_seq_1, aa_seq_2, rc_buffer)
        
        print("\n--- Gene 1 Analysis ---")
        print(f"AA Sequence: {aa_seq_1}")
        stops_1_indices = np.where(aa_seq_1 == 21)[0]
        print(f"Stop Codon Indices (All): {stops_1_indices}")
        
        # Simulation check logic (post-fix):
        # for i in range(len_aa_1 - 1): if aa_seq_1[i] == 21: error
        internal_stops_1 = [i for i in stops_1_indices if i < len_aa_1 - 1]
        if internal_stops_1:
            print(f"FAIL: Internal Stop Codons found at {internal_stops_1}")
        else:
            print("PASS: No internal stop codons.")
            
        print(f"Terminal AA (Index {len_aa_1-1}): {aa_seq_1[-1]}")

        print("\n--- Gene 2 Analysis ---")
        print(f"AA Sequence: {aa_seq_2}")
        stops_2_indices = np.where(aa_seq_2 == 21)[0]
        print(f"Stop Codon Indices (All): {stops_2_indices}")
        
        internal_stops_2 = [i for i in stops_2_indices if i < len_aa_2 - 1]
        if internal_stops_2:
            print(f"FAIL: Internal Stop Codons found at {internal_stops_2}")
        else:
            print("PASS: No internal stop codons.")
            
        print(f"Terminal AA (Index {len_aa_2-1}): {aa_seq_2[-1]}")
        
        # Check initial energy if possible?
        # Would need full params, skipping for now as STOP check is critical.

    except Exception as e:
        print(f"Error during debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_initial_sequence()
