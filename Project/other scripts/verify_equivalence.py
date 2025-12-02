
import numpy as np
import overlappingGenes as og_opt
import overlappingGenesoriginal as og_orig

def verify_mappings():
    print("--- Verifying Mappings ---")
    
    # 1. Verify AA Mapping
    # Original mapping is inside to_numeric
    orig_map = {'-': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}
    
    # Optimized mapping is via aa_char_to_int (jitted, so we test via wrapper or manual check)
    # We can check the CODON_TABLE_NUMERIC in og_opt which uses the same mapping logic
    
    chars = list(orig_map.keys())
    match = True
    for char in chars:
        expected = orig_map[char]
        # We can't call the jitted function directly easily from here without compiling, 
        # but we can check the result of translation.
        # Or we can just rely on the fact that I read the code. 
        # Let's test via translation.
        
        # Find a codon for this AA
        codon = None
        for c, aa in og_orig.CODON_TABLE.items():
            if aa == char:
                codon = c
                break
        
        if codon:
            # Translate using optimized
            # Convert codon to int array
            seq_int = og_opt.seq_str_to_int_array(codon)
            # Translate
            aa_out = np.zeros(1, dtype=np.int32)
            og_opt.translate_numeric_out(seq_int, aa_out)
            actual = aa_out[0]
            
            if actual != expected:
                print(f"MISMATCH for AA {char}: Expected {expected}, Got {actual}")
                match = False
    
    if match:
        print("AA Mapping: MATCH")
    else:
        print("AA Mapping: FAILED")

def verify_sequence_processing():
    print("\n--- Verifying Sequence Processing ---")
    
    L_aa = 10
    L_nuc = L_aa * 3
    overlap = 10
    
    # Generate random sequence string
    nts = ['A', 'C', 'G', 'T']
    seq_str = "".join(np.random.choice(nts, size=L_nuc + L_nuc - overlap))
    
    # 1. Original Processing
    # Note: original split_sequence_and_to_aa takes lengths in AA? No, in nucleotides?
    # Let's check original code signature: def split_sequence_and_to_aa(sequence, len_1, len_2):
    # It slices sequence[:len_1]. So len_1 is in nucleotides.
    
    len_1_n = L_nuc
    len_2_n = L_nuc
    
    aa_1_orig, aa_2_orig = og_orig.split_sequence_and_to_aa(seq_str, len_1_n, len_2_n)
    # Convert to numeric
    # Original to_numeric fails if stop codons are present. 
    # We need to handle that or ensure no stops. 
    # For verification, let's just compare the AA strings first (before numeric conversion).
    
    # Optimized Processing
    seq_int = og_opt.seq_str_to_int_array(seq_str)
    aa_1_opt_int = np.zeros(len(aa_1_orig), dtype=np.int32)
    aa_2_opt_int = np.zeros(len(aa_2_orig), dtype=np.int32)
    rc_buffer = np.zeros(len_2_n, dtype=np.uint8)
    
    og_opt.split_sequence_and_to_numeric_out(seq_int, len_1_n, len_2_n, aa_1_opt_int, aa_2_opt_int, rc_buffer)
    
    # Convert Optimized Ints back to Chars for comparison with Original Chars
    # We need a reverse map for this test
    int_to_aa = {v: k for k, v in {'-': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, '*': 21}.items()}
    
    aa_1_opt_str = [int_to_aa.get(x, '?') for x in aa_1_opt_int]
    aa_2_opt_str = [int_to_aa.get(x, '?') for x in aa_2_opt_int]
    
    # Compare
    # Note: Original fast_translate returns 'X' for unknown/stop? 
    # Original CODON_TABLE has '*' for stop.
    # Original to_numeric crashes on '*'.
    # But split_sequence_and_to_aa returns the list of chars.
    
    print(f"Seq 1 Match: {list(aa_1_orig) == aa_1_opt_str}")
    if list(aa_1_orig) != aa_1_opt_str:
        print(f"Orig: {list(aa_1_orig)}")
        print(f"Opt:  {aa_1_opt_str}")
        
    print(f"Seq 2 Match: {list(aa_2_orig) == aa_2_opt_str}")
    if list(aa_2_orig) != aa_2_opt_str:
        print(f"Orig: {list(aa_2_orig)}")
        print(f"Opt:  {aa_2_opt_str}")

def verify_energy_delta():
    print("\n--- Verifying Energy Delta Logic (Rigorous) ---")
    
    # Mock Params
    L_aa = 50
    h_size = 21 * L_aa
    J_size = int((L_aa * (L_aa - 1) / 2) * 21 * 21)
    
    np.random.seed(42)
    hvec = np.random.randn(h_size).astype(np.float32)
    Jvec = np.random.randn(J_size).astype(np.float32)
    
    # Random Sequence (Numeric)
    seq = np.random.randint(0, 21, size=L_aa).astype(np.int32)
    
    # Calculate Full Energy Initial
    E_initial = og_opt.calculate_Energy(seq, Jvec, hvec)
    
    print(f"Initial Energy: {E_initial}")
    
    errors = 0
    trials = 100
    
    for i in range(trials):
        # Mutate random position
        pos = np.random.randint(0, L_aa)
        old_val = seq[pos]
        new_val = np.random.randint(0, 21)
        
        if old_val == new_val: continue
            
        seq_new = seq.copy()
        seq_new[pos] = new_val
        
        # Calculate Full Energy New
        E_final = og_opt.calculate_Energy(seq_new, Jvec, hvec)
        
        # Calculate Delta
        delta_E = og_opt.calculate_Delta_Energy(seq, Jvec, hvec, pos, new_val)
        
        # Check
        E_predicted = E_initial + delta_E
        
        if not np.isclose(E_final, E_predicted, atol=1e-4):
            print(f"MISMATCH at trial {i}, pos {pos}:")
            print(f"  E_initial: {E_initial}")
            print(f"  E_final:   {E_final}")
            print(f"  E_pred:    {E_predicted}")
            print(f"  Delta:     {delta_E}")
            print(f"  Real Delta:{E_final - E_initial}")
            errors += 1
            break
        
        # Update for next step (random walk)
        seq = seq_new
        E_initial = E_final

    if errors == 0:
        print(f"Energy Delta: PASSED ({trials} random mutations)")
    else:
        print("Energy Delta: FAILED")

def verify_symmetry():
    print("\n--- Verifying Symmetry (Detailed Balance) ---")
    # Mock Params
    L_aa = 50
    h_size = 21 * L_aa
    J_size = int((L_aa * (L_aa - 1) / 2) * 21 * 21)
    
    np.random.seed(42)
    hvec = np.random.randn(h_size).astype(np.float32)
    Jvec = np.random.randn(J_size).astype(np.float32)
    
    seq = np.random.randint(0, 21, size=L_aa).astype(np.int32)
    
    errors = 0
    trials = 100
    
    for i in range(trials):
        pos = np.random.randint(0, L_aa)
        old_val = seq[pos]
        new_val = np.random.randint(0, 21)
        
        if old_val == new_val: continue
            
        # Forward: old -> new
        delta_fwd = og_opt.calculate_Delta_Energy(seq, Jvec, hvec, pos, new_val)
        
        # Backward: new -> old
        # We need to pass the "new" sequence as the base for backward calculation
        seq_new = seq.copy()
        seq_new[pos] = new_val
        delta_bwd = og_opt.calculate_Delta_Energy(seq_new, Jvec, hvec, pos, old_val)
        
        if not np.isclose(delta_fwd + delta_bwd, 0.0, atol=1e-4):
            print(f"ASYMMETRY at trial {i}: Fwd={delta_fwd}, Bwd={delta_bwd}, Sum={delta_fwd+delta_bwd}")
            errors += 1
            
    if errors == 0:
        print(f"Symmetry: PASSED ({trials} trials)")
    else:
        print("Symmetry: FAILED")

if __name__ == "__main__":
    verify_mappings()
    verify_sequence_processing()
    verify_energy_delta()
    verify_symmetry()
