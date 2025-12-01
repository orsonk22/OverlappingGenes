
import time
import numpy as np
import matplotlib.pyplot as plt
import overlappingGenes as og_opt
import overlappingGenesoriginal as og_slow

def run_validation():
    print("Setting up validation...")
    
    # Mock data generation (same as benchmark)
    L_aa = 50
    L_nuc = L_aa * 3
    
    h_size = 21 * L_aa
    J_size = int((L_aa * (L_aa - 1) / 2) * 441) + 441 
    
    print(f"Generating mock parameters (L={L_aa} aa)...")
    # Use fixed seed for reproducibility of parameters
    np.random.seed(42)
    hvec1 = np.random.randn(h_size).astype(np.float32)
    Jvec1 = np.random.randn(J_size).astype(np.float32)
    
    hvec2 = np.random.randn(h_size).astype(np.float32)
    Jvec2 = np.random.randn(J_size).astype(np.float32)
    
    DCA_params_1 = [Jvec1, hvec1]
    DCA_params_2 = [Jvec2, hvec2]
    
    overlapLen = 20
    print("Generating valid initial sequence...")
    initial_seq = og_opt.initial_seq_no_stops(L_aa, L_aa, overlapLen, quiet=True)
    
    # Parameters
    T1 = 1.0
    T2 = 1.0
    # Multiple Trials Validation
    n_trials = 100
    n_iter_per_trial = 50000
    
    print(f"\nRunning Multiple Trials Validation (Trials={n_trials}, Iterations/Trial={n_iter_per_trial})...")
    
    final_energies_slow = []
    final_energies_fast = []
    
    start_time_total = time.time()
    
    for i in range(n_trials):
        if (i + 1) % 10 == 0:
            print(f"  Trial {i+1}/{n_trials}...")
            
        # Independent seed for each trial
        trial_seed = 1000 + i
        
        # Run Original
        np.random.seed(trial_seed)
        out_slow = og_slow.overlapped_sequence_generator_int(
            DCA_params_1, DCA_params_2, initial_seq, 
            T1=T1, T2=T2, numberofiterations=n_iter_per_trial, quiet=True, whentosave=1.0
        )
        final_energies_slow.append(out_slow[2][-1]) # Last energy
        
        # Run Optimized
        np.random.seed(trial_seed)
        out_fast = og_opt.overlapped_sequence_generator_int(
            DCA_params_1, DCA_params_2, initial_seq, 
            T1=T1, T2=T2, numberofiterations=n_iter_per_trial, quiet=True, whentosave=1.0
        )
        final_energies_fast.append(out_fast[2][-1]) # Last energy

    end_time_total = time.time()
    print(f"Total time for {n_trials} trials: {end_time_total - start_time_total:.2f}s")

    # Statistics
    print("\n--- Trial Statistics (Final Energies) ---")
    print(f"Original: Mean={np.mean(final_energies_slow):.4f}, Std={np.std(final_energies_slow):.4f}")
    print(f"Optimized: Mean={np.mean(final_energies_fast):.4f}, Std={np.std(final_energies_fast):.4f}")
    
    mean_diff = abs(np.mean(final_energies_slow) - np.mean(final_energies_fast))
    print(f"Mean Difference: {mean_diff:.4f}")

    # Plotting
    print("Generating plot...")
    plt.figure(figsize=(10, 6))
    plt.hist(final_energies_slow, bins=30, alpha=0.5, label='Original', density=True, color='red')
    plt.hist(final_energies_fast, bins=30, alpha=0.5, label='Optimized', density=True, color='green')
    plt.xlabel('Final Energy (Sequence 1)')
    plt.ylabel('Density')
    plt.title(f'Distribution of Final Energies ({n_trials} Trials, {n_iter_per_trial} Iterations)')
    plt.legend()
    plt.savefig('validation_histogram_trials.png')
    print("Plot saved to validation_histogram_trials.png")

if __name__ == "__main__":
    run_validation()
