import numpy as np
import matplotlib.pyplot as plt
import time
import os
import overlappingGenes as og
import traceback

# --- CONFIGURATION ---
NUM_TRIALS = 20              # Number of independent trials per overlap length
OVERLAP_LENGTHS = list(range(10, 101, 5)) # Overlap lengths to test (10, 20, ..., 100)
MAX_ITERATIONS = 500000000 # Maximum MC steps before declaring non-convergence
# ---------------------

def load_dca_params(base_dir, pf_name):
    params_file = os.path.join(base_dir, pf_name, f"{pf_name}_params.dat")
    print(f"Loading parameters from {params_file}...")
    J, h = og.extract_params(params_file)
    return (J, h)

def load_natural_energies_stats(base_dir, pf_name):
    filename = os.path.join(base_dir, pf_name, f"{pf_name}_naturalenergies.txt")
    energies = og.load_natural_energies(filename)
    return np.mean(energies), np.std(energies)

def run_benchmark(overlap_lengths, num_trials):
    # Load parameters
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Loaded overlappingGenes from: {og.__file__}")
    dca_params_1 = load_dca_params(base_dir, "PF00004")
    dca_params_2 = load_dca_params(base_dir, "PF00041")
    
    # Calculate target energy stats
    mean_e1, std_e1 = load_natural_energies_stats(base_dir, "PF00004")
    mean_e2, std_e2 = load_natural_energies_stats(base_dir, "PF00041")
    print(f"Target Ranges: PF00004 [{mean_e1-std_e1:.2f}, {mean_e1+std_e1:.2f}], PF00041 [{mean_e2-std_e2:.2f}, {mean_e2+std_e2:.2f}]")
    
    # Calculate protein lengths
    len_h1 = len(dca_params_1[1])
    len_h2 = len(dca_params_2[1])
    prot1_len = int(len_h1 / 21)
    prot2_len = int(len_h2 / 21)
    
    print(f"Protein 1 length: {prot1_len}, Protein 2 length: {prot2_len}")

    # Results storage
    results_overlap = {'x': [], 'time_mean': [], 'time_std': [], 'iter_mean': [], 'iter_std': []}

    # 1. Scaling with Overlap Length (Time to Convergence)
    print(f"\nBenchmarking Overlap Length Scaling (Time to Convergence)...")
    print(f"Configuration: {num_trials} trials, Max Iters: {MAX_ITERATIONS}")
    
    for overlap in overlap_lengths:
        times = []
        iters = []
        print(f"  Testing overlap: {overlap}")
        for trial in range(num_trials):
            try:
                # Generate initial sequence
                # Removed quiet=True to debug
                initial_seq = og.initial_seq_no_stops(prot1_len, prot2_len, overlap)
                
                start_time = time.perf_counter()
                # Run convergence generator
                iterations, converged, final_E1, final_E2 = og.overlapped_sequence_generator_convergence(
                    dca_params_1, dca_params_2, initial_seq, 
                    mean_e1=mean_e1, std_e1=std_e1,
                    mean_e2=mean_e2, std_e2=std_e2,
                    max_iterations=MAX_ITERATIONS
                )
                end_time = time.perf_counter()
                
                if converged:
                    times.append(end_time - start_time)
                    iters.append(iterations)
                else:
                    print(f"    Trial {trial} did not converge in {iterations} iterations. Final E1: {final_E1:.2f}, E2: {final_E2:.2f}")
                
            except Exception as e:
                print(f"    Failed to generate initial sequence for overlap {overlap}: {e}")
                traceback.print_exc()
                continue
        
        if times:
            results_overlap['x'].append(overlap)
            results_overlap['time_mean'].append(np.mean(times))
            results_overlap['time_std'].append(np.std(times))
            results_overlap['iter_mean'].append(np.mean(iters))
            results_overlap['iter_std'].append(np.std(iters))
            print(f"    Mean Time: {np.mean(times):.6f}s, Mean Iters: {np.mean(iters):.1f}")

    return results_overlap

def plot_results(results_overlap):
    print("Results:", results_overlap)
    
    if not results_overlap['x']:
        print("No data to plot!")
        return

    # Filter out zero or negative values for log plot
    x = np.array(results_overlap['x'])
    y = np.array(results_overlap['time_mean'])
    y_err = np.array(results_overlap['time_std'])
    
    # Add small epsilon to avoid log(0) if any
    y_safe = np.maximum(y, 1e-9)

    # Plot 1: Time to Convergence vs Overlap Length (Linear)
    plt.figure(figsize=(10, 6))
    plt.errorbar(x, y, yerr=y_err, 
                 fmt='o-', capsize=5, ecolor='red', color='blue')
    plt.title(f'Time to Convergence vs Overlap Length\n(Trials={NUM_TRIALS})')
    plt.xlabel('Overlap Length (nucleotides)')
    plt.ylabel('Time to Convergence (seconds)')
    plt.grid(True)
    plt.savefig('convergence_time_vs_overlap.png')
    print("Saved convergence_time_vs_overlap.png")
    plt.close()

    # Plot 2: Time to Convergence vs Overlap Length (Log-Log)
    plt.figure(figsize=(10, 6))
    # Ensure x is also positive (it is, but good to be safe)
    valid_mask = (x > 0) & (y_safe > 0)
    if np.any(valid_mask):
        plt.errorbar(x[valid_mask], y_safe[valid_mask], yerr=y_err[valid_mask], 
                     fmt='o-', capsize=5, ecolor='red', color='blue')
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'Time to Convergence vs Overlap Length (Log-Log)\n(Trials={NUM_TRIALS})')
        plt.xlabel('Overlap Length (nucleotides)')
        plt.ylabel('Time to Convergence (seconds)')
        plt.grid(True, which="both", ls="-")
        plt.savefig('convergence_time_vs_overlap_loglog.png')
        print("Saved convergence_time_vs_overlap_loglog.png")
    else:
        print("No valid positive data for log-log plot.")
    plt.close()
    
    # Plot 3: Iterations to Convergence vs Overlap Length (Linear)
    plt.figure(figsize=(10, 6))
    plt.errorbar(x, results_overlap['iter_mean'], yerr=results_overlap['iter_std'], 
                 fmt='o-', capsize=5, ecolor='red', color='green')
    plt.title(f'Iterations to Convergence vs Overlap Length\n(Trials={NUM_TRIALS})')
    plt.xlabel('Overlap Length (nucleotides)')
    plt.ylabel('Iterations')
    plt.grid(True)
    plt.savefig('convergence_iters_vs_overlap.png')
    print("Saved convergence_iters_vs_overlap.png")
    plt.close()

if __name__ == "__main__":
    # Run benchmark
    print(f"Starting convergence benchmark ({NUM_TRIALS} trials)...")
    res_overlap = run_benchmark(OVERLAP_LENGTHS, NUM_TRIALS)
    
    # Plot
    plot_results(res_overlap)
    print("Done.")
