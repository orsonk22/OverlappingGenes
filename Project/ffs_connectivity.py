import numpy as np
import overlappingGenes as og
import os
import json
import time

# --- CONFIGURATION START ---
FFS_CONFIG = {
    # Number of successful crossings to collect from the initial basin (Source -> Interface 0)
    "n_crossings_basin": 50,
    
    # Number of trials to run per interface (to estimate P(i+1|i))
    "k_trials_interface": 1000,
    
    # Step size for interfaces (Hamming distance)
    # Smaller = more interfaces, higher probability per step, but more steps.
    # Larger = fewer interfaces, lower probability, harder to cross.
    "interface_step_size": 1,
    
    # Maximum steps for a single "shoot" simulation
    "max_steps_shoot": 500000,
    
    # Maximum attempts to leave the initial basin
    "max_attempts_basin": 200000
}
# --- CONFIGURATION END ---

def load_and_run():
    pool_file = "converged_pool.json"
    if not os.path.exists(pool_file):
        print("Pool file not found.")
        return

    with open(pool_file, 'r') as f:
        pool = json.load(f)
        
    print(f"Loaded {len(pool)} sequences.")
    
    # Load Params
    project_dir = "c:/Users/orson/OneDrive/Documents/OverlappingGenes/Project"
    pf1 = "PF00004"
    pf2 = "PF00041"
    J1, h1 = og.extract_params(os.path.join(project_dir, pf1, f"{pf1}_params.dat"))
    J2, h2 = og.extract_params(os.path.join(project_dir, pf2, f"{pf2}_params.dat"))
    dca_1 = [J1, h1]
    dca_2 = [J2, h2]

    # Batch Run on All Unique Pairs
    # We run both A->B and B->A as connectivity might be asymmetric
    n_seqs = len(pool)
    results = []
    
    print(f"Starting Batch FFS on {n_seqs} sequences ({n_seqs*(n_seqs-1)} pairs)...")
    
    for i in range(n_seqs):
        for j in range(n_seqs):
            if i == j: continue
            
            s1 = pool[i]['sequence']
            s2 = pool[j]['sequence']
            
            print(f"\n=============================================")
            print(f"Running Connectivity: Seq {i} -> Seq {j}")
            print(f"=============================================")
            
            try:
                max_lambda = run_ffs_pairwise(dca_1, dca_2, s1, s2, pair_id=f"{i}_to_{j}")
                results.append({
                    "source": i,
                    "target": j,
                    "max_lambda": max_lambda
                })
            except Exception as e:
                print(f"Error running pair {i}->{j}: {e}")
                
    # Summary
    print("\n\n=== Batch Connectivity Results ===")
    print(f"{'Pair':<10} | {'Max Lambda':<15}")
    print("-" * 30)
    for res in results:
        print(f"{res['source']} -> {res['target']:<5} | {res['max_lambda']}")
        
    # Save Summary
    with open("ffs_batch_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Plot Heatmap
    plot_heatmap(results, n_seqs)
        
def run_ffs_pairwise(dca_params_1, dca_params_2, source_seq, target_seq, pair_id="run"):
    """
    Runs FFS to calculate connectivity from Source to Target.
    Returns: Max Lambda Reached (Penetration Depth)
    """
    
    # 1. Setup Same as before
    source_seq_int = og.seq_str_to_int_array(source_seq)
    target_seq_int = og.seq_str_to_int_array(target_seq)
    L = len(source_seq_int)
    
    lambda_start = og.count_matches(source_seq_int, target_seq_int)
    lambda_final = L
    
    print(f"FFS Start: Length={L}, Lambda_Start={lambda_start}, Lambda_Target={L}")
    
    # Define Interfaces
    step_size = FFS_CONFIG["interface_step_size"]
    interfaces = list(range(lambda_start + step_size, lambda_final, step_size))
    if len(interfaces) == 0 or interfaces[-1] != lambda_final:
        interfaces.append(lambda_final)
        
    print(f"Interfaces ({len(interfaces)}): {interfaces}")
    
    # Data for plotting
    lambdas_plot = [lambda_start]
    probs_plot = [1.0]
    max_lambda_reached = lambda_start
    
    # 2. Basin Sampling
    print(f"\n--- Basin Sampling (A -> {interfaces[0]}) ---")
    
    success_count = 0
    attempts = 0
    max_attempts = FFS_CONFIG["max_attempts_basin"]
    n_crossings = FFS_CONFIG["n_crossings_basin"]
    
    hits_collected = []
    
    while len(hits_collected) < n_crossings and attempts < max_attempts:
        attempts += 1
        res, final_seq = og.run_ffs_shoot(
            dca_params_1, dca_params_2, source_seq_int, target_seq_int,
            target_lambda=interfaces[0],
            fail_lambda=lambda_start - 5, 
            max_steps=10000
        )
        if res == 1:
            hits_collected.append(final_seq.copy())
            
    if attempts > 0:
        prob_0 = len(hits_collected) / attempts
    else:
        prob_0 = 0.0
        
    flux_0 = prob_0 
    print(f"Basin Crossing Prob: {prob_0:.4e} ({len(hits_collected)}/{attempts})")
    
    lambdas_plot.append(interfaces[0])
    probs_plot.append(prob_0)

    if len(hits_collected) == 0:
        print("Failed to exit basin.")
        plot_results(lambdas_plot, probs_plot, pair_id)
        return float(max_lambda_reached)

    current_states = hits_collected
    total_prob = prob_0
    max_lambda_reached = interfaces[0]
    
    # 3. Interface Propagation
    k_trials = FFS_CONFIG["k_trials_interface"]
    
    for i in range(len(interfaces) - 1):
        target_L = interfaces[i+1]
        current_L = interfaces[i]
        
        successes_next = []
        n_trials_this = 0
        success_count = 0
        
        num_to_run = k_trials
        
        for _ in range(num_to_run):
            if len(current_states) == 0: break
            start_state = current_states[np.random.randint(len(current_states))]
            
            fail_lambda_val = interfaces[i-1] if i > 0 else lambda_start
            
            res, final_seq = og.run_ffs_shoot(
                dca_params_1, dca_params_2, start_state, target_seq_int,
                target_lambda=target_L,
                fail_lambda=fail_lambda_val, 
                max_steps=FFS_CONFIG["max_steps_shoot"]
            )
            
            n_trials_this += 1
            if res == 1:
                success_count += 1
                successes_next.append(final_seq.copy())
        
        if n_trials_this == 0: p_i = 0
        else: p_i = success_count / n_trials_this
        
        total_prob *= p_i
        
        lambdas_plot.append(target_L)
        probs_plot.append(total_prob)
        
        if len(successes_next) == 0:
            print(f"  Path broken at lambda={target_L}. Max Reached: {max_lambda_reached}")
            plot_results(lambdas_plot, probs_plot, pair_id)
            return float(max_lambda_reached)
            
        current_states = successes_next
        max_lambda_reached = target_L

    plot_results(lambdas_plot, probs_plot, pair_id)
    return float(max_lambda_reached)

def plot_results(lambdas, probs, label):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(lambdas, probs, 'o-', linewidth=2)
        plt.yscale('log')
        plt.xlabel('Reaction Coordinate (Hamming Match Count)')
        plt.ylabel('Cumulative Probability / Flux')
        plt.title(f'FFS Connectivity Profile: {label}')
        plt.grid(True, which="both", ls="-")
        filename = f'ffs_connectivity_{label}.png'
        plt.savefig(filename)
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")

def plot_heatmap(results, n_seqs):
    try:
        import matplotlib.pyplot as plt
        
        matrix = np.zeros((n_seqs, n_seqs))
        # Initialize diagonal or empty spots?
        
        for res in results:
            i, j = res['source'], res['target']
            matrix[i, j] = res['max_lambda']
            
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, cmap='viridis', origin='upper')
        plt.colorbar(label='Max Lambda Reached')
        plt.xlabel('Target Sequence Index')
        plt.ylabel('Source Sequence Index')
        plt.title('Connectivity Penetration Depth (Max Matches)')
        plt.xticks(range(n_seqs))
        plt.yticks(range(n_seqs))
        
        # Annotate
        for i in range(n_seqs):
            for j in range(n_seqs):
                if i != j:
                    plt.text(j, i, f"{matrix[i, j]:.0f}", ha='center', va='center', color='w')
        
        plt.savefig('ffs_penetration_heatmap.png')
        print("Heatmap saved to ffs_penetration_heatmap.png")
    except Exception as e:
        print(f"Heatmap plotting failed: {e}")

if __name__ == "__main__":
    load_and_run()
