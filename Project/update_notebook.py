import json

notebook_path = r"c:\Users\orson\OneDrive\Documents\OverlappingGenes\Project\single_family_temperature_scan.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Modify Data Loading Cell
data_loading_cell = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if "load_family_data" in source_str:
            data_loading_cell = cell
            break

if data_loading_cell:
    print("Found data loading cell.")
    new_code = """

# Calculate Statistics for Convergence Check
mean_e1 = np.mean(nat_E1)
std_e1 = np.std(nat_E1)
mean_e2 = np.mean(nat_E2)
std_e2 = np.std(nat_E2)

print(f"PF00004 Stats: Mean={mean_e1:.2f}, Std={std_e1:.2f}")
print(f"PF00041 Stats: Mean={mean_e2:.2f}, Std={std_e2:.2f}")
"""
    # Append to the existing source
    # Ensure source is a list of strings
    if isinstance(data_loading_cell['source'], list):
        # Add newline if needed
        if data_loading_cell['source'] and not data_loading_cell['source'][-1].endswith('\n'):
             data_loading_cell['source'][-1] += '\n'
        data_loading_cell['source'].append(new_code)
    else:
        data_loading_cell['source'] += new_code
else:
    print("Could not find data loading cell!")

# 2. Modify Overlap Scan Cell
overlap_scan_cell = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if "def run_overlap_scan" in source_str:
            overlap_scan_cell = cell
            break

if overlap_scan_cell:
    print("Found overlap scan cell.")
    new_source = """def run_overlap_scan(overlap_range, T1, T2, J1, h1, J2, h2, num_trials=NUM_TRIALS_OVERLAP_SCAN):
    overlap_results = {
        'overlap': [],
        'E1': [],
        'E2': []
    }
    
    L1 = h1.shape[0]
    L2 = h2.shape[0]
    
    print("Starting Overlap Scan (Optimized)...")
    
    for ov in overlap_range:
        print(f"  Testing Overlap = {ov}...", end="", flush=True)
        for i in range(num_trials):
            init_seq = og.initial_seq_no_stops(L1, L2, ov, quiet=True)
            
            # Use convergence generator
            iterations, converged, final_E1, final_E2 = og.overlapped_sequence_generator_convergence(
                (J1, h1), 
                (J2, h2), 
                init_seq,
                mean_e1=mean_e1, std_e1=std_e1,
                mean_e2=mean_e2, std_e2=std_e2,
                max_iterations=NUM_ITERATIONS,
                T1=T1, 
                T2=T2
            )
            
            overlap_results['overlap'].append(ov)
            overlap_results['E1'].append(final_E1)
            overlap_results['E2'].append(final_E2)
            print(".", end="", flush=True)
        print(" Done.")
            
    return pd.DataFrame(overlap_results)

# Run Scan
df_results = run_overlap_scan(OVERLAP_RANGE, OPTIMAL_T1, OPTIMAL_T2, J1, h1, J2, h2)

# Plot Dot Plot
plt.figure(figsize=(12, 6))
plt.scatter(df_results['overlap'], df_results['E1'], label='PF00004 Energy', alpha=0.6)
plt.scatter(df_results['overlap'], df_results['E2'], label='PF00041 Energy', alpha=0.6)
plt.xlabel('Overlap Length (nt)')
plt.ylabel('Final Energy')
plt.title('Energy vs Overlap Length (Dot Plot)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""
    
    # Convert new source to list of lines for JSON format (optional but good practice)
    # Actually, Jupyter accepts a single string too, but list is standard.
    # Let's just split by newline to be safe and consistent.
    overlap_scan_cell['source'] = [line + '\n' for line in new_source.split('\n')]
    # Remove last newline from last item
    if overlap_scan_cell['source']:
        overlap_scan_cell['source'][-1] = overlap_scan_cell['source'][-1].rstrip('\n')

else:
    print("Could not find overlap scan cell!")

# Write back
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
