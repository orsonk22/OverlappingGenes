import json

notebook_path = r"c:\Users\orson\OneDrive\Documents\OverlappingGenes\Project\single_family_temperature_scan.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define new cells
helper_functions_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ==========================================\n",
        "# HELPER FUNCTIONS FOR TEMP SCAN\n",
        "# ==========================================\n",
        "\n",
        "def run_temp_scan(target_family_idx, temp_range, J1, h1, J2, h2, natural_energies, num_trials=NUM_TRIALS_TEMP_SCAN):\n",
        "    \"\"\"\n",
        "    Runs simulation for a range of temperatures for the target family,\n",
        "    while keeping the other family at a very high temperature (HIGH_TEMP).\n",
        "    \n",
        "    target_family_idx: 1 for PF00004, 2 for PF00041\n",
        "    \"\"\"\n",
        "    results = {}\n",
        "    \n",
        "    # Determine lengths for initial sequence generation\n",
        "    L1 = h1.shape[0]\n",
        "    L2 = h2.shape[0]\n",
        "    # Use a fixed overlap for scanning (e.g., small or medium)\n",
        "    scan_overlap = 10 \n",
        "    \n",
        "    # Calculate stats for convergence (we need to pass them even if ignored)\n",
        "    # We can just use the passed natural energies for the target, and dummy for other\n",
        "    mean_target = np.mean(natural_energies)\n",
        "    std_target = np.std(natural_energies)\n",
        "    \n",
        "    print(f\"Starting Temperature Scan for Family {target_family_idx}...\")\n",
        "    \n",
        "    for T in temp_range:\n",
        "        print(f\"  Testing T = {T}...\", end=\"\", flush=True)\n",
        "        energies = []\n",
        "        \n",
        "        for i in range(num_trials):\n",
        "            # Generate random initial sequence\n",
        "            init_seq = og.initial_seq_no_stops(L1, L2, scan_overlap, quiet=True)\n",
        "            \n",
        "            # Set temperatures based on target\n",
        "            if target_family_idx == 1:\n",
        "                T1_val = T\n",
        "                T2_val = HIGH_TEMP\n",
        "                check_e1 = True\n",
        "                check_e2 = False\n",
        "            else:\n",
        "                T1_val = HIGH_TEMP\n",
        "                T2_val = T\n",
        "                check_e1 = False\n",
        "                check_e2 = True\n",
        "            \n",
        "            # Run Simulation\n",
        "            # Note: We pass dummy stats for the ignored family, it doesn't matter\n",
        "            _, _, e1_final, e2_final = og.overlapped_sequence_generator_convergence(\n",
        "                (J1, h1), (J2, h2), init_seq,\n",
        "                mean_e1=mean_target if target_family_idx==1 else 0,\n",
        "                std_e1=std_target if target_family_idx==1 else 1,\n",
        "                mean_e2=mean_target if target_family_idx==2 else 0,\n",
        "                std_e2=std_target if target_family_idx==2 else 1,\n",
        "                max_iterations=NUM_ITERATIONS,\n",
        "                T1=T1_val, T2=T2_val,\n",
        "                check_e1=check_e1, check_e2=check_e2\n",
        "            )\n",
        "            \n",
        "            if target_family_idx == 1:\n",
        "                energies.append(e1_final)\n",
        "            else:\n",
        "                energies.append(e2_final)\n",
        "                \n",
        "        results[T] = energies\n",
        "        print(\" Done.\")\n",
        "        \n",
        "    return results\n",
        "\n",
        "def plot_histogram_comparison(results, natural_energies, family_name):\n",
        "    \"\"\"\n",
        "    Plots histograms of simulated energies at different temperatures\n",
        "    overlaid with the natural energy distribution.\n",
        "    \"\"\"\n",
        "    num_temps = len(results)\n",
        "    cols = 3\n",
        "    rows = int(np.ceil(num_temps / cols))\n",
        "    \n",
        "    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows), sharex=True)\n",
        "    axes = axes.flatten()\n",
        "    \n",
        "    # Natural energy stats\n",
        "    nat_mean = np.mean(natural_energies)\n",
        "    nat_std = np.std(natural_energies)\n",
        "    \n",
        "    for i, (T, energies) in enumerate(results.items()):\n",
        "        ax = axes[i]\n",
        "        \n",
        "        # Plot Natural\n",
        "        ax.hist(natural_energies, bins=30, alpha=0.5, label='Natural', density=True, color='gray')\n",
        "        \n",
        "        # Plot Simulated\n",
        "        ax.hist(energies, bins=30, alpha=0.5, label=f'Sim T={T}', density=True, color='blue')\n",
        "        \n",
        "        # Add lines for mean\n",
        "        ax.axvline(nat_mean, color='k', linestyle='--', label='Nat Mean')\n",
        "        ax.axvline(np.mean(energies), color='b', linestyle='--', label='Sim Mean')\n",
        "        \n",
        "        ax.set_title(f\"{family_name} at T={T}\")\n",
        "        ax.legend(fontsize='small')\n",
        "    \n",
        "    # Hide unused subplots\n",
        "    for j in range(i+1, len(axes)):\n",
        "        axes[j].axis('off')\n",
        "        \n",
        "    plt.tight_layout()\n",
        "    plt.show()\n"
    ]
}

scan_pf00004_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ==========================================\n",
        "# PHASE 1: TEMPERATURE SCANNING\n",
        "# ==========================================\n",
        "\n",
        "# Scan PF00004\n",
        "print(\"Scanning PF00004...\")\n",
        "results_pf00004 = run_temp_scan(1, TEMP_RANGE_PF00004, J1, h1, J2, h2, nat_E1)\n",
        "plot_histogram_comparison(results_pf00004, nat_E1, \"PF00004\")"
    ]
}

scan_pf00041_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Scan PF00041\n",
        "print(\"Scanning PF00041...\")\n",
        "results_pf00041 = run_temp_scan(2, TEMP_RANGE_PF00041, J1, h1, J2, h2, nat_E2)\n",
        "plot_histogram_comparison(results_pf00041, nat_E2, \"PF00041\")"
    ]
}

# Insert cells
# Find where to insert. We want to insert after data loading (index 3).
# Current structure:
# 0: Markdown (Intro)
# 1: Code (Imports)
# 2: Code (Params)
# 3: Code (Data Loading)
# 4: Markdown (Scan PF00041 - Placeholder?) -> REPLACE THIS
# 5: Markdown (Phase 2)
# 6: Code (Optimal Temps)
# 7: Code (Overlap Scan)

# We will replace cell 4 with our new cells.
if len(nb['cells']) > 4:
    # Insert helper functions after data loading
    nb['cells'].insert(4, helper_functions_cell)
    # Insert PF00004 scan
    nb['cells'].insert(5, scan_pf00004_cell)
    # Replace the old placeholder markdown for PF00041 with actual code
    nb['cells'][6] = scan_pf00041_cell 
else:
    print("Notebook structure unexpected, appending cells.")
    nb['cells'].append(helper_functions_cell)
    nb['cells'].append(scan_pf00004_cell)
    nb['cells'].append(scan_pf00041_cell)

# Write back
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
