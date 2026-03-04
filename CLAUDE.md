# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UCL MSci project studying **overlapping genes** — how two proteins can be encoded in the same DNA stretch using different reading frames. Uses **Direct Coupling Analysis (DCA)** energy models with **Monte Carlo (MC) simulations** to evolve overlapping nucleotide sequences and measure protein stability trade-offs.

## Architecture

### Core Module: `overlappingGenes.py`

Multiple variants exist, each tailored to its context:

| Location | Purpose |
|---|---|
| `Project/overlappingGenes.py` | Main implementation (1692 lines). Full-featured with all utilities. |
| `Project/overlapping_genes_cluster.py` | Cluster-optimized: integer encoding, pre-allocated arrays, Numba throughout. |
| `Large Scale/overlappingGenes.py` | Version used by large-scale multiprocessing notebooks. |
| `GA/overlappingGenes.py` | Genetic algorithm variant. |

All variants are imported as `og` and share the same core interface: loading DCA parameters, building overlapping nucleotide sequences, computing DCA energies, and running MC simulations.

### Key Computational Pipeline

1. **Load DCA parameters** from `bmDCA/<PF_name>/<PF_name>_params.dat` (J couplings + h fields)
2. **Load natural energy statistics** from `bmDCA/<PF_name>/<PF_name>_naturalenergies.txt` for z-score normalization
3. **Construct overlapping nucleotide sequence** from two amino acid sequences at a specified overlap length and reading frame
4. **Run MC simulation** (Metropolis-Hastings): propose single-nucleotide mutations, accept/reject via Boltzmann criterion on combined DCA energy, reject if stop codons introduced
5. **Collect results**: final energies, z-scores, convergence iteration, energy trajectories

### Parallelization Pattern

Worker functions **must** live in separate `.py` files (not notebook cells) due to Windows `spawn`-based multiprocessing requiring picklable imports.

- `Large Scale/multiprocessing_worker.py` — general pair-processing worker
- `Large Scale/temp_optimization_worker.py` — temperature sweep worker
- `GA/ga_worker.py` — genetic algorithm worker
- `Project/cluster_overlap_analysis.py` — standalone coordinator script with checkpointing

### Data Layout

```
bmDCA/<PF_name>/
  ├── <PF_name>_params.dat          # DCA J and h parameters
  ├── <PF_name>_naturalenergies.txt # Natural energy distribution
  ├── <PF_name>_chains.fasta        # Protein sequences
  └── <PF_name>_weights.dat         # Alignment weights
```

12 protein families: PF00004, PF00018, PF00041, PF00072, PF00076, PF00096, PF00153, PF00271, PF00397, PF00512, PF00595, PF01029.

Optimal MC temperatures per family are stored in `optimal_temperatures.json` (z-score rescaled).

## Running Simulations

### From Jupyter notebooks (primary workflow)
Notebooks in `Large Scale/`, `GA/`, `Replica Exchange/`, and `Project/` are the main entry points. Open and run cells sequentially.

### Cluster script
```bash
python Project/cluster_overlap_analysis.py              # Full analysis
python Project/cluster_overlap_analysis.py --test        # Quick test mode
python Project/cluster_overlap_analysis.py --resume checkpoint.pkl  # Resume
```

Configuration constants (ITERATIONS, N_TRIALS, OVERLAP_START/STOP/STEP, N_WORKERS) are edited directly at the top of the script.

## Key Technical Details

- **Numba `@njit`** on all hot-path functions (energy calculation, MC loop). First call triggers compilation — expect a delay.
- **Integer encoding**: nucleotides and amino acids stored as `uint8` arrays in the cluster-optimized version for Numba compatibility.
- **Delta energy**: single-mutation energy updates are O(L) not O(L²), exploiting the structure of DCA Hamiltonians.
- **Reading frames**: Frame 0 (overlap divisible by 3), Frame +1 (overlap % 3 == 1), Frame +2 (overlap % 3 == 2). Overlap lengths are in nucleotides.
- **Z-score normalization**: `z = (E - mean_natural) / std_natural`, used to compare energies across protein families of different sizes.
- **Stop codon avoidance**: mutations producing TAA, TAG, or TGA (or reverse complement equivalents) are rejected.

## Dependencies

Python with: `numpy`, `numba`, `matplotlib`, `seaborn`, `pandas`, `palettable`, `jupyter`. No `requirements.txt` — install manually.
