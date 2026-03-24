"""
Fix remaining colour/grid inconsistencies across notebooks.
Run from repo root: python fix_colours_grids.py
"""
import json
import re
import os

BASE = os.path.dirname(os.path.abspath(__file__))


def load_nb(rel_path):
    with open(os.path.join(BASE, rel_path), encoding='utf-8') as f:
        return json.load(f)


def save_nb(nb, rel_path):
    with open(os.path.join(BASE, rel_path), 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Saved: {rel_path}")


def fix_source(src, replacements, remove_patterns=None):
    """Apply str replacements, then remove full lines matching any pattern."""
    for old, new in replacements:
        src = src.replace(old, new)
    if remove_patterns:
        for pat in remove_patterns:
            # Remove the entire line (including leading whitespace and trailing \n)
            src = re.sub(r'[^\n]*' + re.escape(pat) + r'[^\n]*\n?', '', src)
    return src


def apply_to_cell(nb, cell_id, replacements, remove_patterns=None):
    """Apply fixes to a cell identified by its 'id' field."""
    for cell in nb['cells']:
        if cell.get('id') == cell_id:
            src = ''.join(cell['source'])
            new_src = fix_source(src, replacements, remove_patterns)
            if new_src != src:
                cell['source'] = new_src.splitlines(keepends=True)
            return True
    print(f"  WARNING: cell id '{cell_id}' not found")
    return False


def apply_to_cell_index(nb, idx, replacements, remove_patterns=None):
    """Apply fixes to a cell by zero-based index."""
    cell = nb['cells'][idx]
    src = ''.join(cell['source'])
    new_src = fix_source(src, replacements, remove_patterns)
    if new_src != src:
        cell['source'] = new_src.splitlines(keepends=True)


# ─── 1. Project/temp_benchmark.ipynb (nbformat 4.4 — use indices) ───────────
print("temp_benchmark.ipynb")
nb = load_nb("Project/temp_benchmark.ipynb")

# Cells 8 & 11: fill_between + axvline use 'b' for a band → SKY_BLUE
for idx in (8, 11):
    apply_to_cell_index(nb, idx, [
        ("color='b', alpha=0.2", "color=SKY_BLUE, alpha=0.2"),
        ("color='b', alpha=0.6", "color=SKY_BLUE, alpha=0.6"),
    ])

# Cell 13: errorbar line → BLUE; grid remove; hist colours
apply_to_cell_index(nb, 13, [
    ("color='b', capsize=5",   "color=BLUE, capsize=5"),
    ("color='grey', label='Natural'", "color=BLACK, label='Natural'"),
    ("color='blue', label='Sim'",     "color=BLUE, label='Sim'"),
], remove_patterns=["plt.grid(True)"])

# Cell 15: same pattern for protein 2
apply_to_cell_index(nb, 15, [
    ("color='b', capsize=5",   "color=BLUE, capsize=5"),
    ("color='grey', label='Natural'",    "color=BLACK, label='Natural'"),
    ("color='blue', label='Generated'",  "color=BLUE, label='Generated'"),
], remove_patterns=["plt.grid(True)"])

save_nb(nb, "Project/temp_benchmark.ipynb")


# ─── 2. Project/optimization_benchmark_v2.ipynb (nbformat 4.4 — use index) ──
print("optimization_benchmark_v2.ipynb")
nb = load_nb("Project/optimization_benchmark_v2.ipynb")

apply_to_cell_index(nb, 9, [
    ("color='C2', alpha=0.20", "color=GREEN, alpha=0.20"),
    ("color='C2', lw=1",       "color=GREEN, lw=1"),
], remove_patterns=["ax.grid(True, alpha=0.3, which='both')"])

save_nb(nb, "Project/optimization_benchmark_v2.ipynb")


# ─── 3. Large Scale/large_scale_OVERLAPS_multiprocessing.ipynb ───────────────
print("large_scale_OVERLAPS_multiprocessing.ipynb")
nb = load_nb("Large Scale/large_scale_OVERLAPS_multiprocessing.ipynb")

apply_to_cell(nb, "cell-18", [],
    remove_patterns=["ax.grid(True, which='both'"])

save_nb(nb, "Large Scale/large_scale_OVERLAPS_multiprocessing.ipynb")


# ─── 4. Replica Exchange/re2d_cluster_results.ipynb ─────────────────────────
print("re2d_cluster_results.ipynb")
nb = load_nb("Replica Exchange/re2d_cluster_results.ipynb")

# Cell cell-3: heatmap — 'r*' → marker+color; markeredgecolor='k' → BLACK
apply_to_cell(nb, "cell-3", [
    ("'r*', ms=MARKER_SIZE,",
     "marker='*', color=VERMILLION, ms=MARKER_SIZE,"),
    ("markeredgecolor='k', markeredgewidth=0.5)",
     "markeredgecolor=BLACK, markeredgewidth=0.5)"),
])

# Cell cell-6: pareto — 'b*', edgecolor, color='k', c='gray'
apply_to_cell(nb, "cell-6", [
    ("c='gray',",                          "c=BLACK,"),
    ("'b*', ms=MARKER_SIZE, markeredgecolor='k',",
     "marker='*', color=BLUE, ms=MARKER_SIZE, markeredgecolor=BLACK,"),
    ("edgecolor='blue',",                  "edgecolor=BLUE,"),
    ("color='k', lw=0.3)",                 "color=BLACK, lw=0.3)"),
])

save_nb(nb, "Replica Exchange/re2d_cluster_results.ipynb")


# ─── 5. Replica Exchange/re2d_cross_analysis.ipynb ──────────────────────────
print("re2d_cross_analysis.ipynb")
nb = load_nb("Replica Exchange/re2d_cross_analysis.ipynb")

# Cell cell-2-frames
apply_to_cell(nb, "cell-2-frames", [
    ("{'Frame 0': '#1f77b4', 'Frame +1': '#ff7f0e', 'Frame +2': '#2ca02c'}",
     "{'Frame 0': FRAME_COLORS[0], 'Frame +1': FRAME_COLORS[1], 'Frame +2': FRAME_COLORS[2]}"),
    ("color='k', size=3",        "color=BLACK, size=3"),
    ("facecolor='gray', alpha=0.5", "facecolor=BLACK, alpha=0.5"),
    ("markerfacecolor='k',",     "markerfacecolor=BLACK,"),
])

# Cell cell-7-pareto
apply_to_cell(nb, "cell-7-pareto", [
    # inline frame_palette dict → use the one from cell-2-frames
    ("{'Frame 0': '#1f77b4', 'Frame +1': '#ff7f0e', 'Frame +2': '#2ca02c'}[fl]",
     "{'Frame 0': FRAME_COLORS[0], 'Frame +1': FRAME_COLORS[1], 'Frame +2': FRAME_COLORS[2]}[fl]"),
    ("color='k', size=3",           "color=BLACK, size=3"),
    ("color='gray', ls=':',",        "color=BLACK, ls=':',"),
    ("edgecolor='k', linewidth=0.3)", "edgecolor=BLACK, linewidth=0.3)"),
    ("edgecolor='k', label=",        "edgecolor=BLACK, label="),
])

# Cell cell-8-family
apply_to_cell(nb, "cell-8-family", [
    ("edgecolor='k', linewidth=0.3, capsize=3)", "edgecolor=BLACK, linewidth=0.3, capsize=3)"),
    ("color='k', size=2",  "color=BLACK, size=2"),
    # bar chart edgecolor
    ("edgecolor='k', linewidth", "edgecolor=BLACK, linewidth"),
])

# Cell 2ltijepcygo (sigma sweep)
apply_to_cell(nb, "2ltijepcygo", [
    ("color='gray', ls=':',", "color=BLACK, ls=':',"),
])

save_nb(nb, "Replica Exchange/re2d_cross_analysis.ipynb")


# ─── 6. GA/GA_convergence_heatmap.ipynb ─────────────────────────────────────
print("GA_convergence_heatmap.ipynb")
nb = load_nb("GA/GA_convergence_heatmap.ipynb")

# nbformat 4.4 — no cell ids; cell 7 by index
apply_to_cell_index(nb, 7, [],
    remove_patterns=["ax.grid(True, alpha=0.3)"])

save_nb(nb, "GA/GA_convergence_heatmap.ipynb")


# ─── 7. GA/GA_hamming_vs_energy.ipynb ───────────────────────────────────────
print("GA_hamming_vs_energy.ipynb")
nb = load_nb("GA/GA_hamming_vs_energy.ipynb")

# Cell 12: axvline color='b' → BLUE
apply_to_cell_index(nb, 12, [
    ("color='b', linestyle='--', linewidth=2,",
     "color=BLUE, linestyle='--', linewidth=2,"),
    # also fix the orange axvline while we're here
    ("color='orange', linestyle='--', linewidth=2,",
     "color=ORANGE, linestyle='--', linewidth=2,"),
])

# Cell 25: colorizer typo → color=BLUE; remove grid
apply_to_cell_index(nb, 25, [
    ("colorizer='blue'", "color=BLUE"),
], remove_patterns=["ax.grid(True, alpha=0.7)"])

# Cell 27: remove grid
apply_to_cell_index(nb, 27, [],
    remove_patterns=["ax.grid(True, alpha=0.3)"])

# Cell 32: 'r-' format string → explicit color
apply_to_cell_index(nb, 32, [
    ("p(x_line), 'r-', linewidth=2.5,",
     "p(x_line), color=VERMILLION, linewidth=2.5,"),
])

save_nb(nb, "GA/GA_hamming_vs_energy.ipynb")


# ─── 8. GA/hamming_controlled_barrier.ipynb ─────────────────────────────────
print("hamming_controlled_barrier.ipynb")
nb = load_nb("GA/hamming_controlled_barrier.ipynb")

# Cell 15: color='red' → VERMILLION
apply_to_cell_index(nb, 15, [
    ("color='red', linestyle='--',", "color=VERMILLION, linestyle='--',"),
])

save_nb(nb, "GA/hamming_controlled_barrier.ipynb")


print("\nDone.")
