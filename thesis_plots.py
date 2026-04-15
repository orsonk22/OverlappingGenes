"""
Thesis plotting configuration.

Usage in any notebook:
    import sys; sys.path.insert(0, '<repo_root>')
    from thesis_plots import *
    apply_style()
"""
import matplotlib.pyplot as plt
import os

# --- Okabe-Ito palette (colourblind-safe) ---
BLACK       = "#1D24BC"
ORANGE      = '#E69F00'
SKY_BLUE    = '#56B4E9'
GREEN       = '#009E73'
YELLOW      = '#F0E442'
BLUE        = '#0072B2'
VERMILLION  = '#D55E00'
PURPLE      = '#CC79A7'

# Ordered list matching the prop_cycle in thesis_style.mplstyle
PALETTE = [BLACK, ORANGE, SKY_BLUE, GREEN, YELLOW, BLUE, VERMILLION, PURPLE]

# Semantic aliases for this project
FRAME_COLORS = {
    0: BLUE,
    1: ORANGE,
    2: GREEN,
}


def apply_style():
    """Load the thesis style sheet."""
    style_path = os.path.join(os.path.dirname(__file__), 'thesis_style.mplstyle')
    plt.style.use(style_path)
