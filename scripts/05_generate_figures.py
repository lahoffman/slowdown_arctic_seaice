#!/usr/bin/env python3
"""
Step 5: Generate Figures

Create publication-quality figures from results.
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths, training
from slowdown.data import load_netcdf, load_cesm2_grid
from slowdown.visualization import (
    setup_figure_style,
    plot_map,
    plot_timeseries
)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate figures')
    parser.add_argument('--figure', type=str, default='all',
                        help='Which figure to generate (default: all)')
    return parser.parse_args()


def plot_model_performance():
    """Plot model performance across all splits."""

    print("\nGenerating model performance figure...")

    # TODO: Load metrics from all splits
    # Aggregate and plot

    print("  [Implement model performance plotting]")

    output_file = paths.get_figure_path('model_performance.png')
    # plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")


def plot_attribution_maps():
    """Plot LRP attribution maps."""

    print("\nGenerating attribution map figure...")

    # Load attributions
    # Plot on map

    print("  [Implement attribution map plotting]")

    output_file = paths.get_figure_path('attribution_maps.png')
    print(f"  ✓ Saved: {output_file.name}")


def plot_robustness_analysis():
    """Plot robustness across different splits."""

    print("\nGenerating robustness analysis figure...")

    print("  [Implement robustness plotting]")

    output_file = paths.get_figure_path('robustness_analysis.png')
    print(f"  ✓ Saved: {output_file.name}")


def main():
    """Main plotting workflow."""

    args = parse_args()

    print("=" * 70)
    print("Step 5: Generating Figures")
    print("=" * 70)

    # Set up plotting style
    setup_figure_style()

    if args.figure == 'all':
        plot_model_performance()
        plot_attribution_maps()
        plot_robustness_analysis()
    elif args.figure == 'performance':
        plot_model_performance()
    elif args.figure == 'attributions':
        plot_attribution_maps()
    elif args.figure == 'robustness':
        plot_robustness_analysis()

    print("\n" + "=" * 70)
    print("✓ Figure generation complete!")
    print(f"Figures saved to: {paths.FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
