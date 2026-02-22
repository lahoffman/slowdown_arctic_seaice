#!/usr/bin/env python3
"""
Step 4: Compute XAI Attributions

Compute LRP attributions for trained models.
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths, training
from slowdown.data import load_netcdf, save_netcdf
from slowdown.models import load_model
from slowdown.xai import compute_lrp_analysis


def parse_args():
    parser = argparse.ArgumentParser(description='Compute XAI attributions')
    parser.add_argument('--split', type=int, default=None,
                        help='Compute only this split (default: all splits)')
    parser.add_argument('--seed', type=int, default=training.TRAINING_CONFIG['random_seeds'][0],
                        help='Random seed')
    return parser.parse_args()


def compute_attribution_for_split(split_idx, seed):
    """Compute LRP attributions for a single split."""

    print(f"\n{'=' * 70}")
    print(f"Computing Attributions for Split {split_idx}")
    print('=' * 70)

    # Load model
    model_file = paths.get_model_path(f'model_split_{split_idx}_seed_{seed}.h5')
    if not model_file.exists():
        print(f"✗ Model not found: {model_file}")
        print("  Run 03_train_models.py first!")
        return

    model = load_model(model_file)

    # Load test data
    data_file = paths.get_data_path(f'tvt_split_{split_idx}.nc')
    data = load_netcdf(data_file)

    # TODO: Extract test data
    # X_test = data['sst_test']

    print("  [Implement LRP computation here]")

    # Compute LRP
    # attributions = compute_lrp_analysis(model, X_test, method='lrp.z')

    # Save attributions
    output_file = paths.get_attribution_path(f'lrp_split_{split_idx}.nc')
    # save_netcdf(..., output_file)

    print(f"  ✓ Attributions saved: {output_file.name}")


def main():
    """Main XAI workflow."""

    args = parse_args()

    print("=" * 70)
    print("Step 4: Computing XAI Attributions")
    print("=" * 70)

    if args.split is not None:
        # Single split
        compute_attribution_for_split(args.split, args.seed)
    else:
        # All splits
        for split_idx in range(training.SPLIT_CONFIG['n_splits']):
            compute_attribution_for_split(split_idx, args.seed)

    print("\n" + "=" * 70)
    print("✓ XAI computation complete!")
    print(f"Attributions saved to: {paths.ATTRIBUTIONS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
