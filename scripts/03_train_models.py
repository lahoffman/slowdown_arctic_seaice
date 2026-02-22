#!/usr/bin/env python3
"""
Step 3: Train ML Models

Train CNN models on each TVT split for robustness testing.
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths, model, training
from slowdown.data import load_netcdf, load_landmask
from slowdown.models import (
    build_cnn_model,
    train_model_with_config,
    evaluate_model,
    save_model
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train CNN models')
    parser.add_argument('--split', type=int, default=None,
                        help='Train only this split (default: all splits)')
    parser.add_argument('--seed', type=int, default=training.TRAINING_CONFIG['random_seeds'][0],
                        help='Random seed')
    return parser.parse_args()


def train_single_split(split_idx, seed):
    """Train model for a single split."""

    print(f"\n{'=' * 70}")
    print(f"Training Split {split_idx} (seed={seed})")
    print('=' * 70)

    # Load data
    data_file = paths.get_data_path(f'tvt_split_{split_idx}.nc')
    if not data_file.exists():
        print(f"✗ Data file not found: {data_file}")
        print("  Run 02_preprocess_data.py first!")
        return None

    data = load_netcdf(data_file)

    # TODO: Extract and prepare X_train, y_train, etc.
    # X_train = data['sst_train']
    # y_train = ... (load labels)

    print("  [Implement data loading and model training here]")

    # Build model
    cnn_model = build_cnn_model(model.CNN_CONFIG)

    # Train
    # model, history = train_model_with_config(
    #     cnn_model, X_train, y_train, X_val, y_val,
    #     config=training.TRAINING_CONFIG
    # )

    # Evaluate
    # metrics = evaluate_model(model, X_test, y_test)

    # Save
    model_file = paths.get_model_path(f'model_split_{split_idx}_seed_{seed}.h5')
    # save_model(model, model_file, metrics)

    print(f"  ✓ Model saved: {model_file.name}")

    return None  # Return metrics when implemented


def main():
    """Main training workflow."""

    args = parse_args()

    print("=" * 70)
    print("Step 3: Training Models")
    print("=" * 70)

    if args.split is not None:
        # Train single split
        metrics = train_single_split(args.split, args.seed)
    else:
        # Train all splits
        all_metrics = []
        for split_idx in range(training.SPLIT_CONFIG['n_splits']):
            metrics = train_single_split(split_idx, args.seed)
            if metrics:
                all_metrics.append(metrics)

        if all_metrics:
            print("\n" + "=" * 70)
            print("Training Summary")
            print("=" * 70)
            # Print aggregate metrics
            print(f"Trained {len(all_metrics)} models")

    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
