#!/usr/bin/env python3
"""
Step 1: Download Data

Download and organize raw data from various sources.
This script handles downloading CESM2-LE, observational data, etc.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths


def main():
    """Download all required datasets."""

    print("=" * 70)
    print("Step 1: Downloading Data")
    print("=" * 70)

    # Check if data already exists
    if paths.CESM2_GRID_FILE.exists():
        print(f"✓ CESM2 grid file exists: {paths.CESM2_GRID_FILE}")
    else:
        print(f"✗ Missing: {paths.CESM2_GRID_FILE}")
        print("  Download from: [provide download instructions]")

    if paths.LANDMASK_FILE.exists():
        print(f"✓ Landmask file exists")
    else:
        print(f"✗ Missing landmask file")

    if paths.ERSST_FILE.exists():
        print(f"✓ ERSST file exists")
    else:
        print(f"✗ Missing ERSST file")

    print("\n" + "=" * 70)
    print("Data inventory complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
