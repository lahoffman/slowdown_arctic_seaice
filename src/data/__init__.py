"""
src.data — data loading, processing, and metrics sub-packages.

Sub-packages:
  cesm2le      — CESM2 Large Ensemble model output (download, combine, metrics, regrid)
  observations — Observational datasets
    └── ersst  — NOAA ERSSTv5 SST (download, regrid, climate indices)
"""

from . import cesm2le
from . import observations
