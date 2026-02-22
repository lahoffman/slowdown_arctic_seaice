"""
slowdown.data.observations.nsidc
---------------------------------
NSIDC Sea Ice Index processing.

  preprocess      — load, interpolate, and clean monthly SIE from Excel
  define_slowdown — compute decadal trends; define slowdown / riles events
"""

from .preprocess import (
    load_nsidc_sie,
    preprocess_nsidc_sie,
)
from .define_slowdown import (
    compute_decadal_trends,
    define_slowdown_threshold,
    define_riles_threshold,
    save_slowdown_thresholds,
    save_slowdown_events,
    save_riles_events,
)
