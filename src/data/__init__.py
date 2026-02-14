"""
Analysis functions for climate indices and statistical operations.
"""

from .climate_indices import (
    compute_nino34_index,
    compute_ipo_index,
    label_enso_phases
)

from .statistics import (
    compute_anomalies,
    compute_climatology,
    normalize_by_std
)

from .trends import (
    compute_linear_trend,
    remove_linear_trend
)

__all__ = [
    'compute_nino34_index',
    'compute_ipo_index',
    'label_enso_phases',
    'compute_anomalies',
    'compute_climatology',
    'normalize_by_std',
    'compute_linear_trend',
    'remove_linear_trend'
]
