"""
Observational Datasets

Subpackages:
- ersst: NOAA ERSSTv5 monthly mean SST (download, regrid, climate indices)
- nsidc: NSIDC Sea Ice Index (preprocess, slowdown/riles detection)
"""

from . import ersst
from . import nsidc

__all__ = ['ersst', 'nsidc']
