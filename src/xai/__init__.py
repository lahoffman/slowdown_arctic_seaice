"""
src/xai — Explainable AI (XAI) modules for the arctic sea ice slowdown project.

Modules
-------
lrp : Layer-wise Relevance Propagation (LRP-z) attributions using iNNvestigate.
      Includes model preparation (sigmoid stripping), chunked analysis, and
      save/load utilities.
"""

from .lrp import (
    strip_sigmoid,
    compute_lrp_z,
    save_lrp,
    load_lrp,
)

__all__ = [
    "strip_sigmoid",
    "compute_lrp_z",
    "save_lrp",
    "load_lrp",
]
