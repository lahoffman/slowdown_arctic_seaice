"""
src/cnn — CNN training pipeline for arctic sea ice slowdown prediction.

Modules
-------
splits  : JJA SST preparation, block-based TVT splitting, standardisation,
          and save/load helpers.
model   : CNN architecture (build_cnn), custom loss functions, and metrics.
train   : Seed setting, class-weight computation, training loop, prediction,
          and metrics collection across splits.
"""

# Submodules are imported directly in each script to avoid pulling in
# heavy optional dependencies (sklearn, tensorflow, etc.) unnecessarily.
#   from src.cnn.splits import ...
#   from src.cnn.model  import ...
#   from src.cnn.train  import ...
