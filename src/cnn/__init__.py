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

from .splits import (
    load_jja_sst_demeaned,
    block_tvt_split,
    standardize,
    apply_landmask,
    save_tvt_split,
    load_tvt_split,
)

from .model import (
    build_cnn,
    focal_loss,
    compute_metrics,
)

from .train import (
    set_seed,
    compute_class_weights,
    train_model,
    predict_splits,
    collect_metrics_dataset,
)

__all__ = [
    # splits
    "load_jja_sst_demeaned",
    "block_tvt_split",
    "standardize",
    "apply_landmask",
    "save_tvt_split",
    "load_tvt_split",
    # model
    "build_cnn",
    "focal_loss",
    "compute_metrics",
    # train
    "set_seed",
    "compute_class_weights",
    "train_model",
    "predict_splits",
    "collect_metrics_dataset",
]
