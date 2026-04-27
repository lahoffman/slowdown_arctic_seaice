"""
CNN architecture, loss functions, and evaluation metrics.

Contents
--------
build_cnn         : Construct the 2-layer convolutional classifier used throughout
                    the project.
focal_loss        : Keras-compatible focal loss for handling class imbalance.
compute_metrics   : Compute a comprehensive dict of classification metrics from
                    predicted probabilities and binary truth labels.

The model takes spatial maps of shape ``(n_samples, nx, ny, nch)`` and outputs
a scalar sigmoid probability for each sample.  For the JJA-only version
``nch = 1`` (one channel: the JJA mean SST map).

Hyperparameter defaults mirror the values used in the original M1 script and
are exposed as module-level constants so they can be overridden without editing
function signatures.

Authors: Lauren Hoffman  <lhoffma2@ucsc.edu>
"""

import numpy as np
from typing import Dict

# Keras / TensorFlow imports are deferred to avoid import-time overhead when
# only the metric functions are needed (e.g. during data processing steps).
try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import (
        Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    )
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.models import Model
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix, 
    ConfusionMatrixDisplay,
)


# =============================================================================
# Hyperparameter defaults
# =============================================================================

#: L2 regularisation strength applied to convolutional kernels.
RL2 = 1e-5

#: Dropout rate applied to the flattened representation before the output layer.
DROP = 0.2

#: Focal-loss gamma (focusing parameter).  Higher values down-weight easy examples.
FOCAL_GAMMA = 2.0

#: Focal-loss alpha (class-balance weight for the positive class).
FOCAL_ALPHA = 0.75


# =============================================================================
# Loss functions
# =============================================================================

def norm_root_mean_squared_error(y_true, y_pred):
    """
    Normalised root mean squared error (NRMSE), Keras metric-compatible.

    NRMSE = RMSE / std(y_true)

    Useful as a secondary monitoring metric during training.
    """
    return (K.sqrt(K.mean(K.square(y_pred - y_true)))) / K.std(y_true)


def pearson_correlation(y_true, y_pred):
    """
    Pearson correlation coefficient, Keras metric-compatible.

    Used as a secondary monitoring metric.  Note: for binary classification
    the primary metric is AUPRC; this is supplementary.
    """
    num = K.sum((y_true - K.mean(y_true)) * (y_pred - K.mean(y_pred)))
    den = (K.sqrt(K.sum(K.square(y_true - K.mean(y_true)))) *
           K.sqrt(K.sum(K.square(y_pred - K.mean(y_pred)))))
    return num / (den + K.epsilon())


# =============================================================================
# Model architecture
# =============================================================================

def build_cnn(
    nx: int,
    ny: int,
    nch: int,
    rl2: float = RL2,
    drop: float = DROP,
) -> "Model":
    """
    Build the binary-classification CNN used for slowdown prediction.
    """
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow is required to build the CNN model.")

    inputs = Input(shape=(nx, ny, nch))

    x = Conv2D(32, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(rl2))(inputs)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(rl2))(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dropout(drop)(x)

    output = Dense(1, activation='sigmoid')(x)

    return Model(inputs=inputs, outputs=output)


# =============================================================================
# Evaluation metrics
# =============================================================================

#: Ordered list of metric names returned by compute_metrics.
METRIC_NAMES = [
    "AUPRC",
    "AUROC",
    "Brier",
    "Precision",
    "Recall",
    "F1",
    "Accuracy",
    "Threshold",
    "Prec@Thr",
    "Rec@Thr",
    "Prevalence",
    "BinaryCrossEntropy",
    "MCC",
]


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """
    Compute a comprehensive set of binary-classification evaluation metrics.

    The operating threshold is chosen as the point on the precision–recall
    curve where precision and recall are closest to one another (the
    precision–recall intersection).  All point metrics (Precision, Recall, F1,
    Accuracy, MCC) are computed at this threshold.

    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)

    # Choose threshold at the precision–recall intersection
    if thresholds.size == 0:
        thr          = 0.5
        prec_at_thr  = precision[0]
        rec_at_thr   = recall[0]
    else:
        idx         = np.argmin(np.abs(precision[:-1] - recall[:-1]))
        thr         = float(thresholds[idx])
        prec_at_thr = float(precision[idx])
        rec_at_thr  = float(recall[idx])

    y_pred = (y_scores >= thr).astype(int)

    return {
        "AUPRC":              float(auprc),
        "AUROC":              float(roc_auc_score(y_true, y_scores)),
        "Brier":              float(brier_score_loss(y_true, y_scores)),
        "Precision":          float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall":             float(recall_score(y_true, y_pred, zero_division=0)),
        "F1":                 float(f1_score(y_true, y_pred, zero_division=0)),
        "Accuracy":           float(accuracy_score(y_true, y_pred)),
        "Threshold":          thr,
        "Prec@Thr":           prec_at_thr,
        "Rec@Thr":            rec_at_thr,
        "Prevalence":         float(np.mean(y_true)),
        "BinaryCrossEntropy": float(log_loss(y_true, y_scores, labels=[0, 1])),
        "MCC":                float(matthews_corrcoef(y_true, y_pred)),
    }
