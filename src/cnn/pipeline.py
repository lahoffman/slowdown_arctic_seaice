"""
ML pipeline utilities for train-validate-test workflows.
"""

import numpy as np
import json
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


def create_tvt_splits(data, n_blocks=10, n_splits=9):
    """
    Create train-validate-test splits using block-based splitting.

    Parameters
    ----------
    data : ndarray
        Data array with shape (n_samples, ...)
    n_blocks : int, optional
        Number of blocks to split data into (default: 10)
    n_splits : int, optional
        Number of different split configurations (default: 9)

    Returns
    -------
    list of dict
        List of split configurations, each containing 'train', 'validate', 'test' indices
    """
    n_samples = data.shape[0]
    block_size = n_samples // n_blocks

    if n_samples % n_blocks != 0:
        print(f"Warning: {n_samples} samples doesn't divide evenly by {n_blocks} blocks")

    splits = []

    for k in range(n_splits):
        test_block = k
        val_block = (k + 1) % n_blocks
        train_blocks = [i for i in range(n_blocks) if i != test_block and i != val_block]

        # Get indices for each set
        test_indices = list(range(test_block * block_size, (test_block + 1) * block_size))
        val_indices = list(range(val_block * block_size, (val_block + 1) * block_size))

        train_indices = []
        for block in train_blocks:
            train_indices.extend(range(block * block_size, (block + 1) * block_size))

        splits.append({
            'split_idx': k,
            'train': np.array(train_indices),
            'validate': np.array(val_indices),
            'test': np.array(test_indices),
            'train_blocks': train_blocks,
            'val_block': val_block,
            'test_block': test_block
        })

    return splits


def train_model_with_config(model, X_train, y_train, X_val, y_val, config, callbacks=None):
    """
    Train a Keras model with specified configuration.

    Parameters
    ----------
    model : keras.Model
        Model to train
    X_train, y_train : ndarray
        Training data and labels
    X_val, y_val : ndarray
        Validation data and labels
    config : dict
        Training configuration
    callbacks : list, optional
        Additional Keras callbacks

    Returns
    -------
    tuple
        (trained_model, history)
    """
    if not ML_AVAILABLE:
        raise ImportError("TensorFlow not available")

    # Compile model
    model.compile(
        optimizer=config.get('optimizer', 'adam'),
        loss=config.get('loss', 'categorical_crossentropy'),
        metrics=config.get('metrics', ['accuracy'])
    )

    # Prepare callbacks
    if callbacks is None:
        callbacks = []

    # Add early stopping if configured
    if config.get('early_stopping'):
        es_config = config['early_stopping']
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=es_config.get('monitor', 'val_loss'),
                patience=es_config.get('patience', 10),
                restore_best_weights=es_config.get('restore_best_weights', True)
            )
        )

    # Compute class weights if needed
    class_weight = None
    if config.get('use_class_weights', False):
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(np.argmax(y_train, axis=1))
        weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=np.argmax(y_train, axis=1)
        )
        class_weight = {int(c): w for c, w in zip(classes, weights)}

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.get('epochs', 100),
        batch_size=config.get('batch_size', 32),
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    return model, history


def evaluate_model(model, X_test, y_test, return_predictions=False):
    """
    Evaluate model performance with comprehensive metrics.

    Parameters
    ----------
    model : keras.Model
        Trained model
    X_test : ndarray
        Test data
    y_test : ndarray
        True labels (one-hot encoded)
    return_predictions : bool, optional
        Whether to return predictions

    Returns
    -------
    dict or tuple
        Metrics dictionary, optionally with predictions
    """
    if not ML_AVAILABLE:
        raise ImportError("TensorFlow not available")

    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

    # Add per-class metrics
    n_classes = y_test.shape[1]
    for i in range(n_classes):
        metrics[f'precision_class_{i}'] = precision_score(
            y_true, y_pred, labels=[i], average=None, zero_division=0
        )[0] if i in y_true else 0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    # ROC-AUC if binary classification
    if n_classes == 2:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])

    if return_predictions:
        return metrics, y_pred, y_pred_proba

    return metrics


def save_model_and_metrics(model, metrics, model_path, metrics_path=None):
    """
    Save model and metrics.

    Parameters
    ----------
    model : keras.Model
        Model to save
    metrics : dict
        Metrics dictionary
    model_path : str or Path
        Path to save model
    metrics_path : str or Path, optional
        Path to save metrics (auto-generated if None)
    """
    if not ML_AVAILABLE:
        raise ImportError("TensorFlow not available")

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save metrics
    if metrics_path is None:
        metrics_path = model_path.with_suffix('.json')

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


def load_model_and_metrics(model_path, metrics_path=None):
    """
    Load model and metrics.

    Parameters
    ----------
    model_path : str or Path
        Path to model file
    metrics_path : str or Path, optional
        Path to metrics file (auto-detected if None)

    Returns
    -------
    tuple
        (model, metrics)
    """
    if not ML_AVAILABLE:
        raise ImportError("TensorFlow not available")

    model_path = Path(model_path)

    # Load model
    model = keras.models.load_model(model_path)

    # Load metrics
    if metrics_path is None:
        metrics_path = model_path.with_suffix('.json')

    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = None

    return model, metrics


def aggregate_metrics_across_splits(metrics_list):
    """
    Aggregate metrics across multiple splits.

    Parameters
    ----------
    metrics_list : list of dict
        List of metrics dictionaries from different splits

    Returns
    -------
    dict
        Aggregated metrics with mean and std
    """
    aggregated = {}

    # Get all metric keys (excluding confusion_matrix)
    keys = [k for k in metrics_list[0].keys() if k != 'confusion_matrix']

    for key in keys:
        values = [m[key] for m in metrics_list]
        aggregated[f'{key}_mean'] = np.mean(values)
        aggregated[f'{key}_std'] = np.std(values)
        aggregated[f'{key}_min'] = np.min(values)
        aggregated[f'{key}_max'] = np.max(values)

    return aggregated
