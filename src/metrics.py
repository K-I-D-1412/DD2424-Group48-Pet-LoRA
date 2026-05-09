"""
Classification metric utilities.

Thin wrappers around sklearn so we get standard, well-tested implementations
of macro F1, per-class accuracy and per-class F1. They operate on numpy arrays
of integer class indices (or anything convertible to such).

Usage:
    from src.metrics import (
        compute_top1_accuracy,
        compute_macro_f1,
        compute_per_class_accuracy,
        compute_per_class_f1,
        compute_confusion_matrix,
    )
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix as sk_confusion_matrix


def _as_int_array(x) -> np.ndarray:
    """Convert input (list / torch tensor / numpy) to a 1-D int64 numpy array."""
    arr = np.asarray(x)
    return arr.astype(np.int64).reshape(-1)


def compute_top1_accuracy(predictions, labels) -> float:
    """Top-1 accuracy = fraction of correctly predicted samples."""
    predictions = _as_int_array(predictions)
    labels = _as_int_array(labels)
    if predictions.size == 0:
        return 0.0
    return float((predictions == labels).mean())


def compute_macro_f1(predictions, labels, num_classes: int | None = None) -> float:
    """
    Macro F1 score: the unweighted average of per-class F1.
    Use this rather than micro-F1, which equals accuracy in the multi-class
    single-label setting.
    """
    predictions = _as_int_array(predictions)
    labels = _as_int_array(labels)
    if predictions.size == 0:
        return 0.0

    if num_classes is not None:
        label_list = list(range(num_classes))
        return float(
            f1_score(
                labels,
                predictions,
                labels=label_list,
                average="macro",
                zero_division=0,
            )
        )

    return float(f1_score(labels, predictions, average="macro", zero_division=0))


def compute_per_class_accuracy(predictions, labels, num_classes: int) -> list:
    """
    Per-class accuracy (recall) for each of `num_classes` classes.
    For a class that is absent from `labels`, returns 0.0 for that class.
    """
    predictions = _as_int_array(predictions)
    labels = _as_int_array(labels)

    accs = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            accs.append(0.0)
        else:
            accs.append(float((predictions[mask] == c).mean()))
    return accs


def compute_per_class_f1(predictions, labels, num_classes: int) -> list:
    """Per-class F1 score, returned as a Python list of length `num_classes`."""
    predictions = _as_int_array(predictions)
    labels = _as_int_array(labels)
    label_list = list(range(num_classes))

    return f1_score(
        labels,
        predictions,
        labels=label_list,
        average=None,
        zero_division=0,
    ).tolist()


def compute_confusion_matrix(predictions, labels, num_classes: int) -> list:
    """
    Confusion matrix as a nested Python list (so it serializes to JSON).
    Rows = true labels, columns = predictions.
    """
    predictions = _as_int_array(predictions)
    labels = _as_int_array(labels)
    label_list = list(range(num_classes))

    cm = sk_confusion_matrix(labels, predictions, labels=label_list)
    return cm.tolist()