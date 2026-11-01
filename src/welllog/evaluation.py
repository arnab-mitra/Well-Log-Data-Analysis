"""FORCE 2020 penalty scoring and classification metrics."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

# Official ordering from the FORCE 2020 competition scoring script.
CLASS_CODES = (30000, 65030, 65000, 80000, 74000, 70000, 70032, 88000, 86000, 99000, 90000, 93000)
CLASS_NAMES = (
    "Sandstone",
    "Sandstone/Shale",
    "Shale",
    "Marl",
    "Dolomite",
    "Limestone",
    "Chalk",
    "Halite",
    "Anhydrite",
    "Tuff",
    "Coal",
    "Basement",
)
FORCE_PENALTY_MATRIX = np.array(
    [
        [0, 2, 3.5, 3, 3.75, 3.5, 3.5, 4, 4, 2.5, 3.875, 3.25],
        [2, 0, 2.375, 2.75, 4, 3.75, 3.75, 3.875, 4, 3, 3.75, 3],
        [3.5, 2.375, 0, 2, 3.5, 3.5, 3.75, 4, 4, 2.75, 3.25, 3],
        [3, 2.75, 2, 0, 2.5, 2, 2.25, 4, 4, 3.375, 3.75, 3.25],
        [3.75, 4, 3.5, 2.5, 0, 2.625, 2.875, 3.75, 3.25, 3, 4, 3.625],
        [3.5, 3.75, 3.5, 2, 2.625, 0, 1.375, 4, 3.75, 3.5, 4, 3.625],
        [3.5, 3.75, 3.75, 2.25, 2.875, 1.375, 0, 4, 3.75, 3.125, 4, 3.75],
        [4, 3.875, 4, 4, 3.75, 4, 4, 0, 2.75, 3.75, 3.75, 4],
        [4, 4, 4, 4, 3.25, 3.75, 3.75, 2.75, 0, 4, 4, 3.875],
        [2.5, 3, 2.75, 3.375, 3, 3.5, 3.125, 3.75, 4, 0, 2.5, 3.25],
        [3.875, 3.75, 3.25, 3.75, 4, 4, 4, 3.75, 4, 2.5, 0, 4],
        [3.25, 3, 3, 3.25, 3.625, 3.625, 3.75, 4, 3.875, 3.25, 4, 0],
    ],
    dtype=float,
)


def load_penalty_matrix(path: str | Path | None = None) -> np.ndarray:
    """Load a 12x12 matrix from NPY/CSV/XLSX, or use the official fallback."""
    if path is None:
        return FORCE_PENALTY_MATRIX.copy()
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        matrix = np.load(path, allow_pickle=False)
    elif suffix in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, header=None)
        numeric = raw.apply(pd.to_numeric, errors="coerce").dropna(how="all").dropna(axis=1, how="all")
        candidates = [numeric.to_numpy()]
        if numeric.shape[0] > 12:
            candidates.append(numeric.iloc[-12:].to_numpy())
        if numeric.shape[1] > 12:
            candidates.append(numeric.iloc[:, -12:].to_numpy())
        if numeric.shape[0] > 12 and numeric.shape[1] > 12:
            candidates.append(numeric.iloc[-12:, -12:].to_numpy())
        matrix = next((candidate for candidate in candidates if candidate.shape == (12, 12)), numeric)
    else:
        matrix = pd.read_csv(path, header=None).apply(pd.to_numeric, errors="coerce").to_numpy()
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (len(CLASS_CODES), len(CLASS_CODES)):
        raise ValueError(f"Penalty matrix must have shape (12, 12), got {matrix.shape}")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Penalty matrix must contain finite non-negative values")
    return matrix


def _class_indices(values: Sequence[int] | np.ndarray) -> np.ndarray:
    mapping = {code: index for index, code in enumerate(CLASS_CODES)}
    array = np.asarray(values)
    try:
        return np.array([mapping[int(value)] for value in array], dtype=int)
    except (KeyError, TypeError, ValueError) as exc:
        invalid = sorted(set(array.tolist()) - set(CLASS_CODES))
        raise ValueError(f"Unknown FORCE class codes: {invalid}") from exc


def force_score(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    penalty_matrix: np.ndarray | None = None,
) -> float:
    """Compute the official score: negative mean classification penalty."""
    truth, prediction = np.asarray(y_true), np.asarray(y_pred)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if truth.size == 0:
        raise ValueError("Cannot score empty arrays")
    matrix = load_penalty_matrix() if penalty_matrix is None else np.asarray(penalty_matrix, dtype=float)
    if matrix.shape != (12, 12):
        raise ValueError(f"Penalty matrix must have shape (12, 12), got {matrix.shape}")
    return -float(matrix[_class_indices(truth), _class_indices(prediction)].mean())


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    penalty_matrix: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Return practical scalar metrics including the official FORCE score."""
    truth, prediction = np.asarray(y_true), np.asarray(y_pred)
    return {
        "samples": int(truth.size),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=CLASS_CODES, average="macro", zero_division=0)),
        "force_score": force_score(truth, prediction, penalty_matrix),
    }


def confusion_frame(
    y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray
) -> pd.DataFrame:
    """Return a code-labelled confusion matrix in official class order."""
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_CODES)
    return pd.DataFrame(matrix, index=pd.Index(CLASS_CODES, name="true"), columns=pd.Index(CLASS_CODES, name="predicted"))


def uncertainty_from_probabilities(
    probabilities: np.ndarray,
    classes: Sequence[int] | np.ndarray = CLASS_CODES,
) -> pd.DataFrame:
    """Summarize probability predictions with confidence and normalized entropy."""
    probabilities = np.asarray(probabilities, dtype=float)
    classes = np.asarray(classes)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("Probability columns must match classes")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Probabilities must be finite and non-negative")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("Every probability row must sum to one")
    confidence = probabilities.max(axis=1)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-15, 1))).sum(axis=1)
    if probabilities.shape[1] > 1:
        entropy /= np.log(probabilities.shape[1])
    return pd.DataFrame(
        {
            "prediction": classes[probabilities.argmax(axis=1)],
            "confidence": confidence,
            "uncertainty": 1 - confidence,
            "entropy": entropy,
        }
    )
