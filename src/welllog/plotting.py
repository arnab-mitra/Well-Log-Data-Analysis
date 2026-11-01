"""Non-interactive plotting helpers for QC and model evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .evaluation import CLASS_CODES, CLASS_NAMES
from .io import DEPTH_COLUMN


def plot_log_tracks(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    depth_column: str = DEPTH_COLUMN,
    title: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot selected curves on depth-aligned tracks."""
    columns = list(columns)
    if not columns:
        raise ValueError("At least one log column is required")
    missing = [column for column in [depth_column, *columns] if column not in frame]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    figure, axes = plt.subplots(1, len(columns), sharey=True, figsize=(3 * len(columns), 8), squeeze=False)
    axes_array = axes.ravel()
    depth = frame[depth_column]
    for axis, column in zip(axes_array, columns):
        axis.plot(frame[column], depth, linewidth=0.8)
        axis.set_xlabel(column)
        axis.grid(alpha=0.25)
    axes_array[0].set_ylabel(depth_column)
    axes_array[0].invert_yaxis()
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    return figure, axes_array


def plot_confusion(
    confusion: pd.DataFrame | np.ndarray,
    *,
    normalize: bool = False,
    axis: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a FORCE confusion matrix in official class order."""
    values = confusion.to_numpy(dtype=float) if isinstance(confusion, pd.DataFrame) else np.asarray(confusion, dtype=float)
    if values.shape != (12, 12):
        raise ValueError("Confusion matrix must have shape (12, 12)")
    if normalize:
        totals = values.sum(axis=1, keepdims=True)
        values = np.divide(values, totals, out=np.zeros_like(values), where=totals != 0)
    if axis is None:
        figure, axis = plt.subplots(figsize=(9, 8))
    else:
        figure = axis.figure
    image = axis.imshow(values, cmap="Blues", aspect="auto", vmin=0)
    labels = [f"{name}\n{code}" for name, code in zip(CLASS_NAMES, CLASS_CODES)]
    axis.set_xticks(range(12), labels=labels, rotation=90)
    axis.set_yticks(range(12), labels=labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    return figure, axis


def plot_uncertainty(
    predictions: pd.DataFrame,
    *,
    depth_column: str = DEPTH_COLUMN,
    uncertainty_column: str = "uncertainty",
    axis: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot uncertainty against measured depth."""
    if depth_column not in predictions or uncertainty_column not in predictions:
        raise KeyError(f"Required columns are {depth_column!r} and {uncertainty_column!r}")
    if axis is None:
        figure, axis = plt.subplots(figsize=(4, 8))
    else:
        figure = axis.figure
    axis.plot(predictions[uncertainty_column], predictions[depth_column], linewidth=0.8)
    axis.set_xlabel("Prediction uncertainty")
    axis.set_ylabel(depth_column)
    axis.set_xlim(0, 1)
    axis.invert_yaxis()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure, axis
