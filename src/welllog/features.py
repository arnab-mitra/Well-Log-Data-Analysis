"""Leakage-safe, well-local feature engineering."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .io import CONFIDENCE_COLUMN, DEPTH_COLUMN, TARGET_COLUMN, WELL_COLUMN

DEFAULT_LOG_COLUMNS = ("GR", "RHOB", "NPHI", "DTC", "PEF", "SP", "BS", "DRHO")


def available_log_columns(
    frame: pd.DataFrame, columns: Sequence[str] = DEFAULT_LOG_COLUMNS
) -> list[str]:
    """Return requested numeric logs present in a frame."""
    return [
        column
        for column in columns
        if column in frame and pd.api.types.is_numeric_dtype(frame[column])
    ]


def build_features(
    frame: pd.DataFrame,
    *,
    log_columns: Sequence[str] | None = None,
    windows: Sequence[int] = (3, 7),
    include_depth: bool = True,
) -> pd.DataFrame:
    """Compute numerical features independently inside each well.

    Every shift, difference, and rolling window is grouped by ``WELL``. Thus a
    feature on the first row of one well cannot use a value from another well.
    Output rows retain the input index and order.
    """
    if WELL_COLUMN not in frame or DEPTH_COLUMN not in frame:
        raise KeyError(f"Required columns are {WELL_COLUMN!r} and {DEPTH_COLUMN!r}")
    if log_columns is None:
        logs = available_log_columns(frame)
    else:
        missing = [column for column in log_columns if column not in frame]
        if missing:
            raise KeyError(f"Missing log columns: {missing}")
        logs = list(dict.fromkeys(log_columns))
    if not logs:
        raise ValueError("No numerical log columns were selected")
    normalized_windows = tuple(sorted({int(window) for window in windows}))
    if any(window < 2 for window in normalized_windows):
        raise ValueError("Feature windows must be integers >= 2")

    working = frame[[WELL_COLUMN, DEPTH_COLUMN, *logs]].copy()
    working["__position"] = np.arange(len(working))
    working = working.sort_values([WELL_COLUMN, DEPTH_COLUMN, "__position"], kind="stable")
    groups = working.groupby(WELL_COLUMN, sort=False, dropna=False)
    features = pd.DataFrame(index=working.index)
    if include_depth:
        features[DEPTH_COLUMN] = pd.to_numeric(working[DEPTH_COLUMN], errors="coerce")

    depth_step = groups[DEPTH_COLUMN].diff().replace(0, np.nan)
    for column in logs:
        values = pd.to_numeric(working[column], errors="coerce")
        features[column] = values
        features[f"{column}__missing"] = values.isna().astype("int8")
        features[f"{column}__gradient"] = groups[column].diff() / depth_step
        for window in normalized_windows:
            roller = groups[column].rolling(window=window, center=True, min_periods=1)
            features[f"{column}__mean_{window}"] = roller.mean().reset_index(level=0, drop=True)
            features[f"{column}__std_{window}"] = roller.std(ddof=0).reset_index(level=0, drop=True)

    return features.loc[frame.index]


def default_feature_logs(frame: pd.DataFrame) -> list[str]:
    """Choose inference-time wireline logs only.

    Lithology confidence describes the human interpretation and is therefore
    label metadata, not a measurement available to a deployed classifier.
    """
    return [
        column
        for column in available_log_columns(frame)
        if column not in {TARGET_COLUMN, CONFIDENCE_COLUMN}
    ]
