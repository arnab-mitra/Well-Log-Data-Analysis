"""Data-quality summaries for well-log tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import DEPTH_COLUMN, WELL_COLUMN


def missingness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing and finite values for every column."""
    rows = []
    total = len(frame)
    for column in frame.columns:
        series = frame[column]
        missing = int(series.isna().sum())
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "count": total,
                "present": total - missing,
                "missing": missing,
                "missing_fraction": missing / total if total else np.nan,
                "unique": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def well_summary(
    frame: pd.DataFrame,
    *,
    well_column: str = WELL_COLUMN,
    depth_column: str = DEPTH_COLUMN,
) -> pd.DataFrame:
    """Report row count, depth coverage, sampling, duplicates, and missing cells per well."""
    if well_column not in frame or depth_column not in frame:
        raise KeyError(f"Required columns are {well_column!r} and {depth_column!r}")
    numeric_logs = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column != depth_column
    ]
    rows = []
    for well, group in frame.groupby(well_column, sort=True, dropna=False):
        depth = pd.to_numeric(group[depth_column], errors="coerce")
        ordered = np.sort(depth.dropna().unique())
        spacing = np.diff(ordered)
        log_cells = len(group) * len(numeric_logs)
        missing_logs = int(group[numeric_logs].isna().sum().sum()) if numeric_logs else 0
        rows.append(
            {
                well_column: well,
                "rows": len(group),
                "depth_min": depth.min(),
                "depth_max": depth.max(),
                "depth_span": depth.max() - depth.min(),
                "median_step": float(np.median(spacing)) if len(spacing) else np.nan,
                "duplicate_depths": int(depth.duplicated(keep=False).sum()),
                "non_monotonic_steps": int((depth.diff().dropna() < 0).sum()),
                "missing_log_cells": missing_logs,
                "missing_log_fraction": missing_logs / log_cells if log_cells else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index(well_column)


def range_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return robust descriptive statistics for numeric columns."""
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame(
            columns=["count", "mean", "std", "min", "p01", "p50", "p99", "max"]
        )
    description = numeric.describe(percentiles=[0.01, 0.5, 0.99]).T
    return description.rename(columns={"1%": "p01", "50%": "p50", "99%": "p99"})[
        ["count", "mean", "std", "min", "p01", "p50", "p99", "max"]
    ]


def qc_report(frame: pd.DataFrame) -> dict[str, pd.DataFrame | int]:
    """Build the complete deterministic QC report."""
    return {
        "rows": len(frame),
        "wells": int(frame[WELL_COLUMN].nunique()) if WELL_COLUMN in frame else 0,
        "missingness": missingness_summary(frame),
        "wells_summary": well_summary(frame),
        "ranges": range_summary(frame),
    }
