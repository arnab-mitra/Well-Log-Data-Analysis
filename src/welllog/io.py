"""LAS ingestion and canonical FORCE schema handling."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import lasio
import numpy as np
import pandas as pd

DEPTH_COLUMN = "DEPTH_MD"
WELL_COLUMN = "WELL"
TARGET_COLUMN = "FORCE_2020_LITHOFACIES_LITHOLOGY"
CONFIDENCE_COLUMN = "FORCE_2020_LITHOFACIES_CONFIDENCE"

CANONICAL_ALIASES = {
    "DEPTH": DEPTH_COLUMN,
    "DEPT": DEPTH_COLUMN,
    "MD": DEPTH_COLUMN,
    "DEPTH_MD": DEPTH_COLUMN,
    "WELL": WELL_COLUMN,
    "WELL_NAME": WELL_COLUMN,
    "LITHOLOGY": TARGET_COLUMN,
    "LITHOLOGY_CODE": TARGET_COLUMN,
    "FORCE_2020_LITHOFACIES_LITHOLOGY": TARGET_COLUMN,
    "LITHOLOGY_CONFIDENCE": CONFIDENCE_COLUMN,
    "FORCE_2020_LITHOFACIES_CONFIDENCE": CONFIDENCE_COLUMN,
}


def _canonical_name(name: object) -> str:
    normalized = str(name).strip().upper().replace(" ", "_").replace("-", "_")
    return CANONICAL_ALIASES.get(normalized, normalized)


def canonicalize_frame(frame: pd.DataFrame, well_name: str | None = None) -> pd.DataFrame:
    """Normalize curve names, nulls, types, ordering, and required identifiers."""
    result = frame.copy()
    result.columns = [_canonical_name(column) for column in result.columns]
    result = result.loc[:, ~result.columns.duplicated(keep="first")]

    if DEPTH_COLUMN not in result.columns:
        index_name = _canonical_name(result.index.name or "")
        if index_name == DEPTH_COLUMN or np.issubdtype(result.index.dtype, np.number):
            result.insert(0, DEPTH_COLUMN, result.index.to_numpy())
        else:
            raise ValueError(f"LAS data has no {DEPTH_COLUMN} curve")

    result.replace([np.inf, -np.inf, -999, -999.0, -999.25, -9999], np.nan, inplace=True)
    if well_name is not None:
        result[WELL_COLUMN] = str(well_name)
    elif WELL_COLUMN not in result.columns:
        raise ValueError(f"Data has no {WELL_COLUMN} identifier")
    else:
        result[WELL_COLUMN] = result[WELL_COLUMN].astype("string")

    for column in result.columns:
        if column != WELL_COLUMN:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.dropna(subset=[DEPTH_COLUMN]).sort_values(
        [WELL_COLUMN, DEPTH_COLUMN], kind="stable"
    )
    first = [WELL_COLUMN, DEPTH_COLUMN]
    remaining = sorted(column for column in result.columns if column not in first)
    return result[first + remaining].reset_index(drop=True)


def _las_well_name(las: lasio.LASFile, path: Path) -> str:
    for mnemonic in ("WELL", "UWI", "API"):
        try:
            value = las.well[mnemonic].value
        except (KeyError, AttributeError):
            continue
        if value is not None and str(value).strip():
            return str(value).strip()
    return path.stem


def load_las_file(path: str | Path, *, ignore_header_errors: bool = True) -> pd.DataFrame:
    """Read one LAS file and return rows in the canonical schema."""
    path = Path(path).expanduser().resolve()
    las = lasio.read(path, ignore_header_errors=ignore_header_errors)
    frame = las.df().reset_index()
    return canonicalize_frame(frame, well_name=_las_well_name(las, path))


def find_las_files(root: str | Path) -> list[Path]:
    """Find LAS files recursively in deterministic path order."""
    root = Path(root).expanduser().resolve()
    if root.is_file():
        if root.suffix.lower() != ".las":
            raise ValueError(f"Expected a .las file: {root}")
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".las"),
        key=lambda path: path.as_posix().lower(),
    )


def load_las_directory(
    root: str | Path,
    *,
    on_error: str = "raise",
    ignore_header_errors: bool = True,
) -> pd.DataFrame:
    """Recursively load LAS files into one canonical, deterministically sorted frame."""
    if on_error not in {"raise", "skip"}:
        raise ValueError("on_error must be 'raise' or 'skip'")
    frames: list[pd.DataFrame] = []
    errors: list[tuple[Path, Exception]] = []
    for path in find_las_files(root):
        try:
            frames.append(load_las_file(path, ignore_header_errors=ignore_header_errors))
        except Exception as exc:
            if on_error == "raise":
                raise RuntimeError(f"Failed to read LAS file {path}") from exc
            errors.append((path, exc))
    if not frames:
        detail = f"; {len(errors)} failed" if errors else ""
        raise ValueError(f"No readable LAS files found under {Path(root)}{detail}")
    return canonicalize_frame(pd.concat(frames, ignore_index=True))


def select_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Select canonicalized column names while retaining well and depth identifiers."""
    names = [_canonical_name(column) for column in columns]
    requested = list(dict.fromkeys([WELL_COLUMN, DEPTH_COLUMN, *names]))
    missing = [column for column in requested if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    return frame.loc[:, requested].copy()
