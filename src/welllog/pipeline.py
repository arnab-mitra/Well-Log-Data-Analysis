"""End-to-end deterministic training and evaluation pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from .evaluation import (
    CLASS_CODES,
    CLASS_NAMES,
    classification_metrics,
    confusion_frame,
    load_penalty_matrix,
)
from .features import build_features, default_feature_logs
from .io import TARGET_COLUMN, WELL_COLUMN, canonicalize_frame, load_las_directory
from .model import LithologyModel, group_train_test_split
from .qc import qc_report


@dataclass
class PipelineResult:
    """Artifacts returned by a pipeline run."""

    model: LithologyModel
    features: pd.DataFrame
    predictions: pd.DataFrame
    metrics: dict[str, float | int]
    confusion: pd.DataFrame
    train_wells: tuple[str, ...]
    test_wells: tuple[str, ...]
    qc: dict[str, Any]
    penalty_matrix: np.ndarray


def run_pipeline(
    data: pd.DataFrame | str | Path,
    *,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
    windows: tuple[int, ...] = (3, 7),
    penalty_matrix: np.ndarray | str | Path | None = None,
    max_iter: int = 150,
) -> PipelineResult:
    """Load/canonicalize data, split by well, train, and evaluate deterministically."""
    if isinstance(data, pd.DataFrame):
        frame = canonicalize_frame(data)
    else:
        frame = load_las_directory(data)
    if target_column not in frame:
        raise KeyError(f"Target column not found: {target_column}")
    labelled = frame.loc[frame[target_column].notna()].reset_index(drop=True)
    if labelled.empty:
        raise ValueError("No labelled rows are available")

    logs = default_feature_logs(labelled)
    features = build_features(labelled, log_columns=logs, windows=windows)
    train_idx, test_idx = group_train_test_split(
        labelled, test_size=test_size, random_state=random_state
    )
    model = LithologyModel(random_state=random_state, max_iter=max_iter)
    model.fit(features.iloc[train_idx], labelled[target_column].iloc[train_idx].astype(int))

    test_features = features.iloc[test_idx]
    uncertainty = model.uncertainty(test_features)
    uncertainty.index = labelled.index[test_idx]
    truth = labelled[target_column].iloc[test_idx].astype(int).to_numpy()
    prediction = uncertainty["prediction"].astype(int).to_numpy()
    matrix = load_penalty_matrix(penalty_matrix) if isinstance(penalty_matrix, (str, Path)) else penalty_matrix
    metrics = classification_metrics(truth, prediction, penalty_matrix=matrix)
    prediction_columns = [WELL_COLUMN, "DEPTH_MD", target_column, *logs]
    confidence_column = "FORCE_2020_LITHOFACIES_CONFIDENCE"
    if confidence_column in labelled:
        prediction_columns.append(confidence_column)
    predictions = labelled.loc[labelled.index[test_idx], prediction_columns].copy()
    predictions = predictions.rename(columns={target_column: "actual"})
    predictions = predictions.join(uncertainty)

    train_wells = tuple(sorted(labelled[WELL_COLUMN].iloc[train_idx].astype(str).unique()))
    test_wells = tuple(sorted(labelled[WELL_COLUMN].iloc[test_idx].astype(str).unique()))
    return PipelineResult(
        model=model,
        features=features,
        predictions=predictions.reset_index(drop=True),
        metrics=metrics,
        confusion=confusion_frame(truth, prediction),
        train_wells=train_wells,
        test_wells=test_wells,
        qc=qc_report(labelled),
        penalty_matrix=load_penalty_matrix() if matrix is None else np.asarray(matrix, dtype=float),
    )


def write_results(result: PipelineResult, output_directory: str | Path) -> Path:
    """Write stable, human-readable evaluation artifacts."""
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    test_predictions = result.predictions
    truth = test_predictions["actual"].astype(int).to_numpy()
    prediction = test_predictions["prediction"].astype(int).to_numpy()
    per_class = pd.DataFrame(
        classification_report(
            truth,
            prediction,
            labels=list(CLASS_CODES),
            target_names=list(CLASS_NAMES),
            output_dict=True,
            zero_division=0,
        )
    ).T.loc[list(CLASS_NAMES), ["precision", "recall", "f1-score", "support"]]
    per_class.index.name = "lithology"
    per_well_rows = []
    for well, group in test_predictions.groupby(WELL_COLUMN, sort=True):
        well_truth = group["actual"].astype(int).to_numpy()
        well_prediction = group["prediction"].astype(int).to_numpy()
        per_well_rows.append(
            {WELL_COLUMN: well, **classification_metrics(
                well_truth, well_prediction, penalty_matrix=result.penalty_matrix
            )}
        )
    per_well = pd.DataFrame(per_well_rows).set_index(WELL_COLUMN)
    confidence_metrics: dict[str, dict[str, float | int]] = {}
    confidence_column = "FORCE_2020_LITHOFACIES_CONFIDENCE"
    matrix = result.penalty_matrix
    if confidence_column in test_predictions:
        confidence_values = test_predictions[confidence_column]
        for value in sorted(confidence_values.dropna().unique()):
            mask = confidence_values.to_numpy() == value
            if mask.any():
                confidence_metrics[str(int(value))] = classification_metrics(
                    truth[mask], prediction[mask], penalty_matrix=matrix
                )
    payload = {
        "metrics": result.metrics,
        "train_wells": list(result.train_wells),
        "test_wells": list(result.test_wells),
        "confidence_strata": confidence_metrics,
    }
    result.predictions.to_csv(output / "predictions.csv", index=False, float_format="%.10g")
    result.confusion.to_csv(output / "confusion.csv")
    per_class.to_csv(output / "per_class_metrics.csv", float_format="%.10g")
    per_well.to_csv(output / "per_well_metrics.csv", float_format="%.10g")
    result.qc["missingness"].to_csv(output / "qc_missingness.csv", float_format="%.10g")
    result.qc["wells_summary"].to_csv(output / "qc_wells.csv", float_format="%.10g")
    result.qc["ranges"].to_csv(output / "qc_ranges.csv", float_format="%.10g")
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return output
