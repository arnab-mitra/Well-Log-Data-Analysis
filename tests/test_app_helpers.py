import json
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import (
    discover_artifacts,
    find_column,
    flatten_metrics,
    prepare_predictions,
    uncertainty_series,
)


def test_discover_artifacts_prefers_canonical_outputs(tmp_path: Path) -> None:
    run = tmp_path / "run-001"
    run.mkdir()
    (run / "prediction.csv").write_text("well,depth\nA,1\n", encoding="utf-8")
    (run / "predictions.csv").write_text("well,depth\nA,1\n", encoding="utf-8")
    (run / "metrics.json").write_text(json.dumps({"accuracy": 0.8}), encoding="utf-8")
    (run / "well_qc.csv").write_text("well,status\nA,pass\n", encoding="utf-8")

    bundle = discover_artifacts(tmp_path)

    assert bundle.ready
    assert bundle.predictions == run / "predictions.csv"
    assert bundle.metrics == run / "metrics.json"
    assert bundle.qc == (run / "well_qc.csv",)


def test_discover_artifacts_returns_informative_empty_bundle(tmp_path: Path) -> None:
    bundle = discover_artifacts(tmp_path / "missing")

    assert not bundle.ready
    assert bundle.predictions is None
    assert bundle.qc == ()


def test_prepare_predictions_handles_aliases_and_sorts_depth() -> None:
    frame = pd.DataFrame(
        {
            "WELL_NAME": ["B", "A", "A"],
            "DEPT": [100.0, 102.0, 101.0],
            "y_pred": [2, 1, 2],
            "target": [2, 2, 2],
        }
    )

    prepared, columns = prepare_predictions(frame)

    assert columns == {
        "well": "WELL_NAME",
        "depth": "DEPT",
        "prediction": "y_pred",
        "reference": "target",
        "uncertainty": None,
        "confidence": None,
    }
    assert prepared.loc[prepared["WELL_NAME"] == "A", "DEPT"].tolist() == [101.0, 102.0]


def test_prepare_predictions_recognizes_force_depth_name() -> None:
    frame = pd.DataFrame({"WELL": ["A"], "DEPTH_MD": [1234.5], "prediction": [65000]})

    prepared, columns = prepare_predictions(frame)

    assert columns["depth"] == "DEPTH_MD"
    assert prepared["DEPTH_MD"].iloc[0] == pytest.approx(1234.5)


def test_prepare_predictions_supplies_project_well_when_missing() -> None:
    prepared, columns = prepare_predictions(pd.DataFrame({"depth": [1, 2]}))

    assert columns["well"] == "well"
    assert prepared["well"].tolist() == ["Project", "Project"]


def test_uncertainty_is_derived_from_percentage_confidence() -> None:
    frame = pd.DataFrame({"prediction_confidence": [90, 60, None]})
    columns = {"uncertainty": None, "confidence": "prediction_confidence"}

    result = uncertainty_series(frame, columns)

    assert result.iloc[0] == pytest.approx(0.1)
    assert result.iloc[1] == pytest.approx(0.4)
    assert pd.isna(result.iloc[2])


def test_flatten_metrics_keeps_numeric_leaves_only() -> None:
    metrics = {
        "test": {"accuracy": 0.91, "f1_macro": 0.82},
        "folds": 5,
        "model": "random_forest",
        "calibrated": True,
    }

    assert flatten_metrics(metrics) == {
        "test.accuracy": 0.91,
        "test.f1_macro": 0.82,
        "folds": 5.0,
    }


def test_find_column_normalises_common_spelling_variants() -> None:
    assert find_column(["Measured Depth", "GR"], ["measured_depth"]) == "Measured Depth"
