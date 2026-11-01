from __future__ import annotations

import hashlib
import io
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from welllog.cli import main
from welllog.download import download_force_archive, md5sum
from welllog.evaluation import (
    CLASS_CODES,
    FORCE_PENALTY_MATRIX,
    classification_metrics,
    force_score,
    uncertainty_from_probabilities,
)
from welllog.features import build_features, default_feature_logs
from welllog.io import load_las_directory
from welllog.model import group_train_test_split
from welllog.pipeline import run_pipeline, write_results
from welllog.plotting import plot_confusion, plot_log_tracks, plot_uncertainty
from welllog.qc import qc_report

LAS_TEMPLATE = """~Version Information
VERS. 2.0
WRAP. NO
~Well Information
STRT.M 1000.0
STOP.M 1002.0
STEP.M 1.0
NULL. -999.25
WELL. {well}
~Curve Information
DEPT.M : Depth
GR.API : Gamma ray
RHOB.G/C3 : Density
FORCE_2020_LITHOFACIES_LITHOLOGY. : target
~ASCII
1000 10 2.10 30000
1001 -999.25 2.20 65030
1002 30 2.30 65000
"""


def synthetic_frame(wells: int = 6, rows: int = 36) -> pd.DataFrame:
    records = []
    codes = CLASS_CODES[:3]
    for well_number in range(wells):
        for row in range(rows):
            klass = codes[(row // 12 + well_number) % len(codes)]
            class_index = codes.index(klass)
            records.append(
                {
                    "WELL": f"W{well_number}",
                    "DEPTH_MD": 1000 + row * 0.5,
                    "GR": class_index * 50 + (row % 4),
                    "RHOB": 2.0 + class_index * 0.25 + (row % 3) * 0.01,
                    "NPHI": 0.35 - class_index * 0.1,
                    "FORCE_2020_LITHOFACIES_LITHOLOGY": klass,
                }
            )
    frame = pd.DataFrame(records)
    frame.loc[5, "GR"] = np.nan
    return frame


def test_downloader_verifies_md5_and_is_atomic(tmp_path: Path) -> None:
    payload = b"synthetic force archive"
    digest = hashlib.md5(payload, usedforsecurity=False).hexdigest()

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    calls = []

    def opener(url, timeout):
        calls.append((url, timeout))
        return Response(payload)

    destination = tmp_path / "force.zip"
    assert download_force_archive(destination, expected_md5=digest, opener=opener) == destination
    assert md5sum(destination) == digest
    assert len(calls) == 1
    download_force_archive(destination, expected_md5=digest, opener=opener)
    assert len(calls) == 1

    bad = tmp_path / "bad.zip"
    with pytest.raises(ValueError, match="MD5"):
        download_force_archive(bad, expected_md5="0" * 32, opener=opener)
    assert not bad.exists()
    assert not list(tmp_path.glob("*.part"))


def test_recursive_las_loader_canonicalizes_and_sorts(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.LAS").write_text(LAS_TEMPLATE.format(well="B"), encoding="ascii")
    (tmp_path / "a.las").write_text(LAS_TEMPLATE.format(well="A"), encoding="ascii")
    frame = load_las_directory(tmp_path)
    assert list(frame.columns[:2]) == ["WELL", "DEPTH_MD"]
    assert list(frame.groupby("WELL", sort=True).size()) == [3, 3]
    assert frame.iloc[0]["WELL"] == "A"
    assert np.isnan(frame.loc[(frame.WELL == "A") & (frame.DEPTH_MD == 1001), "GR"]).all()


def test_qc_summaries_report_missing_and_depth_properties() -> None:
    report = qc_report(synthetic_frame(wells=2, rows=8))
    assert report["rows"] == 16
    assert report["wells"] == 2
    assert report["missingness"].loc["GR", "missing"] == 1
    assert (report["wells_summary"]["median_step"] == 0.5).all()
    assert report["wells_summary"]["duplicate_depths"].sum() == 0


def test_features_do_not_cross_well_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "WELL": ["A", "A", "A", "B", "B", "B"],
            "DEPTH_MD": [1, 2, 3, 1, 2, 3],
            "GR": [1.0, 2.0, 3.0, 100.0, 100.0, 100.0],
        },
        index=[9, 3, 7, 5, 2, 8],
    )
    features = build_features(frame, log_columns=["GR"], windows=(3,))
    assert np.isnan(features.loc[9, "GR__gradient"])
    assert np.isnan(features.loc[5, "GR__gradient"])
    assert features.loc[5, "GR__mean_3"] == 100.0
    assert list(features.index) == list(frame.index)


def test_default_features_exclude_interpretation_metadata() -> None:
    frame = synthetic_frame(wells=2, rows=8)
    frame["FORCE_2020_LITHOFACIES_CONFIDENCE"] = 1
    logs = default_feature_logs(frame)
    assert "FORCE_2020_LITHOFACIES_CONFIDENCE" not in logs
    assert "FORCE_2020_LITHOFACIES_LITHOLOGY" not in logs
    assert {"GR", "RHOB", "NPHI"}.issubset(logs)


def test_group_split_is_disjoint_and_deterministic() -> None:
    frame = synthetic_frame()
    first = group_train_test_split(frame, test_size=0.33, random_state=7)
    second = group_train_test_split(frame, test_size=0.33, random_state=7)
    assert np.array_equal(first[0], second[0])
    train_wells = set(frame.iloc[first[0]].WELL)
    test_wells = set(frame.iloc[first[1]].WELL)
    assert train_wells.isdisjoint(test_wells)


def test_official_force_score_metrics_and_uncertainty() -> None:
    truth = np.array([CLASS_CODES[0], CLASS_CODES[5], CLASS_CODES[11]])
    prediction = np.array([CLASS_CODES[0], CLASS_CODES[6], CLASS_CODES[0]])
    expected = -(0 + 1.375 + 3.25) / 3
    assert force_score(truth, prediction) == pytest.approx(expected)
    custom = np.ones((12, 12))
    np.fill_diagonal(custom, 0)
    assert force_score(truth, prediction, custom) == pytest.approx(-2 / 3)
    metrics = classification_metrics(truth, prediction)
    assert metrics["samples"] == 3
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    probabilities = np.eye(3)[[0, 1]]
    uncertainty = uncertainty_from_probabilities(probabilities, classes=truth)
    assert (uncertainty["confidence"] == 1).all()
    assert (uncertainty["entropy"] == 0).all()
    with pytest.raises(ValueError, match="Unknown"):
        force_score([123], [CLASS_CODES[0]])
    assert FORCE_PENALTY_MATRIX.shape == (12, 12)


def test_pipeline_is_deterministic_grouped_and_writes_results(tmp_path: Path) -> None:
    frame = synthetic_frame()
    result_a = run_pipeline(frame, test_size=0.33, random_state=13, max_iter=25, windows=(3,))
    result_b = run_pipeline(frame, test_size=0.33, random_state=13, max_iter=25, windows=(3,))
    assert set(result_a.train_wells).isdisjoint(result_a.test_wells)
    pd.testing.assert_frame_equal(result_a.predictions, result_b.predictions)
    assert result_a.metrics == result_b.metrics
    assert result_a.model.model_.__class__.__name__ == "HistGradientBoostingClassifier"
    assert result_a.predictions["uncertainty"].between(0, 1).all()
    output = write_results(result_a, tmp_path / "output")
    assert {path.name for path in output.iterdir()} == {
        "confusion.csv",
        "metrics.json",
        "per_class_metrics.csv",
        "per_well_metrics.csv",
        "predictions.csv",
        "qc_missingness.csv",
        "qc_ranges.csv",
        "qc_wells.csv",
    }


def test_plotting_and_cli_entrypoints(tmp_path: Path, capsys) -> None:
    frame = synthetic_frame(wells=2, rows=8)
    figure, axes = plot_log_tracks(frame[frame.WELL == "W0"], ["GR", "RHOB"])
    assert len(axes) == 2
    figure.clf()
    figure, _ = plot_confusion(np.eye(12))
    figure.clf()
    predictions = pd.DataFrame({"DEPTH_MD": [1, 2], "uncertainty": [0.1, 0.8]})
    figure, axis = plot_uncertainty(predictions)
    assert axis.get_xlim() == pytest.approx((0, 1))
    figure.clf()

    data = tmp_path / "las"
    data.mkdir()
    (data / "a.las").write_text(LAS_TEMPLATE.format(well="A"), encoding="ascii")
    assert main(["qc", str(data)]) == 0
    output = capsys.readouterr().out
    assert '"rows": 3' in output
