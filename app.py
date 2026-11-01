"""Artifact-driven Streamlit portfolio for the FORCE lithology workflow.

The app is deliberately inference/training free: it reads persisted outputs produced by
``welllog.pipeline.build_project`` so Streamlit reruns remain fast and predictable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # Keep pure helper functions importable in lightweight test environments.
    import streamlit as st
except ImportError:  # pragma: no cover - exercised only outside the app environment
    st = None  # type: ignore[assignment]


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIRS = (APP_ROOT / "artifacts", APP_ROOT / "outputs", APP_ROOT / "results")
TABLE_SUFFIXES = (".parquet", ".csv")

WELL_ALIASES = ("well", "well_name", "well_id", "uwi", "borehole")
DEPTH_ALIASES = ("depth", "depth_md", "dept", "md", "measured_depth")
PREDICTION_ALIASES = (
    "predicted_lithology",
    "prediction",
    "predicted_class",
    "lithology_prediction",
    "y_pred",
)
REFERENCE_ALIASES = (
    "reference_lithology",
    "lithology",
    "true_lithology",
    "target",
    "y_true",
    "force_2020_lithofacies_lithology",
)
UNCERTAINTY_ALIASES = ("uncertainty", "predictive_uncertainty", "entropy", "prediction_entropy")
CONFIDENCE_ALIASES = ("confidence", "prediction_confidence", "max_probability", "probability")
CURVE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Gamma ray", ("gr", "gamma_ray", "gamma")),
    ("Density", ("rhob", "density", "bulk_density")),
    ("Neutron", ("nphi", "neutron_porosity", "neutron")),
    ("Sonic", ("dt", "dtc", "sonic")),
    ("Resistivity", ("rdep", "deep_resistivity", "resd", "ild")),
)


@dataclass(frozen=True)
class ArtifactBundle:
    """Paths to dashboard-ready outputs."""

    root: Path
    predictions: Path | None = None
    metrics: Path | None = None
    qc: tuple[Path, ...] = ()

    @property
    def ready(self) -> bool:
        return self.predictions is not None


def _normalise_name(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def find_column(columns: Iterable[object], aliases: Iterable[str]) -> str | None:
    """Find a column case-insensitively while preserving its original spelling."""

    lookup = {_normalise_name(column): str(column) for column in columns}
    return next((lookup[_normalise_name(alias)] for alias in aliases if _normalise_name(alias) in lookup), None)


def _ranked_file(root: Path, stems: tuple[str, ...], suffixes: tuple[str, ...]) -> Path | None:
    candidates = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes]
    stem_rank = {stem: rank for rank, stem in enumerate(stems)}

    def score(path: Path) -> tuple[int, int, str]:
        normalised = _normalise_name(path.stem)
        exact = stem_rank.get(normalised, len(stems))
        contains = 0 if any(stem in normalised for stem in stems) else 1
        return exact, contains, str(path)

    matching = [path for path in candidates if any(stem in _normalise_name(path.stem) for stem in stems)]
    return min(matching, key=score) if matching else None


def discover_artifacts(root: str | Path) -> ArtifactBundle:
    """Discover standard pipeline outputs below *root* without loading them."""

    directory = Path(root).expanduser().resolve()
    if not directory.is_dir():
        return ArtifactBundle(root=directory)

    predictions = _ranked_file(
        directory,
        ("predictions", "lithology_predictions", "prediction"),
        TABLE_SUFFIXES,
    )
    metrics = _ranked_file(directory, ("metrics", "model_metrics", "evaluation_metrics"), (".json",))

    qc_paths = tuple(
        sorted(
            (
                path
                for path in directory.rglob("*")
                if path.is_file()
                and path.suffix.lower() in (*TABLE_SUFFIXES, ".json")
                and ("qc" in _normalise_name(path.stem) or "quality" in _normalise_name(path.stem))
            ),
            key=lambda path: str(path),
        )
    )
    return ArtifactBundle(root=directory, predictions=predictions, metrics=metrics, qc=qc_paths)


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a supported tabular artifact."""

    artifact = Path(path)
    if artifact.suffix.lower() == ".parquet":
        return pd.read_parquet(artifact)
    if artifact.suffix.lower() == ".csv":
        return pd.read_csv(artifact)
    raise ValueError(f"Unsupported table format: {artifact.suffix}")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Return a clean copy and the semantic columns used by the dashboard."""

    prepared = frame.copy()
    columns: dict[str, str | None] = {
        "well": find_column(prepared.columns, WELL_ALIASES),
        "depth": find_column(prepared.columns, DEPTH_ALIASES),
        "prediction": find_column(prepared.columns, PREDICTION_ALIASES),
        "reference": find_column(prepared.columns, REFERENCE_ALIASES),
        "uncertainty": find_column(prepared.columns, UNCERTAINTY_ALIASES),
        "confidence": find_column(prepared.columns, CONFIDENCE_ALIASES),
    }
    if columns["well"] is None:
        prepared.insert(0, "well", "Project")
        columns["well"] = "well"
    if columns["depth"] is not None:
        prepared[columns["depth"]] = pd.to_numeric(prepared[columns["depth"]], errors="coerce")
        prepared = prepared.sort_values([columns["well"], columns["depth"]], kind="stable")
    return prepared, columns


def uncertainty_series(frame: pd.DataFrame, columns: Mapping[str, str | None]) -> pd.Series | None:
    """Return uncertainty, deriving it from confidence when needed."""

    if columns.get("uncertainty"):
        return pd.to_numeric(frame[columns["uncertainty"]], errors="coerce")
    if columns.get("confidence"):
        confidence = pd.to_numeric(frame[columns["confidence"]], errors="coerce")
        if confidence.notna().any() and confidence.max() > 1.0:
            confidence = confidence / 100.0
        return (1.0 - confidence).clip(0.0, 1.0)
    return None


def flatten_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    """Flatten numeric leaves from nested metrics JSON for compact display."""

    flattened: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_metrics(child, name))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
        flattened[prefix] = float(value)
    return flattened


def _format_metric(value: float) -> str:
    if 0.0 <= value <= 1.0:
        return f"{value:.1%}"
    return f"{value:,.3f}" if not float(value).is_integer() else f"{value:,.0f}"


def _humanise(name: str) -> str:
    return name.split(".")[-1].replace("_", " ").strip().title()


def _curve_columns(frame: pd.DataFrame) -> list[tuple[str, str]]:
    curves: list[tuple[str, str]] = []
    for label, aliases in CURVE_GROUPS:
        column = find_column(frame.columns, aliases)
        if column:
            curves.append((label, column))
    return curves[:3]


def _fallback_composite(frame: pd.DataFrame, columns: Mapping[str, str | None]):
    """Create a deterministic composite when the package plotter cannot be used."""

    import matplotlib.pyplot as plt

    depth_column = columns.get("depth")
    depth = (
        pd.to_numeric(frame[depth_column], errors="coerce")
        if depth_column
        else pd.Series(np.arange(len(frame)), index=frame.index, name="Sample")
    )
    curves = _curve_columns(frame)
    categorical = [
        ("Reference", columns.get("reference"), "#6A7FDB"),
        ("Prediction", columns.get("prediction"), "#00B8A9"),
    ]
    categorical = [(label, column, colour) for label, column, colour in categorical if column]
    uncertainty = uncertainty_series(frame, columns)
    width_count = max(1, len(curves)) + (1 if categorical else 0) + (1 if uncertainty is not None else 0)
    fig, axes = plt.subplots(1, width_count, figsize=(2.55 * width_count, 8), sharey=True, squeeze=False)
    axes_list = list(axes[0])
    index = 0

    for label, column in curves:
        values = pd.to_numeric(frame[column], errors="coerce")
        axes_list[index].plot(values, depth, color="#203A5F", linewidth=0.85)
        axes_list[index].set_title(label, fontsize=10, fontweight="bold")
        axes_list[index].set_xlabel(column, fontsize=8)
        axes_list[index].grid(alpha=0.18)
        index += 1

    if categorical:
        axis = axes_list[index]
        labels = pd.unique(pd.concat([frame[column].astype(str) for _, column, _ in categorical]).dropna())
        mapping = {label: number for number, label in enumerate(labels)}
        for label, column, colour in categorical:
            codes = frame[column].astype(str).map(mapping)
            axis.step(codes, depth, where="mid", label=label, color=colour, linewidth=1.25, alpha=0.9)
        axis.set_title("Lithology", fontsize=10, fontweight="bold")
        axis.set_xticks(list(mapping.values()), list(mapping.keys()), rotation=90, fontsize=7)
        axis.legend(loc="lower center", fontsize=7)
        axis.grid(alpha=0.18)
        index += 1

    if uncertainty is not None:
        axis = axes_list[index]
        axis.plot(uncertainty, depth, color="#F59E0B", linewidth=1.0)
        axis.fill_betweenx(depth, 0, uncertainty, color="#FCD34D", alpha=0.35)
        axis.set_xlim(0, max(1.0, float(uncertainty.max(skipna=True) or 1.0)))
        axis.set_title("Uncertainty", fontsize=10, fontweight="bold")
        axis.set_xlabel("Higher = review", fontsize=8)
        axis.grid(alpha=0.18)

    axes_list[0].set_ylabel(depth_column or "Sample")
    axes_list[0].invert_yaxis()
    fig.tight_layout()
    return fig


def _package_composite(frame: pd.DataFrame):
    """Use the project's plotting contract when its accepted signature is available."""

    from welllog.plotting import plot_composite_log

    return plot_composite_log(frame)


def _qc_frame(paths: tuple[Path, ...]) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for path in paths:
        try:
            if path.suffix.lower() == ".json":
                value = load_json(path)
                if isinstance(value, (list, Mapping)):
                    part = pd.json_normalize(value)
                else:
                    part = pd.DataFrame({"value": [value]})
            else:
                part = load_table(path)
            part.insert(0, "source", path.name)
            records.append(part)
        except (OSError, ValueError, TypeError, ImportError):
            continue
    return pd.concat(records, ignore_index=True, sort=False) if records else pd.DataFrame()


def _initial_artifact_root() -> Path:
    return next((path for path in DEFAULT_ARTIFACT_DIRS if path.is_dir()), DEFAULT_ARTIFACT_DIRS[0])


def _render_empty_state(bundle: ArtifactBundle) -> None:
    st.markdown(
        """
        <div class="empty-card">
          <div class="eyebrow">READY FOR PIPELINE OUTPUTS</div>
          <h2>Turn model artifacts into a subsurface decision view</h2>
          <p>No predictions table was found. This dashboard never retrains a model on rerun;
          it only reads versioned outputs generated by the package workflow.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Expected handoff")
    left, middle, right = st.columns(3)
    left.markdown("**1 · Build**\n\nRun `build_project(...)` in the workflow notebook or CLI.")
    middle.markdown("**2 · Persist**\n\nSave `predictions.parquet` (or CSV), `metrics.json`, and QC summaries.")
    right.markdown("**3 · Point & review**\n\nSet the artifact folder in the sidebar; no app restart is needed.")
    st.code(str(bundle.root), language=None)
    st.caption("Discovery is recursive, so run-specific subfolders are supported.")


def _inject_style() -> None:
    st.markdown(
        """
        <style>
        :root { --ink:#132238; --muted:#607086; --teal:#00a896; --sand:#f2b95f; }
        .stApp { background: linear-gradient(180deg,#f7fafb 0%,#ffffff 35%); }
        [data-testid="stSidebar"] { background:#10243b; }
        [data-testid="stSidebar"] * { color:#edf6f6; }
        .hero { padding:1.1rem 0 1.6rem 0; border-bottom:1px solid #dfe8eb; margin-bottom:1.2rem; }
        .eyebrow { color:#008b80; font-size:.72rem; font-weight:800; letter-spacing:.13em; }
        .hero h1,.empty-card h2 { color:var(--ink); letter-spacing:-.035em; margin:.25rem 0; }
        .hero p,.empty-card p { color:var(--muted); max-width:760px; font-size:1.03rem; }
        .empty-card { padding:2rem; background:#fff; border:1px solid #dfe8eb; border-left:5px solid var(--teal);
                      border-radius:14px; box-shadow:0 10px 30px rgba(19,34,56,.06); }
        [data-testid="stMetric"] { background:#fff; border:1px solid #e2eaed; padding:.8rem 1rem; border-radius:10px; }
        div[data-testid="stExpander"] { background:#fff; border-color:#dfe8eb; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    if st is None:  # pragma: no cover
        raise RuntimeError("Streamlit is required to run the dashboard: pip install streamlit")

    st.set_page_config(page_title="WellLog · Lithology Review", page_icon="◫", layout="wide")
    _inject_style()
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">FORCE LITHOLOGY · MODEL REVIEW</div>
          <h1>From well curves to review-ready lithology</h1>
          <p>Inspect the evidence, compare reference and predicted facies, and focus expert
          attention where uncertainty or data quality warrants it.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("## Project lens")
    artifact_text = st.sidebar.text_input("Artifact folder", value=str(_initial_artifact_root()))
    if st.sidebar.button("Refresh artifacts", width="stretch"):
        st.cache_data.clear()

    @st.cache_data(show_spinner=False)
    def cached_discover(path: str) -> ArtifactBundle:
        return discover_artifacts(path)

    @st.cache_data(show_spinner="Loading saved predictions…")
    def cached_table(path: str, modified: float) -> pd.DataFrame:
        del modified
        return load_table(path)

    @st.cache_data(show_spinner=False)
    def cached_json(path: str, modified: float) -> Any:
        del modified
        return load_json(path)

    bundle = cached_discover(artifact_text)
    if not bundle.ready:
        _render_empty_state(bundle)
        return

    try:
        raw = cached_table(str(bundle.predictions), bundle.predictions.stat().st_mtime)
        predictions, columns = prepare_predictions(raw)
    except (OSError, ValueError, ImportError) as exc:
        st.error(f"The predictions artifact could not be loaded: {exc}")
        st.caption(str(bundle.predictions))
        return

    well_column = columns["well"]
    wells = sorted(predictions[well_column].dropna().astype(str).unique().tolist())
    if not wells:
        st.warning("The predictions artifact contains no well identifiers or rows to display.")
        return
    selected_well = st.sidebar.selectbox("Well", wells)
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Predictions · {bundle.predictions.name}")
    st.sidebar.caption(f"{len(predictions):,} depth samples · {len(wells):,} wells")
    st.sidebar.success("Artifact mode · no training on rerun")

    selected = predictions.loc[predictions[well_column].astype(str) == selected_well].copy()
    depth_column = columns.get("depth")
    interval = "Depth unavailable"
    if depth_column and selected[depth_column].notna().any():
        interval = f"{selected[depth_column].min():,.1f}–{selected[depth_column].max():,.1f}"

    metrics: dict[str, float] = {}
    if bundle.metrics:
        try:
            metrics = flatten_metrics(cached_json(str(bundle.metrics), bundle.metrics.stat().st_mtime))
        except (OSError, ValueError, TypeError):
            metrics = {}

    overall_accuracy = metrics.get("metrics.accuracy")
    summary_columns = st.columns(4)
    summary_columns[0].metric("Selected well", selected_well)
    summary_columns[1].metric("Logged interval", interval)
    summary_columns[2].metric("Samples", f"{len(selected):,}")
    if overall_accuracy is not None:
        summary_columns[3].metric("Held-out accuracy", _format_metric(overall_accuracy))
    else:
        summary_columns[3].metric("Model metrics", "Not supplied")

    overview_tab, evidence_tab, quality_tab = st.tabs(["Decision view", "Model evidence", "Data quality"])
    with overview_tab:
        st.markdown(f"### {selected_well} · composite interpretation")
        st.caption("Depth increases downward. Lithology disagreement and high uncertainty identify review intervals.")
        try:
            figure = _package_composite(selected)
            st.pyplot(figure, clear_figure=True, width="stretch")
        except (ImportError, TypeError, ValueError, KeyError, AttributeError):
            st.pyplot(_fallback_composite(selected, columns), clear_figure=True, width="stretch")

        uncertainty = uncertainty_series(selected, columns)
        if uncertainty is not None and uncertainty.notna().any():
            threshold = st.slider("Uncertainty review threshold", 0.0, 1.0, 0.35, 0.05)
            review = selected.loc[uncertainty >= threshold].copy()
            st.markdown(f"**{len(review):,} samples** meet the review threshold ({len(review) / len(selected):.1%} of this well).")
            if not review.empty:
                display = [column for column in (depth_column, columns.get("reference"), columns.get("prediction")) if column]
                review["uncertainty"] = uncertainty.loc[review.index]
                st.dataframe(review[[*display, "uncertainty"]], hide_index=True, width="stretch")
        else:
            st.info("No uncertainty or confidence field is present. Add one to predictions to enable review prioritisation.")

    with evidence_tab:
        if metrics:
            st.markdown("### Persisted evaluation metrics")
            metric_items = list(metrics.items())
            for offset in range(0, len(metric_items), 4):
                metric_columns = st.columns(4)
                for metric_column, (key, value) in zip(metric_columns, metric_items[offset : offset + 4]):
                    metric_column.metric(_humanise(key), _format_metric(value))
        else:
            st.info("No readable `metrics.json` was found. Predictions remain available for interpretation review.")
        st.markdown("### Prediction inventory")
        inventory = pd.DataFrame(
            {
                "Field": ["Prediction", "Reference", "Uncertainty", "Depth"],
                "Artifact column": [columns.get("prediction"), columns.get("reference"), columns.get("uncertainty") or columns.get("confidence"), depth_column],
            }
        )
        st.dataframe(inventory, hide_index=True, width="stretch")
        with st.expander("Preview selected-well records"):
            st.dataframe(selected.head(250), hide_index=True, width="stretch")

    with quality_tab:
        qc = _qc_frame(bundle.qc)
        if qc.empty:
            st.info("No readable QC summary was found. Add a JSON, CSV, or Parquet artifact with `qc` or `quality` in its name.")
        else:
            st.markdown("### Pipeline quality controls")
            st.caption(f"Combined from {len(bundle.qc)} persisted QC artifact(s).")
            st.dataframe(qc, hide_index=True, width="stretch")

    st.caption(f"Artifact source: {bundle.root} · Saved-output review only; model training is intentionally excluded from this app.")


if __name__ == "__main__":
    main()
