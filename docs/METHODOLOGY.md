# Methodology

## Purpose and evidence standard

This document defines the intended protocol for evaluating lithology predictions from FORCE 2020 well logs without overstating the evidence. It separates controls already represented in reusable modules from controls that still need to be demonstrated by an end-to-end, archived run.

A result is reportable only when another person can connect it to:

- the exact source archive checksum;
- the included and excluded wells;
- a persisted train/validation/test split manifest;
- the selected logs and derived features;
- preprocessing and model configuration;
- software and dependency versions;
- random seeds;
- predictions, class probabilities, QC outputs, and metrics; and
- the code commit that produced those artifacts.

At present, project-level benchmark metrics are **pending a real run**.

## 1. Data source and integrity

The target data release is the FORCE 2020 Zenodo record, DOI [10.5281/zenodo.4351156](https://doi.org/10.5281/zenodo.4351156). It reports 118 Norwegian Sea wells. The primary LAS archive has:

- filename `LAS_files_Force_2020_all_wells_train_test_blind_hidden_final.zip`;
- size `169,785,172` bytes; and
- MD5 `da0d72d5783a780ac71779453d9fbfbf`.

The acquisition step should fail closed when the checksum differs. Retain the downloaded archive as immutable raw input, record the retrieval date locally, and preserve the DOI and license in derived-data metadata. MD5 is used because it is the checksum published by Zenodo, not because it provides modern adversarial security.

## 2. Ingestion and canonical schema

LAS files are parsed into a table with canonical identifiers:

- `WELL` for the grouping unit;
- `DEPTH_MD` for measured depth;
- `FORCE_2020_LITHOFACIES_LITHOLOGY` for the target; and
- normalized uppercase curve names.

Sentinel null values such as `-999`, `-999.25`, and `-9999`, plus infinities, are mapped to missing values. Rows without depth are removed and data are deterministically ordered by well and depth. Parsing failures must be counted and reported; silently skipping a well is not acceptable in a benchmark run.

## 3. Quality control

QC is descriptive and should precede feature/model decisions. At minimum, persist:

- file and readable-well counts;
- rows per well;
- depth minimum, maximum, and span;
- median sample spacing;
- duplicate and non-monotonic depth counts;
- curve-level missing counts/fractions;
- per-well missing-log fractions; and
- robust numeric ranges, including 1st, 50th, and 99th percentiles.

Domain range checks should be added only with documented units and defensible bounds. Values should be flagged before they are clipped or removed. Exclusion decisions need a reason code and should be derived without consulting held-out model performance.

## 4. Partitioning and leakage prevention

### Required split unit

Use wells—not depth rows—as groups. The current baseline utility uses a deterministic `GroupShuffleSplit` with a configurable held-out fraction and seed. Train and held-out well sets must be disjoint, and the run should assert that their intersection is empty.

For model selection, prefer one of these declared designs:

1. an outer held-out-well test set plus grouped cross-validation among training wells; or
2. grouped cross-validation for developmental estimates, followed by one untouched held-out-well evaluation.

Choose the design before examining final test results.

### Feature boundaries

Differences, gradients, shifts, interpolation, smoothing, and rolling windows must reset at well boundaries. Current rolling features are grouped by `WELL`, which prevents the end of one well from influencing the beginning of another.

Centered rolling windows use samples above and below a depth within the same well. They are suitable for offline interval interpretation, but they are not causal. A deployment that predicts while drilling would require trailing-only features and separate validation.

### Train-only learned operations

Fit all data-dependent operations on training wells only, including:

- imputation statistics;
- scalers or transforms;
- feature selection;
- categorical encoders;
- resampling;
- probability calibration;
- decision thresholds; and
- model hyperparameters.

Do not normalize a complete table before splitting. Do not use held-out labels to decide ranges, logs, filters, classes, thresholds, or windows.

### Target and metadata controls

Exclude the target, target-derived columns, post-interpretation products, split labels, filenames that encode the answer, and any unavailable-at-inference attributes. Confidence annotations require special review: if they were produced as part of the reference interpretation, using them as predictors may leak annotation quality or workflow information. Report results both with and without such fields if their deployment availability is uncertain.

## 5. Features and model

The baseline feature family comprises available numeric logs, missingness indicators, depth, well-local gradients, and well-local rolling mean/standard-deviation summaries. The exact list must be persisted for every run because wells have heterogeneous curve coverage.

The baseline classifier is `HistGradientBoostingClassifier` with a fixed seed, class balancing, and explicit hyperparameters. Missing numeric values can be handled natively by this estimator. This is a practical baseline, not a claim of geological optimality.

Any comparison model must receive the same data partitions and an equivalent tuning budget. Improvements should be measured on untouched wells, not training rows.

## 6. Evaluation

### Classification

Report at least:

- macro F1;
- balanced accuracy;
- per-class precision, recall, F1, and support;
- confusion matrix; and
- the verified FORCE competition score when its class order and cost matrix have been tested against a trusted reference.

Overall accuracy may be included but is insufficient because lithology support can be imbalanced.

### Well-level robustness

In addition to pooled depth-sample metrics, compute the chosen metrics per held-out well and report their distribution (median, interquartile range, and range). A pooled metric can be dominated by long wells. Confidence intervals or bootstrap summaries should resample at the well level, not individual depth rows.

### Probability and uncertainty

Persist class probabilities when supported. Candidate uncertainty summaries include:

- maximum-class confidence;
- `1 - max(probability)`;
- normalized predictive entropy; and
- top-two probability margin.

Evaluate probability quality with log loss or another declared proper scoring rule plus calibration plots/metrics. Define calibration binning before the final run. If calibration is learned, fit it using only grouped training/validation wells.

An uncertainty review threshold is an operational choice, not a universal truth. Report coverage versus error or risk, and state how the threshold was selected. Do not claim that predictive entropy captures structural, label, or geological uncertainty in full.

## 7. Artifact contract

A reproducible run should write a run-scoped directory such as:

```text
artifacts/run-001/
├── manifest.json             # commit, environment, seed, input checksum, configuration
├── split.json                # train/validation/test well identifiers
├── features.json             # selected inputs and derived feature definitions
├── predictions.parquet       # well, depth, truth, prediction, probabilities, uncertainty
├── metrics.json              # pooled, per-class, and per-well results
├── missingness_qc.csv
├── well_qc.csv
└── confusion_matrix.csv
```

The current dashboard recursively discovers predictions, metrics, and QC/quality-named artifacts. Training should happen before dashboard launch; the app reads persisted outputs and does not retrain on UI reruns.

## 8. Reproducibility checklist

Before publishing metrics or using a resume claim, confirm:

- [ ] Downloaded archive size and MD5 match the Zenodo record.
- [ ] All parsing failures and exclusions are recorded.
- [ ] Well sets are disjoint and saved.
- [ ] Every learned transformation is fit on training wells only.
- [ ] Target and target-derived fields are absent from predictors.
- [ ] Context features do not cross well boundaries.
- [ ] Model configuration, seed, dependencies, and commit are saved.
- [ ] Predictions and probabilities for held-out wells are persisted.
- [ ] Metrics include per-class and per-well evidence.
- [ ] Probability calibration and uncertainty claims have a declared protocol.
- [ ] Dataset and software licenses/citations accompany the artifacts.
- [ ] README and resume text use measured values, not estimates or aspirations.

## 9. Threats to validity

- Adjacent depth samples are autocorrelated; sample-level uncertainty intervals are misleading.
- A single grouped split can be sensitive to which wells are held out.
- Rare lithologies may be absent from one partition or receive unstable estimates.
- Missing curves can correlate with well identity, acquisition program, or label distribution.
- Centered context assumes access to future depth samples and is unsuitable for causal deployment.
- Label interpretation and confidence can vary by geoscientist and interval.
- Norwegian Sea performance does not establish transfer to another basin.
- Model confidence is not a substitute for a calibrated geological uncertainty model.

These limitations should remain adjacent to reported metrics so readers can judge the scope of the evidence.
