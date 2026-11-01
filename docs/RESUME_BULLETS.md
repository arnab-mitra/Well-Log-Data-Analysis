# Resume bullets and evidence

## Replace the current submitted bullets

The previous claims about processing “10+ well datasets and 50+ LAS files,” calculating shale volume/Archie saturation, annotating 12+ formation tops, and specific porosity/saturation extrema are not supported by the earlier repositories. Replace them with claims produced by this clean-room implementation.

## Recommended two-line version

**Petrophysical Well-Log QC & Lithology Prediction**  
*Python, Pandas, NumPy, scikit-learn, Matplotlib, lasio, Streamlit*

- Built a reproducible Python/lasio pipeline to ingest and quality-control **118 public Norwegian wells and 2.34M depth samples**, standardizing heterogeneous logs and generating composite QC, lithology, and uncertainty tracks.
- Trained a leakage-safe **12-class lithology model** with complete wells held out; evaluated **286,870 samples across 24 unseen wells**, achieving **69.3% accuracy, 58.1% balanced accuracy, 0.484 macro-F1, and a −0.816 FORCE penalty score**.

## More geoscience-focused alternative

- Audited **118 FORCE 2020 LAS wells / 2.34M depth samples** for curve coverage, nulls, sampling consistency, and depth integrity, then built depth-aligned well-log and lithology visualizations for interpretation review.
- Developed a well-disjoint lithology-classification workflow with per-class and interpreter-confidence analysis, highlighting weak performance on rare dolomite/basement classes and uncertain intervals requiring geoscientist review.

## Evidence map

| Claim | Measured value | Evidence |
|---|---:|---|
| Source archive integrity | MD5 `da0d72d5783a780ac71779453d9fbfbf` | `data/registry.json`, verified local acquisition |
| LAS wells processed | 118 | full `welllog qc` output; `artifacts/force-full-seed42/run_summary.json` |
| Raw depth samples | 2,337,639 | full QC run and run summary |
| Labelled samples | 1,431,383 | `artifacts/force-full-seed42/qc_wells.csv` |
| Held-out wells | 24 | `artifacts/force-full-seed42/metrics.json` |
| Held-out samples | 286,870 | `metrics.json`, `predictions.csv` |
| Accuracy | 0.6928 | `metrics.json` |
| Balanced accuracy | 0.5807 | `metrics.json` |
| Macro-F1 | 0.4838 | `metrics.json` |
| FORCE score | −0.8157 | `metrics.json`; scoring tests in `tests/test_core.py` |
| Automated tests | 16 passing before final QA | local pytest output; CI validates after push |

## Interview explanation

A concise, honest explanation:

> “I replaced an untraceable notebook-only project with a clean-room pipeline built on the licensed FORCE 2020 release. I verified the Zenodo archive, parsed all 118 LAS files, added QC and well-local contextual features, and evaluated a gradient-boosting baseline using complete held-out wells. I deliberately excluded formation and interpreter-confidence fields from training to avoid metadata leakage. The model reached 0.484 macro-F1; the class-level output also shows where it fails, especially rare lithologies, so I present it as a review aid rather than an automated geological decision system.”

## Do not claim

- Archie water saturation, shale volume, net pay, formation tops, or fluid typing: this version does not implement those calculations.
- Cross-validation, probability calibration, deployment impact, or runtime improvement: these were not measured.
- Ownership of the removed historical notebooks: the new package is a clean-room rewrite; Git history remains for transparency.
- Accuracy alone: always pair it with macro-F1/balanced accuracy and the held-out-well protocol.
