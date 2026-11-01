# FORCE 2020 data

This directory is reserved for source data, local extracts, and small machine-readable provenance records. The FORCE 2020 archive is intentionally **not versioned in Git**.

## Source record

| Field | Value |
|---|---|
| Title | FORCE 2020 Well well log and lithofacies dataset for machine learning competition |
| Creators | Peter Bormann; Peder Aursand; Fahad Dilib; Surrender Manral; Peter Dischington |
| Published | 2020-12-18 |
| Repository | Zenodo |
| DOI | [10.5281/zenodo.4351156](https://doi.org/10.5281/zenodo.4351156) |
| Wells | 118 |
| LAS archive | `LAS_files_Force_2020_all_wells_train_test_blind_hidden_final.zip` |
| Archive size | 169,785,172 bytes |
| Published checksum | MD5 `da0d72d5783a780ac71779453d9fbfbf` |
| Dataset license | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

The source description states that the wells are from the Norwegian Sea, that the logs were slightly cleaned and partially despiked, and that skilled geoscientists created the lithofacies/lithology interpretation. It also states that the original well-log data came from the Norwegian government under a NOLD 2.0 license. Consult the Zenodo record and linked source documentation before redistributing or reusing the data.

## Download

From an installed editable checkout:

```bash
python -c "from welllog.download import download_force_archive; download_force_archive('data/raw/force_2020_las.zip')"
```

The downloader writes through a temporary file, checks the published MD5, and only then places the archive at the destination. Existing valid files are reused; an existing invalid file is rejected unless overwrite is requested explicitly.

The CLI equivalent is:

```bash
welllog download data/raw/force_2020_las.zip
```

## Independent integrity check

Using the package helper:

```bash
python -c "from welllog.download import md5sum; print(md5sum('data/raw/force_2020_las.zip'))"
```

Or with a platform utility:

```bash
# macOS/Linux
md5sum data/raw/force_2020_las.zip

# Windows PowerShell
(Get-FileHash data/raw/force_2020_las.zip -Algorithm MD5).Hash.ToLower()
```

Expected digest:

```text
da0d72d5783a780ac71779453d9fbfbf
```

MD5 is documented because it is the checksum published by Zenodo for this file. It verifies equivalence to the deposited archive; it is not presented as a cryptographically secure authenticity mechanism.

## Suggested local layout

```text
data/
├── README.md
├── registry.json
├── raw/
│   ├── force_2020_las.zip       # ignored by Git
│   └── force_2020_las/          # extracted LAS files; ignored by Git
└── processed/                   # generated tables; ignored by Git
```

Keep data immutable under `raw/`. Write derived tables, split manifests, QC reports, and model outputs to generated directories rather than editing source files.

## Licensing boundary

The repository's MIT license applies only to original project code and documentation. It does **not** relicense:

- the FORCE 2020 archive or extracted files;
- lithology labels or source metadata contained in the dataset;
- Norwegian government data referenced by the depositor; or
- any other third-party material.

The Zenodo record designates the deposited dataset as Creative Commons Attribution 4.0 International. Preserve attribution, DOI, license notices, and provenance with any permitted redistribution. Users are responsible for determining how the source record's underlying NOLD 2.0 statement applies to their use.

Machine-readable provenance is maintained in [`registry.json`](registry.json).
