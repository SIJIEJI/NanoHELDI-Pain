# NanoHELDI-MS machine-learning analysis

This repository provides the installable Python source code, command-line tools,
tests, and a small synthetic dataset for the NanoHELDI-MS pain-study machine-learning
workflow. No compiled binary is required.

The manuscript workflow uses a participant-level 80:20 train/test partition followed
by grouped cross-validation within the training partition. Imputation, scaling, and
feature ranking are fitted from training data only. Split assignments, fold-level
scores, test predictions, ROC coordinates, confusion matrices, fitted pipelines,
software versions, and input hashes are saved for inspection.

> **Scope.** The included data are simulated and have no biological meaning. Raw or
> participant-level study data are not included.

## Repository contents

| Path | Contents |
|---|---|
| [`src/nanoheldi_ml`](src/nanoheldi_ml) | Installable evaluation and plotting source code |
| [`examples/data`](examples/data) | Included classification and regression demo CSV files |
| [`examples/make_synthetic_data.py`](examples/make_synthetic_data.py) | Script that generated the demo files |
| [`scripts/run_classification_panels.py`](scripts/run_classification_panels.py) | Batch runner for three classification outcomes |
| [`scripts/run_feature_count_analysis.py`](scripts/run_feature_count_analysis.py) | Fold-local ANOVA/MI feature-count analysis |
| [`scripts/run_interpretability.py`](scripts/run_interpretability.py) | SHAP and integrated-gradients analysis for saved MLPs |
| [`tests`](tests) | Automated tests for grouping, artifacts, and metrics |
| [`docs/METHODS.md`](docs/METHODS.md) | Analysis design and statistical safeguards |
| [`docs/VALIDATION_REPORT.md`](docs/VALIDATION_REPORT.md) | Local validation record |

## 1. System requirements

### Operating system and hardware

- A 64-bit macOS, Linux, or Windows system capable of running Python 3.10 or later is
  required. The code has no OS-specific system calls.
- Validation for this release was performed on **macOS 26.3.1 (build 25D2128),
  Apple Silicon arm64**, using Python 3.11.5. GitHub Actions is configured for
  `ubuntu-latest` with Python 3.11; Windows has not yet been independently tested.
- No non-standard hardware is required. All commands run on CPU; a GPU is neither
  required nor used.
- For the demo, 2 GB available RAM and 1 GB free disk space are sufficient. Memory,
  runtime, and saved-model storage for study data scale with the number of rows,
  features, folds, and models.

### Software dependencies

The authoritative dependency constraints are in [`pyproject.toml`](pyproject.toml)
and [`requirements.txt`](requirements.txt). `pip` installs their transitive
dependencies, including SciPy and Matplotlib support packages.

| Software | Supported constraint | Versions tested locally |
|---|---:|---:|
| Python | `>=3.10` | 3.11.5 |
| NumPy | `>=1.24,<3` | 1.26.4; 2.4.6 |
| pandas | `>=2.0,<3` | 2.1.4; 2.3.3 |
| scikit-learn | `>=1.2,<2` | 1.2.2; 1.9.0 |
| Matplotlib | `>=3.7,<4` | 3.7.2; 3.11.1 |
| joblib | `>=1.2,<2` | 1.2.0; 1.6.0 |
| XGBoost (optional) | `>=2.0,<4` | 3.2.0 |
| SHAP (optional) | `>=0.44,<1` | 0.51.0 |
| pytest (development only) | `>=7,<9` | available through the `dev` extra |
| Ruff (development only) | `>=0.6,<1` | available through the `dev` extra |

## 2. Installation guide

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/SIJIEJI/NanoHELDI-Pain.git
cd NanoHELDI-Pain
python -m venv .venv
source .venv/bin/activate                 # macOS or Linux
# .venv\Scripts\Activate.ps1             # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install .
```

For development and tests, install the `dev` extra. Add `xgboost` only when the XGBoost
model is needed:

```bash
python -m pip install ".[dev]"
python -m pip install ".[xgboost]"        # optional
python -m pip install ".[explain]"        # optional SHAP/IG workflow
```

Alternatively, create the tested Python 3.11 Conda environment:

```bash
conda env create -f environment.yml
conda activate nanoheldi-ml
python -m pip install .
```

Confirm the installation:

```bash
nanoheldi-evaluate --help
nanoheldi-plot --help
nanoheldi-manuscript-plot --help
nanoheldi-explain --help
```

On the tested desktop, a clean `pip install .` including downloaded dependencies took
**14.4 seconds** on a broadband connection. Allow approximately **2–5 minutes** on a
typical desktop when packages are not cached or the connection is slower. XGBoost is
larger and may add installation time.

## 3. Demo

### Included data

Two ready-to-run simulated tables are supplied:

- [`examples/data/synthetic_classification.csv`](examples/data/synthetic_classification.csv)
- [`examples/data/synthetic_regression.csv`](examples/data/synthetic_regression.csv)

Each contains 48 observations from 24 synthetic participants, two observations per
participant, and 20 synthetic `mz_` features. The classification table contains a
balanced binary `target`; the regression table contains a continuous `pain_score`.

### Run the classification demo

From the repository root, run:

```bash
nanoheldi-evaluate classification \
  --input examples/data/synthetic_classification.csv \
  --target target \
  --features 10 \
  --folds 3 \
  --models LR MLP \
  --output results/classification_demo

nanoheldi-plot \
  --summary results/classification_demo/summary_metrics.csv \
  --folds results/classification_demo/fold_metrics.csv \
  --output results/classification_demo/model_comparison
```

The expected terminal messages are:

```text
Saved validated outputs to .../results/classification_demo
Saved plot files with stem .../results/classification_demo/model_comparison
```

The reference run produced mean balanced accuracies of 0.875 for LR and 0.8125 for
MLP. These values test the installation only and must not be interpreted as scientific
results.

### Run the regression demo

```bash
nanoheldi-evaluate regression \
  --input examples/data/synthetic_regression.csv \
  --target pain_score \
  --features 10 \
  --folds 3 \
  --models MLP \
  --output results/regression_demo
```

The reference run produced an MLP mean RMSE of approximately 1.22 and an R² of
approximately 0.76. Small last-decimal differences between supported dependency
versions are acceptable.

### Expected files

Each evaluation directory contains:

```text
fold_metrics.csv                 one row per model and validation fold
summary_metrics.csv              mean, SD, participant count, and observation count
predictions.csv                  out-of-fold predictions only
split_assignments.csv            training/validation membership for every fold
selected_features.csv            features fitted independently in every training fold
run_metadata.json                configuration, versions, input hash, and warnings
models/<model>/fold_*.joblib     fitted fold-specific pipelines
models/<model>/final_full_data.joblib
                                 deployment candidate; not used for CV metrics
```

The plotting command additionally creates `.png`, `.pdf`, `.svg`, and `.tiff` files.
The classification demo saves eight model artifacts: three folds plus one full-data
candidate for each of LR and MLP.

### Expected demo runtime

Times below were measured on the tested Apple Silicon desktop with Python 3.11.5.

| Demo step | Observed wall time | Typical desktop allowance |
|---|---:|---:|
| Classification: LR + MLP, 3 folds | 1.0–1.2 s | under 1 min |
| Four-format figure export | 0.7 s after font cache; 10 s on a clean cache | under 1 min |
| Regression: MLP, 3 folds | 1.0–1.2 s | under 1 min |

To regenerate the included synthetic files in another directory without replacing the
repository copies:

```bash
python examples/make_synthetic_data.py --output /tmp/nanoheldi-demo-data
```

The generator refuses to overwrite existing CSV files unless `--overwrite` is supplied.

## 4. Instructions for use on your own data

### Input table

Supply the deidentified observation-level matrix used for analysis. Each row may be a
measurement, visit, or prespecified replicate-level observation, but its participant
must be identified so that related rows cannot cross evaluation partitions. The
default schema is:

```text
sample_id,participant_id,target,mz_124.97,mz_129.06,...
```

- `sample_id` must identify a row uniquely.
- `participant_id` is mandatory. All observations from one participant remain in the
  same fold, including repeated visits.
- The target may contain class labels for classification or numeric values for
  regression.
- Feature columns must be numeric and begin with `mz_` by default. Use
  `--feature-prefix` to choose another prefix.
- Choose a new or empty output directory for every run. The software refuses to
  replace a non-empty results directory.

See [`data/README.md`](data/README.md) for validation rules and privacy guidance.

### Manuscript classification protocol

This command uses all supplied `mz_` features, creates a participant-disjoint 80:20
training/test partition, and compares the requested models by 10-fold grouped
cross-validation in the training partition. The independent test partition is scored
only after model fitting. Install the `xgboost` extra to include the complete candidate
set used by this command.

```bash
nanoheldi-evaluate manuscript-classification \
  --input data/deidentified_analysis.csv \
  --target headache \
  --test-fraction 0.2 \
  --training-cv-folds 10 \
  --include-xgboost \
  --output results/headache_manuscript

nanoheldi-manuscript-plot classification \
  --run results/headache_manuscript \
  --output results/headache_manuscript/figures
```

The model-comparison figure displays every cross-validation fold as an individual
point over the mean and SD. The same plotting command also generates test ROC curves
and the confusion matrix for the model selected by training-CV macro one-vs-rest AUC.

### Manuscript regression protocol

```bash
nanoheldi-evaluate manuscript-regression \
  --input data/deidentified_analysis.csv \
  --target headache_severity \
  --test-fraction 0.2 \
  --training-cv-folds 10 \
  --models MLP \
  --tolerance 2 \
  --output results/headache_severity_manuscript

nanoheldi-manuscript-plot regression \
  --run results/headache_severity_manuscript \
  --model MLP \
  --output results/headache_severity_manuscript/predicted_vs_observed
```

Regression reports MAE, RMSE, R², and the fraction of independent test predictions
within the declared ±2-unit tolerance.

### Feature-count curves

The following command reproduces the 10/20/30/40/50-feature analysis pattern. ANOVA F
and mutual information are recomputed inside each training fold before the top-ranked
features are selected, so validation observations do not influence the ranking.

```bash
python scripts/run_feature_count_analysis.py classification \
  --input data/deidentified_analysis.csv \
  --target headache \
  --counts 10 20 30 40 50 \
  --folds 5 \
  --output results/headache_feature_curve

nanoheldi-manuscript-plot feature-curve \
  --summary results/headache_feature_curve/feature_curve_summary.csv \
  --metric accuracy \
  --output results/headache_feature_curve/accuracy_by_feature_count
```

### MLP interpretation

After a manuscript evaluation has saved its training-only MLP pipeline, SHAP and
integrated gradients can be calculated for at most 100 designated test observations.
This step reads the frozen model and split IDs and does not refit the estimator or
alter its test predictions.

```bash
nanoheldi-explain \
  --input data/deidentified_analysis.csv \
  --target headache \
  --model results/headache_manuscript/models/MLP.joblib \
  --max-test-samples 100 \
  --output results/headache_manuscript/mlp_interpretability
```

### General grouped-CV interface

The original interface remains available for analyses that require a user-specified
feature count and grouped cross-validation without a separate test partition:

```bash
nanoheldi-evaluate classification \
  --input data/deidentified_analysis.csv \
  --target headache \
  --features 50 \
  --folds 5 \
  --output results/headache_5fold
```

The default model set is Random, NB, DT, LR, GB, SVM, RF, and MLP. Select a subset with
`--models LR SVM MLP`, or add XGBoost after installing its optional dependency:

```bash
nanoheldi-evaluate classification \
  --input data/deidentified_analysis.csv \
  --target headache \
  --features 50 \
  --folds 5 \
  --include-xgboost \
  --output results/headache_5fold_xgboost
```

### General grouped-CV regression

```bash
nanoheldi-evaluate regression \
  --input data/deidentified_analysis.csv \
  --target headache_severity \
  --features 50 \
  --folds 5 \
  --models MLP \
  --tolerance 2 \
  --output results/headache_severity_5fold
```

Regression reports MAE, RMSE, conventional R², and the fraction of predictions within
the declared tolerance. The tolerance-based value is secondary and should not replace
continuous-error metrics.

### Batch general grouped-CV classification

```bash
python scripts/run_classification_panels.py \
  --input data/deidentified_analysis.csv \
  --targets menstrual_phase headache cramps \
  --features 50 \
  --folds 5 \
  --include-xgboost \
  --output results/three_classification_panels
```

With nine model families, three targets, and five folds, this saves 135 fitted fold
pipelines. Full-data candidates are stored separately and are never used to calculate
the plotted validation scores.

### Validate and plot a general grouped-CV run

```bash
python tools/verify_run.py results/headache_5fold

nanoheldi-plot \
  --summary results/headache_5fold/summary_metrics.csv \
  --folds results/headache_5fold/fold_metrics.csv \
  --output results/headache_5fold/model_comparison
```

The plot overlays every fold score on the model mean. Fold scores are evaluation
partitions, not independent biological replicates. Report `n` as the number of unique
participants and describe error bars as variation across folds.

## 5. Reproducing the analysis

1. Install the package and optional XGBoost dependency as described above.
2. Place the approved, deidentified analysis table under `data/`. Do not commit direct
   identifiers, collection dates, or restricted study data.
3. Run the `manuscript-classification` and `manuscript-regression` commands with the
   prespecified targets and model families from the analysis plan.
4. Run `python tools/verify_run.py <result-directory>` for every model-evaluation run.
5. Run the feature-count command when producing feature-number curves.
6. Generate figures only from the CSV artifacts written by the completed run.
7. Record the git commit, input SHA-256 from `run_metadata.json`, and output directory
   in the manuscript analysis log.

Repository-level checks are:

```bash
python tools/release_check.py
pytest -q
python tools/write_checksums.py --check
```

## Reproducibility policy

- Holdout and cross-validation partitions are participant-grouped.
- A single declared estimator random state is used only where a stochastic algorithm
  requires reproducible fitting; it is never searched or selected using performance.
- Preprocessing is part of each fitted pipeline; feature-count ranking is fitted
  independently inside each training fold.
- All models within a task receive the same outer splits.
- Saved full-data models are deployment candidates, not evidence of validation
  performance.

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The source code is
released under the [`MIT License`](LICENSE). Add the manuscript DOI to the citation file
when it becomes available.
