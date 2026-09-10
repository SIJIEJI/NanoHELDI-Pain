# Validation report

## Material Passport

- Material ID: `nanoheldi-release-validation-2026-09-09`
- Type: synthetic reproducibility validation
- Verification status: `VERIFIED`
- Source scope: generated non-biological data only
- Output: tests, model artifacts, metrics, and figure-format checks

Status: **VERIFIED ON SYNTHETIC DATA**

The release implementation was tested without participant data on macOS 26.3.1
(build 25D2128), Apple Silicon arm64, and Python 3.11.5. The primary environment used
NumPy 1.26.4, pandas 2.1.4, scikit-learn 1.2.2, Matplotlib 3.7.2, and joblib 1.2.0.
A clean environment was also tested with NumPy 2.4.6, pandas 2.3.3,
scikit-learn 1.9.0, Matplotlib 3.11.1, and joblib 1.6.0.

- Four automated tests passed.
- Participant overlap was zero in every classification and regression fold.
- All nine classification model families, including optional XGBoost, completed
  five-fold grouped evaluation.
- Forty-five classification fold-specific model bundles were saved and reloaded.
- Regression produced MAE, RMSE, conventional R², and ±2 accuracy.
- PNG, PDF, SVG, and 600-dpi LZW-compressed TIFF plots were generated.
- A non-empty output directory was confirmed to be protected from overwriting.
- A Python wheel was built successfully from `pyproject.toml`.
- A clean `pip install .` including downloaded dependencies completed in 14.4 seconds.
- The included LR + MLP classification demo completed in 1.0--1.2 seconds; the MLP
  regression demo completed in 1.0--1.2 seconds. Initial figure export took 10 seconds
  while Matplotlib built its font cache and 0.7 seconds thereafter.

This verifies software behavior, not scientific performance. Corrected estimates on the
study data remain author-controlled because the public package intentionally excludes
participant-level records.
