# Changelog

## 0.2.0 — 2026-09-10

- Added a participant-level 80:20 holdout protocol with grouped training-only
  cross-validation, independent test metrics, ROC coordinates, and confusion matrices.
- Added the AdaBoost and KNN model families and manuscript-style MLP pipelines.
- Added fold-local combined ANOVA F/mutual-information feature ranking and feature-count
  curves.
- Added manuscript result plotting, including individual cross-validation fold points.
- Added optional SHAP and integrated-gradients analysis for saved MLP pipelines.
- Added saved holdout and training-CV assignments and fitted holdout models.
- Updated the README and methods documentation with direct reproduction commands.

## 0.1.0 — 2026-09-09

- Replaced seed-search workflows with one prespecified estimator random state.
- Added deterministic participant-grouped cross-validation.
- Moved imputation, scaling, and feature selection inside fold-local pipelines.
- Removed pre-cross-validation oversampling from the release workflow.
- Added continuous pain-intensity regression with conventional metrics.
- Added fold-specific and full-data model persistence with run provenance.
- Added synthetic tests, CI, static release checks, and a complete legacy inventory.
- Included small classification and regression demo datasets with measured demo runtimes.
- Expanded the README with system requirements, tested versions, installation time,
  expected outputs, own-data instructions, and reproducibility commands.
