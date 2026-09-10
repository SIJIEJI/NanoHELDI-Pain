# Release checklist

- [x] Original working files were not modified, deleted, or overwritten by this package.
- [x] Historical seed searches are absent from the canonical analysis workflow.
- [x] One prespecified estimator random state is recorded and never optimized.
- [x] Participant-grouped folds are mandatory.
- [x] The manuscript protocol uses a participant-disjoint 80:20 holdout and grouped
  cross-validation within training data.
- [x] Preprocessing and feature selection are fitted inside each training fold.
- [x] All algorithms within a target reuse exactly the same fold assignments.
- [x] Pre-cross-validation oversampling is absent.
- [x] Pain intensity uses regression and conventional R².
- [x] Fold models and full-data deployment candidates can be saved separately.
- [x] Existing non-empty output directories are protected from overwriting.
- [x] Synthetic tests verify participant separation and fitted pipeline persistence.
- [x] ROC coordinates, confusion matrices, feature-count curves, and fold-level plot
  points are generated from saved run artifacts.
- [x] Small simulated classification and regression datasets are included for the demo.
- [x] README documents requirements, tested versions, installation, expected outputs,
  runtimes, own-data usage, and reproduction steps.
- [x] Raw participant data, exact collection dates, and local machine paths are excluded.
- [ ] Authors have approved the MIT license and completed author metadata.
- [x] Authors have supplied the final repository URL.
- [ ] Authors have supplied the final manuscript citation and DOI.
- [ ] The deidentified analysis table has passed institutional disclosure review.
- [ ] An external or temporally independent validation cohort has been evaluated.
