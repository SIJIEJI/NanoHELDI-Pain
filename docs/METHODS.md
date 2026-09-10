# Publication-oriented analysis methods

## Unit of analysis

Technical mass-spectrometry replicates must be averaged before this workflow. Each
input row represents one biological observation, and every row carries a deidentified
participant identifier. Repeated visits are allowed but are kept together during
validation.

## Model evaluation

Classification uses deterministic five-fold `StratifiedGroupKFold` without fold
shuffling. Regression uses deterministic five-fold `GroupKFold`. The number of folds
is reduced only when the number of participants or the smallest class cannot support
five folds. Before fitting each model, the program asserts that the training and
validation participant sets are disjoint.

All models within a task use the same saved outer-fold assignments. Missing-value
imputation, standardization, and univariate feature selection are members of the
scikit-learn pipeline and are fitted using the training fold only. No synthetic or
duplicated observations are created before cross-validation.

Classification reports accuracy, balanced accuracy, and macro-F1. Balanced accuracy
is the prespecified primary metric when classes are imbalanced. Pain intensity is
treated as a continuous outcome and reports MAE, RMSE, conventional coefficient of
determination (R²), and the prespecified proportion of predictions within ±2 units.

## Randomness policy

Random-state optimization is prohibited. Stochastic estimators use one constant
`random_state=42` solely so that fitted artifacts can be reproduced. The grouped fold
assignment itself is deterministic and does not use a random seed. Changing this state
after inspecting results creates a new exploratory analysis and must not replace the
prespecified result.

## Model artifacts

The software saves one fitted pipeline per outer fold and model, plus a full-data
deployment candidate. The latter is never used to calculate cross-validation metrics.
Model artifacts include the fitted imputer, scaler, selected features, estimator, label
encoder where applicable, and validation sample identifiers. Exact retraining also
requires the recorded input hash and software versions.

## Limitations

Grouped internal cross-validation reduces participant leakage but does not prove
generalization. With a small cohort, fold estimates remain variable and confidence
intervals may be wide. Final claims require an external or temporally independent
validation cohort.
