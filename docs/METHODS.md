# Machine-learning analysis methods

## Analysis table and grouping

The software accepts an observation-level feature matrix with a unique sample ID and a
deidentified participant ID for every row. Repeated measurements and technical
replicates may be retained when required by the prespecified analysis, but every row
from one participant is assigned to the same evaluation partition. Missing feature
values are median-imputed from training data only, followed by standardization fitted
to the same training data.

## Manuscript classification protocol

Classification uses a participant-disjoint 80:20 training/test partition. Candidate
models are compared using up to 10 folds of `StratifiedGroupKFold` within the training
partition. The code reduces the fold count only when the available participants or
class counts cannot support the requested value. All models use identical saved fold
assignments.

The model set contains a stratified random baseline, Gaussian naive Bayes, decision
tree, logistic regression, gradient boosting, support-vector machine, random forest,
AdaBoost, k-nearest neighbors, multilayer perceptron, and optional XGBoost. The MLP has
one 100-unit hidden layer with ReLU activation and Adam optimization. A single declared
random state is used for reproducible randomized operations and is never searched.

Training-CV outputs include accuracy, balanced accuracy, macro-F1, and macro
one-vs-rest ROC AUC for every fold. The model-ranking field in the run metadata is based
only on mean training-CV macro one-vs-rest AUC. The independent test partition is then
evaluated once for every fitted model, producing the same metrics, per-sample class
scores, ROC coordinates, and confusion matrices.

## Manuscript regression protocol

Continuous pain outcomes use a participant-disjoint 80:20 training/test partition and
up to 10 folds of grouped cross-validation within the training partition. Regression
reports mean absolute error, root mean squared error, conventional R², and the fraction
of predictions within the prespecified ±2-unit tolerance. The tolerance metric is
reported alongside, not instead of, continuous-error metrics.

## Feature ranking and feature-count curves

Feature-count analyses combine normalized univariate ANOVA F values and mutual
information scores with equal weight. Imputation and both ranking components are
fitted independently in each training fold. The top 10, 20, 30, 40, and 50 features
are therefore selected without using the corresponding validation fold. Output files
contain every fold score, the mean and SD across folds, and the selected feature names
for audit.

## Model artifacts and provenance

For the manuscript holdout protocol, each saved model is fitted on the training
partition only. The artifact includes the complete preprocessing pipeline, feature
names, target name, training sample IDs, test sample IDs, and label encoder when
applicable. The test partition is not used for fitting or model selection. Run metadata
also records the input SHA-256 hash, software versions, split sizes, requested and used
fold counts, and the declared random state.

## MLP interpretation

The optional interpretation command applies permutation SHAP and finite-difference
integrated gradients to the saved MLP pipeline. A maximum of 100 rows is selected at
even intervals from the designated test partition, while the reference background is
selected from training rows only. Absolute attributions are averaged by feature;
separately normalized SHAP and integrated-gradient importance values are combined with
equal weight. Interpretation occurs after evaluation and does not refit the model or
alter predictions.

## Interpretation

Cross-validation folds are evaluation partitions rather than independent biological
replicates. Figures display fold-level points when presenting fold summaries; the
biological `n` is the number of unique participants recorded in `run_metadata.json`.
Internal validation does not substitute for validation in an external or temporally
independent cohort.
