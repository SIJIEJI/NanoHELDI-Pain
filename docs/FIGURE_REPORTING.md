# Figure reporting guidance

For general grouped-CV model-comparison figures generated from `fold_metrics.csv`, or
manuscript figures generated from `training_cv_fold_metrics.csv`:

- bar center: arithmetic mean across participant-grouped validation folds;
- error bar: sample SD across the same folds (`ddof=1`);
- points: individual fold scores;
- biological `n`: number of unique participants from `run_metadata.json`;
- point count: number of CV folds, which is not the biological sample size;
- manuscript model-selection metric: macro one-vs-rest ROC AUC calculated within the
  training partition.

All algorithms within a target must use the identical saved assignments. Do not
combine one algorithm's score from a different partition with the remaining bars.
When results from several targets are displayed together, report participant and
observation counts for each target if missing labels cause those counts to differ.
