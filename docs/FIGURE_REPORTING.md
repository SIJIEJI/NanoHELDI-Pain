# Figure reporting guidance

For model-comparison figures generated from `fold_metrics.csv`:

- bar center: arithmetic mean across participant-grouped validation folds;
- error bar: population SD across the same folds (`ddof=0`);
- points: individual fold scores;
- biological `n`: number of unique participants from `run_metadata.json`;
- point count: number of CV folds, which is not the biological sample size;
- primary classification metric: balanced accuracy when classes are imbalanced.

All algorithms within a target must use the identical `split_assignments.csv`. Do not
combine one algorithm's score from a different split or random state with the remaining
bars. When results from several targets are displayed together, report participant and
observation counts for each target if missing labels cause those counts to differ.
