# Model card

## Intended use

Research-only evaluation of mass-spectrometry feature tables for pain-related
classification and pain-intensity regression. The software is not a medical device and
must not be used for diagnosis, treatment, triage, or individual clinical decisions.

## Evaluation population

The repository contains no participant data. Users must report cohort inclusion criteria,
unique participant count, observation count, class distribution, missingness, and the
relationship between repeated visits and technical replicates for every run.

## Primary risks

- Small-cohort variance and unstable model rankings.
- Participant leakage when identifiers are missing or incorrect.
- Optimism from post-hoc choice of folds, features, model settings, or random state.
- Distribution shift across collection dates, operators, batches, instruments, or sites.
- Misinterpretation of cross-validation folds as independent biological replicates.

## Required safeguards

Use deidentified participant IDs, preserve grouped folds, keep preprocessing inside the
pipeline, retain all run metadata, and validate the frozen analysis on an external cohort.
