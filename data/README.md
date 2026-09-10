# Data schema

Participant-level data are not included in this repository.

Prepare the deidentified observation-level CSV used by the prespecified analysis. If
technical replicates are retained as separate rows, assign them the same
`participant_id` so they remain in one evaluation partition:

| Column | Required | Description |
|---|---:|---|
| `sample_id` | yes | Unique deidentified observation identifier |
| `participant_id` | yes | Stable deidentified participant/group identifier |
| target column | yes | Classification label or continuous pain score |
| `mz_*` | yes | Numeric mass-spectrometry features |
| `visit_id` | optional | Deidentified visit label; do not include exact dates |

Do not commit the analysis CSV. The repository `.gitignore` excludes CSV and Excel files
under `data/`. If sharing participant-level data is not ethically or legally permitted,
publish aggregate source data separately and state the controlled-access procedure.
