# Migration map

| Historical pattern | Release replacement |
|---|---|
| Searching hundreds or thousands of random seeds | One prespecified reproducibility state; no search |
| Reporting the best seed | Report deterministic grouped-CV estimates |
| `train_test_split` on repeated measurements | Participant-grouped cross-validation |
| Scaling the full matrix before splitting | `StandardScaler` inside each model pipeline |
| Selecting m/z features before splitting | `SelectKBest` inside each model pipeline |
| Oversampling before cross-validation | Removed; balanced metrics and class weighting used where supported |
| Different splits for different algorithms | A single saved split list reused by every algorithm |
| MLP classifier for pain intensity | `MLPRegressor` with conventional regression metrics |
| Custom “tolerant R²” | Conventional R² plus separately named ±2 accuracy |
| `cross_val_score` discarding estimators | Explicit fold loop saving every fitted pipeline |
| Embedded participant records and dates | External deidentified input table required |
| Hard-coded local paths | Command-line paths and repository-relative outputs |
| Overwriting result directories | Refuse to run when the requested output is non-empty |

The complete legacy inventory is in `legacy_code_inventory.csv`. Original scripts remain
untouched in the private working directory and are intentionally absent from this public
package.
