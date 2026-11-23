"""Shared, prespecified analysis constants."""

REPRODUCIBILITY_RANDOM_STATE = 42
DEFAULT_N_SPLITS = 5
SUPPORTED_CV_FOLDS = range(2, DEFAULT_N_SPLITS + 1)
DEFAULT_N_FEATURES = 50
RESERVED_COLUMNS = {
    "sample_id",
    "participant_id",
    "visit_id",
    "replicate_id",
    "target",
    "pain_score",
}


def validate_cv_folds(folds: int) -> None:
    if folds not in SUPPORTED_CV_FOLDS:
        raise ValueError(f"Grouped cross-validation supports 2 through {DEFAULT_N_SPLITS} folds.")
