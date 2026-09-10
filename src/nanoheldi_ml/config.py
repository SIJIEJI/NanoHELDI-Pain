"""Shared, prespecified analysis constants."""

REPRODUCIBILITY_RANDOM_STATE = 42
DEFAULT_N_SPLITS = 5
DEFAULT_N_FEATURES = 50
RESERVED_COLUMNS = {
    "sample_id",
    "participant_id",
    "visit_id",
    "replicate_id",
    "target",
    "pain_score",
}
