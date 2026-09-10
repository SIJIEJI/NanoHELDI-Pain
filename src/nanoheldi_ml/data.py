"""Input validation for deidentified, observation-level analysis tables."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnalysisData:
    X: pd.DataFrame
    y: pd.Series
    groups: np.ndarray
    sample_ids: np.ndarray
    feature_names: list[str]
    input_sha256: str


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_analysis_table(
    path: str | Path,
    target_column: str,
    participant_column: str = "participant_id",
    sample_column: str = "sample_id",
    feature_prefix: str = "mz_",
) -> AnalysisData:
    """Load an observation-level table and enforce publication invariants."""
    path = Path(path).expanduser().resolve()
    table = pd.read_csv(path)
    required = {target_column, participant_column, sample_column}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if table[sample_column].isna().any() or table[sample_column].duplicated().any():
        raise ValueError("sample_id values must be non-missing and unique")
    if table[participant_column].isna().any():
        raise ValueError("participant_id cannot be missing; grouped evaluation is mandatory")
    if table[target_column].isna().any():
        raise ValueError("Target values cannot be missing")

    feature_names = [
        str(column)
        for column in table.columns
        if str(column).startswith(feature_prefix)
    ]
    if not feature_names:
        raise ValueError(f"No feature columns start with {feature_prefix!r}")

    X = table[feature_names].apply(pd.to_numeric, errors="coerce")
    if X.isna().all(axis=0).any():
        bad = X.columns[X.isna().all(axis=0)].tolist()
        raise ValueError(f"Entirely missing feature columns: {bad}")

    groups = table[participant_column].astype(str).to_numpy()
    if np.unique(groups).size < 2:
        raise ValueError("At least two unique participants are required")

    return AnalysisData(
        X=X,
        y=table[target_column].copy(),
        groups=groups,
        sample_ids=table[sample_column].astype(str).to_numpy(),
        feature_names=feature_names,
        input_sha256=file_sha256(path),
    )


def assert_no_group_overlap(train_groups: np.ndarray, test_groups: np.ndarray) -> None:
    overlap = set(map(str, train_groups)).intersection(map(str, test_groups))
    if overlap:
        raise RuntimeError(f"Participant leakage detected across a fold ({len(overlap)} groups)")
