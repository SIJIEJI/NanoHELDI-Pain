"""Fold-local ANOVA/mutual-information ranking and feature-count curves."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

from .config import REPRODUCIBILITY_RANDOM_STATE
from .data import AnalysisData
from .evaluation import prepare_output_directory
from .manuscript_evaluation import _classification_cv_splits, _regression_cv_splits
from .models import manuscript_classification_models, manuscript_regression_models


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    maximum = float(values.max(initial=0.0))
    return values / maximum if maximum > 0 else np.zeros_like(values)


def combined_classification_ranking(
    X: pd.DataFrame, y: np.ndarray
) -> pd.DataFrame:
    """Combine normalized ANOVA F and mutual information scores with equal weight."""
    imputed = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    f_values, _ = f_classif(imputed, y)
    mutual_information = mutual_info_classif(
        imputed, y, random_state=REPRODUCIBILITY_RANDOM_STATE
    )
    combined = (_normalize(f_values) + _normalize(mutual_information)) / 2.0
    frame = pd.DataFrame(
        {
            "feature": X.columns.astype(str),
            "anova_f": np.nan_to_num(f_values, nan=0.0),
            "mutual_information": mutual_information,
            "combined_importance": combined,
        }
    )
    return frame.sort_values("combined_importance", ascending=False).reset_index(drop=True)


def combined_regression_ranking(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    imputed = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    f_values, _ = f_regression(imputed, y)
    mutual_information = mutual_info_regression(
        imputed, y, random_state=REPRODUCIBILITY_RANDOM_STATE
    )
    combined = (_normalize(f_values) + _normalize(mutual_information)) / 2.0
    frame = pd.DataFrame(
        {
            "feature": X.columns.astype(str),
            "anova_f": np.nan_to_num(f_values, nan=0.0),
            "mutual_information": mutual_information,
            "combined_importance": combined,
        }
    )
    return frame.sort_values("combined_importance", ascending=False).reset_index(drop=True)


def _valid_counts(counts: list[int], n_features: int) -> list[int]:
    values = sorted({min(max(1, int(count)), n_features) for count in counts})
    if not values:
        raise ValueError("At least one feature count is required")
    return values


def classification_feature_curve(
    data: AnalysisData,
    output: str | Path,
    counts: list[int],
    folds: int = 5,
) -> Path:
    """Evaluate MLP accuracy across feature counts with ranking fitted in each fold."""
    output = prepare_output_directory(output)
    encoder = LabelEncoder()
    y = encoder.fit_transform(data.y)
    splits = _classification_cv_splits(data.X, y, data.groups, folds)
    counts = _valid_counts(counts, data.X.shape[1])
    estimator = manuscript_classification_models(include_xgboost=False)["MLP"]
    metric_rows: list[dict] = []
    feature_rows: list[dict] = []

    for fold_number, (train_index, validation_index) in enumerate(splits, start=1):
        ranking = combined_classification_ranking(data.X.iloc[train_index], y[train_index])
        for count in counts:
            selected = ranking.head(count)["feature"].tolist()
            fitted = clone(estimator).fit(data.X.iloc[train_index][selected], y[train_index])
            prediction = fitted.predict(data.X.iloc[validation_index][selected])
            metric_rows.append(
                {
                    "fold": fold_number,
                    "n_features": count,
                    "accuracy": accuracy_score(y[validation_index], prediction),
                    "n_validation_participants": int(
                        np.unique(data.groups[validation_index]).size
                    ),
                    "participant_overlap": 0,
                }
            )
            for rank, feature in enumerate(selected, start=1):
                feature_rows.append(
                    {
                        "fold": fold_number,
                        "n_features": count,
                        "rank": rank,
                        "feature": feature,
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    summary = metrics.groupby("n_features", sort=True)["accuracy"].agg(["mean", "std"])
    summary.columns = ["accuracy_mean", "accuracy_sd"]
    summary.reset_index().to_csv(output / "feature_curve_summary.csv", index=False)
    metrics.to_csv(output / "feature_curve_fold_metrics.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(output / "selected_features.csv", index=False)
    metadata = {
        "task": "classification",
        "target": str(data.y.name),
        "model": "MLP",
        "ranking": "equal-weight normalized ANOVA F plus mutual information",
        "ranking_scope": "fitted independently inside each training fold",
        "folds": len(splits),
        "counts": counts,
        "n_participants": int(np.unique(data.groups).size),
        "input_sha256": data.input_sha256,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def regression_feature_curve(
    data: AnalysisData,
    output: str | Path,
    counts: list[int],
    folds: int = 5,
    tolerance: float = 2.0,
) -> Path:
    """Evaluate MLP regression across feature counts with ranking fitted in each fold."""
    output = prepare_output_directory(output)
    y = pd.to_numeric(data.y, errors="raise").to_numpy(float)
    splits = _regression_cv_splits(data.X, data.groups, folds)
    counts = _valid_counts(counts, data.X.shape[1])
    estimator = manuscript_regression_models()["MLP"]
    rows: list[dict] = []

    for fold_number, (train_index, validation_index) in enumerate(splits, start=1):
        ranking = combined_regression_ranking(data.X.iloc[train_index], y[train_index])
        for count in counts:
            selected = ranking.head(count)["feature"].tolist()
            fitted = clone(estimator).fit(data.X.iloc[train_index][selected], y[train_index])
            prediction = fitted.predict(data.X.iloc[validation_index][selected])
            error = np.abs(y[validation_index] - prediction)
            rows.append(
                {
                    "fold": fold_number,
                    "n_features": count,
                    "mae": mean_absolute_error(y[validation_index], prediction),
                    "r2": r2_score(y[validation_index], prediction),
                    "within_tolerance_accuracy": float(np.mean(error <= tolerance)),
                    "participant_overlap": 0,
                }
            )

    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("n_features", sort=True)[
        ["mae", "r2", "within_tolerance_accuracy"]
    ].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(output / "feature_curve_summary.csv", index=False)
    metrics.to_csv(output / "feature_curve_fold_metrics.csv", index=False)
    metadata = {
        "task": "regression",
        "target": str(data.y.name),
        "model": "MLP",
        "ranking": "equal-weight normalized ANOVA F plus mutual information",
        "ranking_scope": "fitted independently inside each training fold",
        "folds": len(splits),
        "counts": counts,
        "tolerance": float(tolerance),
        "n_participants": int(np.unique(data.groups).size),
        "input_sha256": data.input_sha256,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
