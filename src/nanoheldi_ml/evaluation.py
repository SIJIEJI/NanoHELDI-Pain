"""Grouped cross-validation, artifact persistence, and run provenance."""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Iterable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from .config import REPRODUCIBILITY_RANDOM_STATE
from .data import AnalysisData, assert_no_group_overlap


def prepare_output_directory(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(
            f"Output path is not an empty directory: {output}. "
            "Choose a new path to preserve prior runs."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def _classification_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    requested_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_splits = min(requested_splits, np.unique(groups).size, np.bincount(y).min())
    if n_splits < 2:
        raise ValueError("At least two folds and two observations per class are required")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
    splits = list(splitter.split(X, y, groups))
    for train_index, test_index in splits:
        assert_no_group_overlap(groups[train_index], groups[test_index])
    return splits


def _regression_splits(
    X: pd.DataFrame,
    groups: np.ndarray,
    requested_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_splits = min(requested_splits, np.unique(groups).size)
    if n_splits < 2:
        raise ValueError("At least two participant groups are required")
    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(X, groups=groups))
    for train_index, test_index in splits:
        assert_no_group_overlap(groups[train_index], groups[test_index])
    return splits


def _selected_features(fitted_pipeline, feature_names: list[str]) -> list[str]:
    selector = fitted_pipeline.named_steps.get("select")
    if selector is None or not hasattr(selector, "get_support"):
        return feature_names
    support = selector.get_support()
    return np.asarray(feature_names, dtype=object)[support].astype(str).tolist()


def _write_common_outputs(
    output: Path,
    fold_rows: list[dict],
    prediction_rows: list[dict],
    split_rows: list[dict],
    feature_rows: list[dict],
    metadata: dict,
    metric_columns: list[str],
) -> None:
    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    splits = pd.DataFrame(split_rows).drop_duplicates().sort_values(["fold", "sample_id"])
    selected = pd.DataFrame(feature_rows)

    summary_rows = []
    for model_name, subset in fold_metrics.groupby("model", sort=False):
        row = {
            "model": model_name,
            "n_folds": len(subset),
            "n_participants": int(metadata["n_participants"]),
            "n_observations": int(metadata["n_observations"]),
        }
        for metric in metric_columns:
            values = subset[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=0))
        summary_rows.append(row)

    fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output / "summary_metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    splits.to_csv(output / "split_assignments.csv", index=False)
    selected.to_csv(output / "selected_features.csv", index=False)
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _base_metadata(data: AnalysisData, task: str, target: str, n_splits: int) -> dict:
    return {
        "analysis_version": "0.2.0",
        "task": task,
        "target": target,
        "n_observations": len(data.y),
        "n_participants": int(np.unique(data.groups).size),
        "n_input_features": int(data.X.shape[1]),
        "n_splits_requested": int(n_splits),
        "input_sha256": data.input_sha256,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "reproducibility_random_state": REPRODUCIBILITY_RANDOM_STATE,
        "random_state_policy": (
            "One prespecified state for stochastic estimator reproducibility; "
            "no state search or performance-based selection. CV splitting is deterministic."
        ),
        "validation_policy": (
            "Participant-grouped outer cross-validation with imputation, scaling, and "
            "feature selection fitted independently inside each training fold."
        ),
        "warnings": [
            "Cross-validation folds are not independent biological replicates.",
            "Internal validation does not replace an external validation cohort.",
            "The full-data model is not used to calculate reported validation metrics.",
        ],
    }


def evaluate_classification(
    data: AnalysisData,
    models: dict,
    output: str | Path,
    target_name: str,
    n_splits: int = 5,
    save_models: bool = True,
) -> Path:
    output = prepare_output_directory(output)
    encoder = LabelEncoder()
    y = encoder.fit_transform(data.y)
    splits = _classification_splits(data.X, y, data.groups, n_splits)
    model_root = output / "models"
    if save_models:
        model_root.mkdir()

    fold_rows: list[dict] = []
    prediction_rows: list[dict] = []
    split_rows: list[dict] = []
    feature_rows: list[dict] = []

    for fold_number, (train_index, test_index) in enumerate(splits, start=1):
        for index in train_index:
            split_rows.append(
                {
                    "fold": fold_number,
                    "role": "train",
                    "sample_id": data.sample_ids[index],
                    "participant_id": data.groups[index],
                }
            )
        for index in test_index:
            split_rows.append(
                {
                    "fold": fold_number,
                    "role": "validation",
                    "sample_id": data.sample_ids[index],
                    "participant_id": data.groups[index],
                }
            )

        for model_name, estimator in models.items():
            fitted = clone(estimator)
            fitted.fit(data.X.iloc[train_index], y[train_index])
            prediction = fitted.predict(data.X.iloc[test_index])
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold_number,
                    "accuracy": accuracy_score(y[test_index], prediction),
                    "balanced_accuracy": balanced_accuracy_score(y[test_index], prediction),
                    "macro_f1": f1_score(
                        y[test_index], prediction, average="macro", zero_division=0
                    ),
                    "n_train": len(train_index),
                    "n_validation": len(test_index),
                    "n_train_participants": int(np.unique(data.groups[train_index]).size),
                    "n_validation_participants": int(np.unique(data.groups[test_index]).size),
                    "participant_overlap": 0,
                }
            )
            for row_index, encoded_prediction in zip(test_index, prediction):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "fold": fold_number,
                        "sample_id": data.sample_ids[row_index],
                        "participant_id": data.groups[row_index],
                        "observed": data.y.iloc[row_index],
                        "predicted": encoder.inverse_transform([int(encoded_prediction)])[0],
                    }
                )
            for selection_index, feature in enumerate(
                _selected_features(fitted, data.feature_names), start=1
            ):
                feature_rows.append(
                    {
                        "model": model_name,
                        "fold": fold_number,
                        "selection_index": selection_index,
                        "feature": feature,
                    }
                )
            if save_models:
                model_directory = model_root / model_name
                model_directory.mkdir(exist_ok=True)
                joblib.dump(
                    {
                        "pipeline": fitted,
                        "label_encoder": encoder,
                        "feature_names": data.feature_names,
                        "target": target_name,
                        "fold": fold_number,
                        "validation_sample_ids": data.sample_ids[test_index].tolist(),
                    },
                    model_directory / f"fold_{fold_number}.joblib",
                )

    if save_models:
        for model_name, estimator in models.items():
            fitted = clone(estimator).fit(data.X, y)
            joblib.dump(
                {
                    "pipeline": fitted,
                    "label_encoder": encoder,
                    "feature_names": data.feature_names,
                    "target": target_name,
                    "training_scope": "all available observations; not used for CV metrics",
                },
                model_root / model_name / "final_full_data.joblib",
            )

    metadata = _base_metadata(data, "classification", target_name, n_splits)
    metadata["models"] = list(models)
    metadata["class_labels"] = [str(item) for item in encoder.classes_]
    metadata["n_splits_used"] = len(splits)
    metadata["saved_fold_models"] = bool(save_models)
    _write_common_outputs(
        output,
        fold_rows,
        prediction_rows,
        split_rows,
        feature_rows,
        metadata,
        ["accuracy", "balanced_accuracy", "macro_f1"],
    )
    return output


def evaluate_regression(
    data: AnalysisData,
    models: dict,
    output: str | Path,
    target_name: str,
    n_splits: int = 5,
    tolerance: float = 2.0,
    save_models: bool = True,
) -> Path:
    output = prepare_output_directory(output)
    y = pd.to_numeric(data.y, errors="raise").to_numpy(float)
    splits = _regression_splits(data.X, data.groups, n_splits)
    model_root = output / "models"
    if save_models:
        model_root.mkdir()

    fold_rows: list[dict] = []
    prediction_rows: list[dict] = []
    split_rows: list[dict] = []
    feature_rows: list[dict] = []

    for fold_number, (train_index, test_index) in enumerate(splits, start=1):
        for role, indices in (("train", train_index), ("validation", test_index)):
            for index in indices:
                split_rows.append(
                    {
                        "fold": fold_number,
                        "role": role,
                        "sample_id": data.sample_ids[index],
                        "participant_id": data.groups[index],
                    }
                )

        for model_name, estimator in models.items():
            fitted = clone(estimator)
            fitted.fit(data.X.iloc[train_index], y[train_index])
            prediction = fitted.predict(data.X.iloc[test_index])
            errors = np.abs(y[test_index] - prediction)
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold_number,
                    "mae": mean_absolute_error(y[test_index], prediction),
                    "rmse": mean_squared_error(y[test_index], prediction) ** 0.5,
                    "r2": r2_score(y[test_index], prediction),
                    "within_tolerance_accuracy": float(np.mean(errors <= tolerance)),
                    "tolerance": float(tolerance),
                    "n_train": len(train_index),
                    "n_validation": len(test_index),
                    "n_train_participants": int(np.unique(data.groups[train_index]).size),
                    "n_validation_participants": int(np.unique(data.groups[test_index]).size),
                    "participant_overlap": 0,
                }
            )
            for row_index, predicted in zip(test_index, prediction):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "fold": fold_number,
                        "sample_id": data.sample_ids[row_index],
                        "participant_id": data.groups[row_index],
                        "observed": y[row_index],
                        "predicted": float(predicted),
                    }
                )
            for selection_index, feature in enumerate(
                _selected_features(fitted, data.feature_names), start=1
            ):
                feature_rows.append(
                    {
                        "model": model_name,
                        "fold": fold_number,
                        "selection_index": selection_index,
                        "feature": feature,
                    }
                )
            if save_models:
                model_directory = model_root / model_name
                model_directory.mkdir(exist_ok=True)
                joblib.dump(
                    {
                        "pipeline": fitted,
                        "feature_names": data.feature_names,
                        "target": target_name,
                        "fold": fold_number,
                        "validation_sample_ids": data.sample_ids[test_index].tolist(),
                    },
                    model_directory / f"fold_{fold_number}.joblib",
                )

    if save_models:
        for model_name, estimator in models.items():
            fitted = clone(estimator).fit(data.X, y)
            joblib.dump(
                {
                    "pipeline": fitted,
                    "feature_names": data.feature_names,
                    "target": target_name,
                    "training_scope": "all available observations; not used for CV metrics",
                },
                model_root / model_name / "final_full_data.joblib",
            )

    metadata = _base_metadata(data, "regression", target_name, n_splits)
    metadata["models"] = list(models)
    metadata["n_splits_used"] = len(splits)
    metadata["tolerance"] = float(tolerance)
    metadata["saved_fold_models"] = bool(save_models)
    _write_common_outputs(
        output,
        fold_rows,
        prediction_rows,
        split_rows,
        feature_rows,
        metadata,
        ["mae", "rmse", "r2", "within_tolerance_accuracy"],
    )
    return output


def select_models(models: dict, requested: Iterable[str] | None) -> dict:
    if not requested:
        return models
    requested = list(requested)
    unknown = sorted(set(requested).difference(models))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}; available models are {list(models)}")
    return {name: models[name] for name in requested}
