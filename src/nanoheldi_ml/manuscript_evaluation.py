"""Manuscript-aligned subject-level holdout and training-CV evaluation."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, label_binarize

from .config import REPRODUCIBILITY_RANDOM_STATE
from .data import AnalysisData, assert_no_group_overlap
from .evaluation import prepare_output_directory


def _classification_holdout(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be greater than 0 and less than 0.5")
    n_splits = max(2, round(1.0 / test_fraction))
    n_splits = min(n_splits, np.unique(groups).size, int(np.bincount(y).min()))
    if n_splits < 2:
        raise ValueError("The classes and participant groups cannot support a holdout split")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=REPRODUCIBILITY_RANDOM_STATE,
    )
    candidates = list(splitter.split(X, y, groups))
    overall = np.bincount(y, minlength=int(y.max()) + 1) / len(y)

    def split_distance(indices: tuple[np.ndarray, np.ndarray]) -> float:
        _, test_index = indices
        test_distribution = np.bincount(
            y[test_index], minlength=len(overall)
        ) / len(test_index)
        return abs(len(test_index) / len(y) - test_fraction) + float(
            np.abs(test_distribution - overall).sum()
        )

    train_index, test_index = min(candidates, key=split_distance)
    assert_no_group_overlap(groups[train_index], groups[test_index])
    return train_index, test_index


def _regression_holdout(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be greater than 0 and less than 0.5")
    n_splits = max(2, round(1.0 / test_fraction))
    n_splits = min(n_splits, np.unique(groups).size)
    n_bins = min(5, max(2, len(y) // (2 * n_splits)))
    ranked = pd.Series(y).rank(method="first")
    bins = pd.qcut(ranked, q=n_bins, labels=False, duplicates="drop").to_numpy(int)
    try:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=REPRODUCIBILITY_RANDOM_STATE,
        )
        train_index, test_index = next(splitter.split(X, bins, groups))
        method = "stratified_group_holdout_using_quantile_binned_continuous_target"
    except ValueError:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_fraction,
            random_state=REPRODUCIBILITY_RANDOM_STATE,
        )
        train_index, test_index = next(splitter.split(X, y, groups))
        method = "group_holdout_without_stratification_fallback"
    assert_no_group_overlap(groups[train_index], groups[test_index])
    return train_index, test_index, method


def _classification_cv_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    requested_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_splits = min(requested_splits, np.unique(groups).size, int(np.bincount(y).min()))
    if n_splits < 2:
        raise ValueError("The training set cannot support at least two grouped CV folds")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=REPRODUCIBILITY_RANDOM_STATE,
    )
    splits = list(splitter.split(X, y, groups))
    for train_index, validation_index in splits:
        assert_no_group_overlap(groups[train_index], groups[validation_index])
    return splits


def _regression_cv_splits(
    X: pd.DataFrame,
    groups: np.ndarray,
    requested_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_splits = min(requested_splits, np.unique(groups).size)
    if n_splits < 2:
        raise ValueError("The training set cannot support at least two grouped CV folds")
    splits = list(GroupKFold(n_splits=n_splits).split(X, groups=groups))
    for train_index, validation_index in splits:
        assert_no_group_overlap(groups[train_index], groups[validation_index])
    return splits


def _score_matrix(fitted, X: pd.DataFrame, n_classes: int) -> np.ndarray:
    if hasattr(fitted, "predict_proba"):
        raw = np.asarray(fitted.predict_proba(X), dtype=float)
    elif hasattr(fitted, "decision_function"):
        decision = np.asarray(fitted.decision_function(X), dtype=float)
        if decision.ndim == 1:
            positive = 1.0 / (1.0 + np.exp(-decision))
            raw = np.column_stack([1.0 - positive, positive])
        else:
            shifted = decision - decision.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            raw = exp / exp.sum(axis=1, keepdims=True)
    else:
        prediction = np.asarray(fitted.predict(X), dtype=int)
        raw = np.eye(n_classes, dtype=float)[prediction]

    classes = np.asarray(getattr(fitted, "classes_", np.arange(raw.shape[1])), dtype=int)
    aligned = np.zeros((len(X), n_classes), dtype=float)
    for source_column, encoded_class in enumerate(classes):
        aligned[:, int(encoded_class)] = raw[:, source_column]
    return aligned


def _macro_auc(y: np.ndarray, scores: np.ndarray) -> float:
    try:
        if scores.shape[1] == 2:
            return float(roc_auc_score(y, scores[:, 1]))
        return float(roc_auc_score(y, scores, average="macro", multi_class="ovr"))
    except ValueError:
        return float("nan")


def _metadata(data: AnalysisData, task: str, target: str, test_fraction: float) -> dict:
    return {
        "analysis_version": "0.2.0",
        "protocol": "manuscript_subject_holdout",
        "task": task,
        "target": target,
        "n_observations": len(data.y),
        "n_participants": int(np.unique(data.groups).size),
        "n_input_features": int(data.X.shape[1]),
        "test_fraction_requested": float(test_fraction),
        "input_sha256": data.input_sha256,
        "random_state": REPRODUCIBILITY_RANDOM_STATE,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "selection_policy": (
            "The test set is not used to select the model. Candidate models are ranked "
            "using grouped cross-validation within the training partition."
        ),
    }


def evaluate_manuscript_classification(
    data: AnalysisData,
    models: dict,
    output: str | Path,
    target_name: str,
    test_fraction: float = 0.2,
    training_cv_folds: int = 10,
    save_models: bool = True,
) -> Path:
    """Evaluate classification using a subject-level 80:20 holdout and training-only CV."""
    output = prepare_output_directory(output)
    encoder = LabelEncoder()
    y = encoder.fit_transform(data.y)
    n_classes = len(encoder.classes_)
    outer_train, outer_test = _classification_holdout(
        data.X, y, data.groups, test_fraction
    )
    X_train = data.X.iloc[outer_train]
    y_train = y[outer_train]
    groups_train = data.groups[outer_train]
    cv_splits = _classification_cv_splits(
        X_train, y_train, groups_train, training_cv_folds
    )

    fold_rows: list[dict] = []
    cv_prediction_rows: list[dict] = []
    test_rows: list[dict] = []
    test_prediction_rows: list[dict] = []
    roc_rows: list[dict] = []
    confusion_rows: list[dict] = []
    cv_assignment_rows: list[dict] = []
    model_root = output / "models"
    if save_models:
        model_root.mkdir()

    for fold_number, (cv_train, cv_validation) in enumerate(cv_splits, start=1):
        for role, indices in (("train", cv_train), ("validation", cv_validation)):
            for local_index in indices:
                global_index = outer_train[local_index]
                cv_assignment_rows.append(
                    {
                        "fold": fold_number,
                        "role": role,
                        "sample_id": data.sample_ids[global_index],
                        "participant_id": data.groups[global_index],
                    }
                )
        for model_name, estimator in models.items():
            fitted = clone(estimator).fit(X_train.iloc[cv_train], y_train[cv_train])
            prediction = fitted.predict(X_train.iloc[cv_validation]).astype(int)
            scores = _score_matrix(fitted, X_train.iloc[cv_validation], n_classes)
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold_number,
                    "accuracy": accuracy_score(y_train[cv_validation], prediction),
                    "balanced_accuracy": balanced_accuracy_score(
                        y_train[cv_validation], prediction
                    ),
                    "macro_f1": f1_score(
                        y_train[cv_validation], prediction, average="macro", zero_division=0
                    ),
                    "macro_ovr_auc": _macro_auc(y_train[cv_validation], scores),
                    "n_train_participants": int(
                        np.unique(groups_train[cv_train]).size
                    ),
                    "n_validation_participants": int(
                        np.unique(groups_train[cv_validation]).size
                    ),
                    "participant_overlap": 0,
                }
            )
            for row, encoded_prediction, score_row in zip(
                cv_validation, prediction, scores
            ):
                global_index = outer_train[row]
                record = {
                    "model": model_name,
                    "fold": fold_number,
                    "sample_id": data.sample_ids[global_index],
                    "participant_id": data.groups[global_index],
                    "observed": data.y.iloc[global_index],
                    "predicted": encoder.inverse_transform([encoded_prediction])[0],
                }
                record.update(
                    {
                        f"score_{label}": float(score_row[index])
                        for index, label in enumerate(encoder.classes_)
                    }
                )
                cv_prediction_rows.append(record)

    for model_name, estimator in models.items():
        fitted = clone(estimator).fit(X_train, y_train)
        prediction = fitted.predict(data.X.iloc[outer_test]).astype(int)
        scores = _score_matrix(fitted, data.X.iloc[outer_test], n_classes)
        test_rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y[outer_test], prediction),
                "balanced_accuracy": balanced_accuracy_score(y[outer_test], prediction),
                "macro_f1": f1_score(
                    y[outer_test], prediction, average="macro", zero_division=0
                ),
                "macro_ovr_auc": _macro_auc(y[outer_test], scores),
                "n_test_observations": len(outer_test),
                "n_test_participants": int(np.unique(data.groups[outer_test]).size),
            }
        )
        matrix = confusion_matrix(y[outer_test], prediction, labels=np.arange(n_classes))
        for true_index, true_label in enumerate(encoder.classes_):
            for predicted_index, predicted_label in enumerate(encoder.classes_):
                confusion_rows.append(
                    {
                        "model": model_name,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": int(matrix[true_index, predicted_index]),
                    }
                )
        binary = label_binarize(y[outer_test], classes=np.arange(n_classes))
        if n_classes == 2:
            binary = np.column_stack([1 - binary[:, 0], binary[:, 0]])
        for class_index, label in enumerate(encoder.classes_):
            fpr, tpr, thresholds = roc_curve(binary[:, class_index], scores[:, class_index])
            for fpr_value, tpr_value, threshold in zip(fpr, tpr, thresholds):
                roc_rows.append(
                    {
                        "model": model_name,
                        "class_label": label,
                        "fpr": float(fpr_value),
                        "tpr": float(tpr_value),
                        "threshold": float(threshold),
                    }
                )
        for global_index, encoded_prediction, score_row in zip(
            outer_test, prediction, scores
        ):
            record = {
                "model": model_name,
                "sample_id": data.sample_ids[global_index],
                "participant_id": data.groups[global_index],
                "observed": data.y.iloc[global_index],
                "predicted": encoder.inverse_transform([encoded_prediction])[0],
            }
            record.update(
                {
                    f"score_{label}": float(score_row[index])
                    for index, label in enumerate(encoder.classes_)
                }
            )
            test_prediction_rows.append(record)
        if save_models:
            joblib.dump(
                {
                    "pipeline": fitted,
                    "label_encoder": encoder,
                    "feature_names": data.feature_names,
                    "target": target_name,
                    "training_sample_ids": data.sample_ids[outer_train].tolist(),
                    "test_sample_ids": data.sample_ids[outer_test].tolist(),
                },
                model_root / f"{model_name}.joblib",
            )

    fold_frame = pd.DataFrame(fold_rows)
    cv_summary = (
        fold_frame.groupby("model", sort=False)[
            ["accuracy", "balanced_accuracy", "macro_f1", "macro_ovr_auc"]
        ]
        .agg(["mean", "std"])
    )
    cv_summary.columns = ["_".join(column) for column in cv_summary.columns]
    cv_summary = cv_summary.reset_index()
    eligible = cv_summary.loc[cv_summary["model"].ne("Random")]
    if eligible.empty:
        eligible = cv_summary
    best_model = eligible.sort_values(
        ["macro_ovr_auc_mean", "accuracy_mean"], ascending=False
    ).iloc[0]["model"]

    pd.DataFrame(fold_rows).to_csv(output / "training_cv_fold_metrics.csv", index=False)
    cv_summary.to_csv(output / "training_cv_summary.csv", index=False)
    pd.DataFrame(cv_prediction_rows).to_csv(
        output / "training_cv_predictions.csv", index=False
    )
    pd.DataFrame(test_rows).to_csv(output / "test_metrics.csv", index=False)
    pd.DataFrame(test_prediction_rows).to_csv(
        output / "test_predictions.csv", index=False
    )
    pd.DataFrame(roc_rows).to_csv(output / "test_roc_curves.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(
        output / "test_confusion_matrices.csv", index=False
    )
    pd.DataFrame(cv_assignment_rows).drop_duplicates().to_csv(
        output / "training_cv_assignments.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "role": "train",
                "sample_id": data.sample_ids[index],
                "participant_id": data.groups[index],
            }
            for index in outer_train
        ]
        + [
            {
                "role": "test",
                "sample_id": data.sample_ids[index],
                "participant_id": data.groups[index],
            }
            for index in outer_test
        ]
    ).to_csv(output / "holdout_assignments.csv", index=False)

    metadata = _metadata(data, "classification", target_name, test_fraction)
    metadata.update(
        {
            "n_training_observations": len(outer_train),
            "n_test_observations": len(outer_test),
            "n_training_participants": int(np.unique(data.groups[outer_train]).size),
            "n_test_participants": int(np.unique(data.groups[outer_test]).size),
            "training_cv_folds_requested": int(training_cv_folds),
            "training_cv_folds_used": len(cv_splits),
            "models": list(models),
            "saved_models": bool(save_models),
            "class_labels": [str(item) for item in encoder.classes_],
            "model_selected_by_training_cv_macro_ovr_auc": str(best_model),
        }
    )
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def evaluate_manuscript_regression(
    data: AnalysisData,
    models: dict,
    output: str | Path,
    target_name: str,
    test_fraction: float = 0.2,
    training_cv_folds: int = 10,
    tolerance: float = 2.0,
    save_models: bool = True,
) -> Path:
    """Evaluate regression using a subject-level 80:20 holdout and training-only CV."""
    output = prepare_output_directory(output)
    y = pd.to_numeric(data.y, errors="raise").to_numpy(float)
    outer_train, outer_test, holdout_method = _regression_holdout(
        data.X, y, data.groups, test_fraction
    )
    X_train = data.X.iloc[outer_train]
    y_train = y[outer_train]
    groups_train = data.groups[outer_train]
    cv_splits = _regression_cv_splits(X_train, groups_train, training_cv_folds)
    fold_rows: list[dict] = []
    cv_prediction_rows: list[dict] = []
    cv_assignment_rows: list[dict] = []
    prediction_rows: list[dict] = []
    test_rows: list[dict] = []
    model_root = output / "models"
    if save_models:
        model_root.mkdir()

    for fold_number, (cv_train, cv_validation) in enumerate(cv_splits, start=1):
        for role, indices in (("train", cv_train), ("validation", cv_validation)):
            for local_index in indices:
                global_index = outer_train[local_index]
                cv_assignment_rows.append(
                    {
                        "fold": fold_number,
                        "role": role,
                        "sample_id": data.sample_ids[global_index],
                        "participant_id": data.groups[global_index],
                    }
                )
        for model_name, estimator in models.items():
            fitted = clone(estimator).fit(X_train.iloc[cv_train], y_train[cv_train])
            prediction = fitted.predict(X_train.iloc[cv_validation])
            error = np.abs(y_train[cv_validation] - prediction)
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold_number,
                    "mae": mean_absolute_error(y_train[cv_validation], prediction),
                    "rmse": mean_squared_error(y_train[cv_validation], prediction) ** 0.5,
                    "r2": r2_score(y_train[cv_validation], prediction),
                    "within_tolerance_accuracy": float(np.mean(error <= tolerance)),
                    "participant_overlap": 0,
                }
            )
            for local_index, predicted in zip(cv_validation, prediction):
                global_index = outer_train[local_index]
                cv_prediction_rows.append(
                    {
                        "model": model_name,
                        "fold": fold_number,
                        "sample_id": data.sample_ids[global_index],
                        "participant_id": data.groups[global_index],
                        "observed": y[global_index],
                        "predicted": float(predicted),
                    }
                )

    for model_name, estimator in models.items():
        fitted = clone(estimator).fit(X_train, y_train)
        prediction = fitted.predict(data.X.iloc[outer_test])
        error = np.abs(y[outer_test] - prediction)
        test_rows.append(
            {
                "model": model_name,
                "mae": mean_absolute_error(y[outer_test], prediction),
                "rmse": mean_squared_error(y[outer_test], prediction) ** 0.5,
                "r2": r2_score(y[outer_test], prediction),
                "within_tolerance_accuracy": float(np.mean(error <= tolerance)),
                "tolerance": float(tolerance),
                "n_test_observations": len(outer_test),
                "n_test_participants": int(np.unique(data.groups[outer_test]).size),
            }
        )
        for global_index, predicted in zip(outer_test, prediction):
            prediction_rows.append(
                {
                    "model": model_name,
                    "sample_id": data.sample_ids[global_index],
                    "participant_id": data.groups[global_index],
                    "observed": y[global_index],
                    "predicted": float(predicted),
                }
            )
        if save_models:
            joblib.dump(
                {
                    "pipeline": fitted,
                    "feature_names": data.feature_names,
                    "target": target_name,
                    "training_sample_ids": data.sample_ids[outer_train].tolist(),
                    "test_sample_ids": data.sample_ids[outer_test].tolist(),
                },
                model_root / f"{model_name}.joblib",
            )

    fold_frame = pd.DataFrame(fold_rows)
    summary = (
        fold_frame.groupby("model", sort=False)[
            ["mae", "rmse", "r2", "within_tolerance_accuracy"]
        ]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output / "training_cv_summary.csv", index=False)
    fold_frame.to_csv(output / "training_cv_fold_metrics.csv", index=False)
    pd.DataFrame(cv_prediction_rows).to_csv(
        output / "training_cv_predictions.csv", index=False
    )
    pd.DataFrame(cv_assignment_rows).drop_duplicates().to_csv(
        output / "training_cv_assignments.csv", index=False
    )
    pd.DataFrame(test_rows).to_csv(output / "test_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "role": "train",
                "sample_id": data.sample_ids[index],
                "participant_id": data.groups[index],
            }
            for index in outer_train
        ]
        + [
            {
                "role": "test",
                "sample_id": data.sample_ids[index],
                "participant_id": data.groups[index],
            }
            for index in outer_test
        ]
    ).to_csv(output / "holdout_assignments.csv", index=False)

    metadata = _metadata(data, "regression", target_name, test_fraction)
    metadata.update(
        {
            "holdout_method": holdout_method,
            "n_training_observations": len(outer_train),
            "n_test_observations": len(outer_test),
            "n_training_participants": int(np.unique(data.groups[outer_train]).size),
            "n_test_participants": int(np.unique(data.groups[outer_test]).size),
            "training_cv_folds_requested": int(training_cv_folds),
            "training_cv_folds_used": len(cv_splits),
            "models": list(models),
            "saved_models": bool(save_models),
            "model_selected_by_training_cv_mae": str(
                summary.loc[summary["mae_mean"].idxmin(), "model"]
            ),
            "tolerance": float(tolerance),
        }
    )
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
