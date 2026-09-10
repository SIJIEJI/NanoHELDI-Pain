from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nanoheldi_ml.data import load_analysis_table
from nanoheldi_ml.feature_analysis import classification_feature_curve
from nanoheldi_ml.interpretability import _integrated_gradients
from nanoheldi_ml.manuscript_evaluation import (
    evaluate_manuscript_classification,
    evaluate_manuscript_regression,
)
from nanoheldi_ml.manuscript_plotting import plot_classification
from nanoheldi_ml.models import (
    manuscript_classification_models,
    manuscript_regression_models,
)


def _write_table(path: Path, regression: bool = False) -> Path:
    generator = np.random.default_rng(123)
    n_participants = 20
    participant = np.repeat(np.arange(n_participants), 2)
    features = generator.normal(size=(len(participant), 12))
    features[:, 0] += (participant % 2) * 1.5
    table = pd.DataFrame(features, columns=[f"mz_{index}" for index in range(12)])
    table.loc[0, "mz_2"] = np.nan
    table.insert(0, "participant_id", [f"P{item:02d}" for item in participant])
    table.insert(0, "sample_id", [f"S{item:03d}" for item in range(len(table))])
    if regression:
        table.insert(2, "pain_score", 4.0 + features[:, 0] - features[:, 1] * 0.5)
    else:
        table.insert(2, "target", (participant % 2).astype(int))
    table.to_csv(path, index=False)
    return path


def _assert_disjoint(assignments: pd.DataFrame, left: str, right: str) -> None:
    left_groups = set(
        assignments.loc[assignments["role"].eq(left), "participant_id"].astype(str)
    )
    right_groups = set(
        assignments.loc[assignments["role"].eq(right), "participant_id"].astype(str)
    )
    assert left_groups.isdisjoint(right_groups)


def test_manuscript_model_set_contains_named_families() -> None:
    models = manuscript_classification_models(include_xgboost=False)
    assert set(models) == {
        "Random",
        "NB",
        "DT",
        "LR",
        "GB",
        "SVM",
        "RF",
        "AdaBoost",
        "KNN",
        "MLP",
    }


def test_manuscript_classification_has_isolated_test_set_and_artifacts(
    tmp_path: Path,
) -> None:
    data = load_analysis_table(_write_table(tmp_path / "classification.csv"), "target")
    models = {"LR": manuscript_classification_models(False)["LR"]}
    output = evaluate_manuscript_classification(
        data,
        models,
        tmp_path / "classification_run",
        "target",
        training_cv_folds=3,
    )
    holdout = pd.read_csv(output / "holdout_assignments.csv")
    _assert_disjoint(holdout, "train", "test")
    folds = pd.read_csv(output / "training_cv_fold_metrics.csv")
    assert folds["participant_overlap"].eq(0).all()
    assert len(folds) == 3
    assert (output / "test_roc_curves.csv").is_file()
    assert (output / "test_confusion_matrices.csv").is_file()
    artifact = joblib.load(output / "models" / "LR.joblib")
    assert set(artifact["training_sample_ids"]).isdisjoint(artifact["test_sample_ids"])

    figure_directory = tmp_path / "classification_figures"
    plot_classification(output, figure_directory)
    assert (figure_directory / "training_cv_model_comparison.png").is_file()
    assert (figure_directory / "test_roc_curves.pdf").is_file()
    assert (figure_directory / "selected_model_confusion_matrix.svg").is_file()


def test_feature_ranking_is_fold_local_and_handles_missing_values(tmp_path: Path) -> None:
    data = load_analysis_table(_write_table(tmp_path / "classification.csv"), "target")
    output = classification_feature_curve(
        data,
        tmp_path / "feature_curve",
        counts=[3, 6],
        folds=3,
    )
    metrics = pd.read_csv(output / "feature_curve_fold_metrics.csv")
    selected = pd.read_csv(output / "selected_features.csv")
    assert set(metrics["n_features"]) == {3, 6}
    assert metrics.groupby("n_features").size().eq(3).all()
    assert selected.groupby(["fold", "n_features"]).size().to_dict() == {
        (fold, count): count for fold in (1, 2, 3) for count in (3, 6)
    }


def test_manuscript_regression_saves_training_and_test_predictions(tmp_path: Path) -> None:
    data = load_analysis_table(
        _write_table(tmp_path / "regression.csv", regression=True), "pain_score"
    )
    models = {"Mean": manuscript_regression_models()["Mean"]}
    output = evaluate_manuscript_regression(
        data,
        models,
        tmp_path / "regression_run",
        "pain_score",
        training_cv_folds=3,
    )
    holdout = pd.read_csv(output / "holdout_assignments.csv")
    _assert_disjoint(holdout, "train", "test")
    assert (output / "training_cv_predictions.csv").is_file()
    assert (output / "training_cv_assignments.csv").is_file()
    assert (output / "test_predictions.csv").is_file()


def test_integrated_gradients_uses_the_fitted_pipeline(tmp_path: Path) -> None:
    data = load_analysis_table(_write_table(tmp_path / "classification.csv"), "target")
    pipeline = manuscript_classification_models(False)["MLP"]
    pipeline.fit(data.X.iloc[4:], data.y.iloc[4:])
    values = _integrated_gradients(
        pipeline,
        data.X.iloc[:4],
        data.X.iloc[4:],
        steps=3,
        epsilon=1e-4,
    )
    assert values.shape == (4, data.X.shape[1])
    assert np.isfinite(values).all()
