from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from nanoheldi_ml.data import assert_no_group_overlap, load_analysis_table
from nanoheldi_ml.evaluation import evaluate_classification, evaluate_regression
from nanoheldi_ml.models import classification_models, regression_models


def write_table(path: Path, regression: bool = False) -> Path:
    n_participants = 12
    participant = np.repeat(np.arange(n_participants), 2)
    visit = np.tile([1, 2], n_participants)
    base = np.arange(len(participant) * 8, dtype=float).reshape(len(participant), 8) / 100.0
    table = pd.DataFrame(base, columns=[f"mz_{index}" for index in range(8)])
    table.insert(0, "participant_id", [f"P{item:02d}" for item in participant])
    table.insert(0, "sample_id", [f"P{p:02d}_V{v}" for p, v in zip(participant, visit)])
    if regression:
        table.insert(2, "pain_score", 1.0 + base[:, 0] * 3.0)
    else:
        table.insert(2, "target", (participant % 2).astype(int))
    table.to_csv(path, index=False)
    return path


def test_overlap_guard() -> None:
    with pytest.raises(RuntimeError):
        assert_no_group_overlap(np.array(["P01", "P02"]), np.array(["P02", "P03"]))


def test_classification_is_grouped_and_saves_fitted_pipeline(tmp_path: Path) -> None:
    data = load_analysis_table(write_table(tmp_path / "classification.csv"), "target")
    models = {"LR": classification_models(4)["LR"]}
    output = evaluate_classification(data, models, tmp_path / "classification_run", "target", 3)
    metrics = pd.read_csv(output / "fold_metrics.csv")
    assignments = pd.read_csv(output / "split_assignments.csv")
    assert metrics["participant_overlap"].eq(0).all()
    for fold in assignments["fold"].unique():
        subset = assignments[assignments["fold"].eq(fold)]
        train = set(subset.loc[subset["role"].eq("train"), "participant_id"])
        validation = set(subset.loc[subset["role"].eq("validation"), "participant_id"])
        assert train.isdisjoint(validation)
    artifact = joblib.load(output / "models" / "LR" / "fold_1.joblib")
    assert hasattr(artifact["pipeline"].named_steps["scaler"], "mean_")
    assert len(artifact["pipeline"].named_steps["select"].get_support(indices=True)) == 4


def test_regression_uses_conventional_metrics(tmp_path: Path) -> None:
    data = load_analysis_table(write_table(tmp_path / "regression.csv", True), "pain_score")
    models = {"Mean": regression_models(4)["Mean"]}
    output = evaluate_regression(data, models, tmp_path / "regression_run", "pain_score", 3)
    metrics = pd.read_csv(output / "fold_metrics.csv")
    assert {"mae", "rmse", "r2", "within_tolerance_accuracy"}.issubset(metrics.columns)
    assert metrics["participant_overlap"].eq(0).all()


def test_existing_nonempty_output_is_never_overwritten(tmp_path: Path) -> None:
    data = load_analysis_table(write_table(tmp_path / "classification.csv"), "target")
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError):
        evaluate_classification(
            data, {"LR": classification_models(4)["LR"]}, output, "target", 3
        )
    assert (output / "keep.txt").read_text(encoding="utf-8") == "do not overwrite"
