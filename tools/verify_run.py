#!/usr/bin/env python3
"""Verify a saved run without accessing the original participant-level table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    metadata_path = run / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("Missing run file: run_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("protocol") == "manuscript_subject_holdout":
        verify_manuscript_run(run, metadata)
        return

    required = {
        "fold_metrics.csv",
        "summary_metrics.csv",
        "predictions.csv",
        "split_assignments.csv",
        "selected_features.csv",
        "run_metadata.json",
    }
    missing = sorted(name for name in required if not (run / name).is_file())
    if missing:
        raise FileNotFoundError(f"Missing run files: {missing}")

    folds = pd.read_csv(run / "fold_metrics.csv")
    assignments = pd.read_csv(run / "split_assignments.csv")
    if not folds["participant_overlap"].eq(0).all():
        raise RuntimeError("Non-zero participant overlap recorded")
    for fold in assignments["fold"].unique():
        subset = assignments[assignments["fold"].eq(fold)]
        train = set(subset.loc[subset["role"].eq("train"), "participant_id"].astype(str))
        validation = set(
            subset.loc[subset["role"].eq("validation"), "participant_id"].astype(str)
        )
        if train.intersection(validation):
            raise RuntimeError(f"Participant leakage detected in fold {fold}")

    artifacts = sorted((run / "models").glob("*/fold_*.joblib"))
    expected = len(metadata["models"]) * int(metadata["n_splits_used"])
    if metadata.get("saved_fold_models") and len(artifacts) != expected:
        raise RuntimeError(f"Expected {expected} fold artifacts, found {len(artifacts)}")
    for artifact in artifacts:
        bundle = joblib.load(artifact)
        if "pipeline" not in bundle or "feature_names" not in bundle:
            raise RuntimeError(f"Incomplete model bundle: {artifact}")
    print(
        f"PASS: {run} contains {len(metadata['models'])} models, "
        f"{metadata['n_splits_used']} grouped folds, and {len(artifacts)} loadable fold artifacts"
    )


def verify_manuscript_run(run: Path, metadata: dict) -> None:
    task = metadata["task"]
    required = {
        "training_cv_fold_metrics.csv",
        "training_cv_summary.csv",
        "training_cv_predictions.csv",
        "training_cv_assignments.csv",
        "test_metrics.csv",
        "test_predictions.csv",
        "holdout_assignments.csv",
        "run_metadata.json",
    }
    if task == "classification":
        required.update({"test_roc_curves.csv", "test_confusion_matrices.csv"})
    missing = sorted(name for name in required if not (run / name).is_file())
    if missing:
        raise FileNotFoundError(f"Missing run files: {missing}")

    folds = pd.read_csv(run / "training_cv_fold_metrics.csv")
    if not folds["participant_overlap"].eq(0).all():
        raise RuntimeError("Non-zero participant overlap recorded")
    holdout = pd.read_csv(run / "holdout_assignments.csv")
    training_groups = set(
        holdout.loc[holdout["role"].eq("train"), "participant_id"].astype(str)
    )
    test_groups = set(
        holdout.loc[holdout["role"].eq("test"), "participant_id"].astype(str)
    )
    if training_groups.intersection(test_groups):
        raise RuntimeError("Participant overlap detected between training and test sets")

    assignments = pd.read_csv(run / "training_cv_assignments.csv")
    for fold in assignments["fold"].unique():
        subset = assignments[assignments["fold"].eq(fold)]
        train = set(
            subset.loc[subset["role"].eq("train"), "participant_id"].astype(str)
        )
        validation = set(
            subset.loc[subset["role"].eq("validation"), "participant_id"].astype(str)
        )
        if train.intersection(validation):
            raise RuntimeError(f"Participant overlap detected in training CV fold {fold}")

    artifacts = sorted((run / "models").glob("*.joblib"))
    expected = len(metadata["models"]) if metadata.get("saved_models") else 0
    if len(artifacts) != expected:
        raise RuntimeError(f"Expected {expected} model artifacts, found {len(artifacts)}")
    for artifact in artifacts:
        bundle = joblib.load(artifact)
        required_keys = {"pipeline", "feature_names", "training_sample_ids", "test_sample_ids"}
        if not required_keys.issubset(bundle):
            raise RuntimeError(f"Incomplete model bundle: {artifact}")
        if set(bundle["training_sample_ids"]).intersection(bundle["test_sample_ids"]):
            raise RuntimeError(f"Overlapping sample IDs in model bundle: {artifact}")
    print(
        f"PASS: {run} contains {len(metadata['models'])} models, "
        f"{metadata['training_cv_folds_used']} grouped training folds, "
        f"an isolated test partition, and {len(artifacts)} loadable model artifacts"
    )


if __name__ == "__main__":
    main()
