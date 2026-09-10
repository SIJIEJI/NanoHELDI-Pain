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

    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
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


if __name__ == "__main__":
    main()
