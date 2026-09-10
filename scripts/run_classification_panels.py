#!/usr/bin/env python3
"""Run multiple classification targets and save every fold-specific model artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from nanoheldi_ml.data import load_analysis_table
from nanoheldi_ml.evaluation import evaluate_classification
from nanoheldi_ml.models import classification_models


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--features", type=int, default=50)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--participant-column", default="participant_id")
    parser.add_argument("--sample-column", default="sample_id")
    parser.add_argument("--feature-prefix", default="mz_")
    parser.add_argument("--include-xgboost", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError(f"Output path is not an empty directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    for target in args.targets:
        data = load_analysis_table(
            args.input,
            target_column=target,
            participant_column=args.participant_column,
            sample_column=args.sample_column,
            feature_prefix=args.feature_prefix,
        )
        n_features = min(max(1, args.features), data.X.shape[1])
        models = classification_models(n_features, include_xgboost=args.include_xgboost)
        destination = args.output / safe_name(target)
        evaluate_classification(
            data,
            models,
            destination,
            target_name=target,
            n_splits=args.folds,
            save_models=True,
        )
        print(f"Completed {target}: {destination}")


if __name__ == "__main__":
    main()
