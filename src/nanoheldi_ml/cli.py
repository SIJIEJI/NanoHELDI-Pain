"""Command-line entry point for publication-oriented grouped evaluation."""

from __future__ import annotations

import argparse

from .config import DEFAULT_N_FEATURES, DEFAULT_N_SPLITS
from .data import load_analysis_table
from .evaluation import evaluate_classification, evaluate_regression, select_models
from .manuscript_evaluation import (
    evaluate_manuscript_classification,
    evaluate_manuscript_regression,
)
from .models import (
    classification_models,
    manuscript_classification_models,
    manuscript_regression_models,
    regression_models,
)


def common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Deidentified row-wise CSV table")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--participant-column", default="participant_id")
    parser.add_argument("--sample-column", default="sample_id")
    parser.add_argument("--feature-prefix", default="mz_")
    parser.add_argument("--features", type=int, default=DEFAULT_N_FEATURES)
    parser.add_argument("--folds", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", nargs="*", help="Optional subset of model names")
    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="Do not save fitted fold pipelines and full-data deployment candidates",
    )


def manuscript_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Deidentified row-wise CSV table")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--participant-column", default="participant_id")
    parser.add_argument("--sample-column", default="sample_id")
    parser.add_argument("--feature-prefix", default="mz_")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--training-cv-folds", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", nargs="*", help="Optional subset of model names")
    parser.add_argument("--no-save-models", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leakage-resistant participant-grouped model evaluation"
    )
    subparsers = parser.add_subparsers(dest="task", required=True)

    classification = subparsers.add_parser("classification")
    common_arguments(classification)
    classification.add_argument("--include-xgboost", action="store_true")

    regression = subparsers.add_parser("regression")
    common_arguments(regression)
    regression.add_argument("--tolerance", type=float, default=2.0)

    manuscript_classification = subparsers.add_parser(
        "manuscript-classification",
        help="Subject-level 80:20 holdout with training-only 10-fold CV and ROC/AUC",
    )
    manuscript_arguments(manuscript_classification)
    manuscript_classification.add_argument("--include-xgboost", action="store_true")

    manuscript_regression = subparsers.add_parser(
        "manuscript-regression",
        help="Subject-level 80:20 holdout with training-only grouped CV",
    )
    manuscript_arguments(manuscript_regression)
    manuscript_regression.add_argument("--tolerance", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = load_analysis_table(
        args.input,
        target_column=args.target,
        participant_column=args.participant_column,
        sample_column=args.sample_column,
        feature_prefix=args.feature_prefix,
    )
    save_models = not args.no_save_models

    if args.task == "classification":
        n_features = min(max(1, args.features), data.X.shape[1])
        models = classification_models(n_features, include_xgboost=args.include_xgboost)
        models = select_models(models, args.models)
        output = evaluate_classification(
            data,
            models,
            args.output,
            target_name=args.target,
            n_splits=args.folds,
            save_models=save_models,
        )
    elif args.task == "regression":
        n_features = min(max(1, args.features), data.X.shape[1])
        models = select_models(regression_models(n_features), args.models)
        output = evaluate_regression(
            data,
            models,
            args.output,
            target_name=args.target,
            n_splits=args.folds,
            tolerance=args.tolerance,
            save_models=save_models,
        )
    elif args.task == "manuscript-classification":
        models = manuscript_classification_models(
            include_xgboost=args.include_xgboost
        )
        models = select_models(models, args.models)
        output = evaluate_manuscript_classification(
            data,
            models,
            args.output,
            target_name=args.target,
            test_fraction=args.test_fraction,
            training_cv_folds=args.training_cv_folds,
            save_models=save_models,
        )
    else:
        models = select_models(manuscript_regression_models(), args.models)
        output = evaluate_manuscript_regression(
            data,
            models,
            args.output,
            target_name=args.target,
            test_fraction=args.test_fraction,
            training_cv_folds=args.training_cv_folds,
            tolerance=args.tolerance,
            save_models=save_models,
        )
    print(f"Saved validated outputs to {output}")


if __name__ == "__main__":
    main()
