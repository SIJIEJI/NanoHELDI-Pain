#!/usr/bin/env python3
"""Run manuscript feature-count curves with fold-local feature ranking."""

from __future__ import annotations

import argparse

from nanoheldi_ml.data import load_analysis_table
from nanoheldi_ml.feature_analysis import (
    classification_feature_curve,
    regression_feature_curve,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["classification", "regression"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--counts", nargs="+", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument("--participant-column", default="participant_id")
    parser.add_argument("--sample-column", default="sample_id")
    parser.add_argument("--feature-prefix", default="mz_")
    args = parser.parse_args()

    data = load_analysis_table(
        args.input,
        target_column=args.target,
        participant_column=args.participant_column,
        sample_column=args.sample_column,
        feature_prefix=args.feature_prefix,
    )
    if args.task == "classification":
        result = classification_feature_curve(
            data, args.output, counts=args.counts, folds=args.folds
        )
    else:
        result = regression_feature_curve(
            data,
            args.output,
            counts=args.counts,
            folds=args.folds,
            tolerance=args.tolerance,
        )
    print(f"Saved feature-count analysis to {result}")


if __name__ == "__main__":
    main()
