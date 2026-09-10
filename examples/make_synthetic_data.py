#!/usr/bin/env python3
"""Create non-biological synthetic tables for installation and CI smoke tests."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def feature_frame(n_participants: int = 24, n_features: int = 20) -> pd.DataFrame:
    state = np.random.default_rng(42)
    participant = np.repeat(np.arange(n_participants), 2)
    visit = np.tile([1, 2], n_participants)
    latent = (participant % 2).astype(float)
    matrix = state.normal(size=(len(participant), n_features))
    matrix[:, :4] += latent[:, None] * 1.2
    frame = pd.DataFrame(matrix, columns=[f"mz_{100 + index:.2f}" for index in range(n_features)])
    frame.insert(0, "participant_id", [f"P{item:03d}" for item in participant])
    frame.insert(0, "sample_id", [f"P{p:03d}_V{v}" for p, v in zip(participant, visit)])
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory for generated CSV files (default: examples/data).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace synthetic CSV files that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    classification_path = output / "synthetic_classification.csv"
    regression_path = output / "synthetic_regression.csv"
    existing = [path for path in (classification_path, regression_path) if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing file(s): {names}")

    classification = feature_frame()
    classification.insert(2, "target", (np.arange(24).repeat(2) % 2).astype(int))
    classification.to_csv(classification_path, index=False)

    regression = feature_frame()
    signal = regression.filter(like="mz_").iloc[:, :3].sum(axis=1)
    regression.insert(2, "pain_score", 4.0 + signal)
    regression.to_csv(regression_path, index=False)
    print(f"Saved synthetic smoke-test data to {output}")


if __name__ == "__main__":
    main()
