"""SHAP and integrated-gradients analysis for saved manuscript MLP pipelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import AnalysisData, load_analysis_table
from .evaluation import prepare_output_directory


def _rows_for_ids(data: AnalysisData, sample_ids: list[str]) -> np.ndarray:
    row_by_id = {str(sample_id): row for row, sample_id in enumerate(data.sample_ids)}
    missing = [sample_id for sample_id in sample_ids if str(sample_id) not in row_by_id]
    if missing:
        raise ValueError(f"Saved sample IDs are absent from the input table: {missing[:3]}")
    return np.asarray([row_by_id[str(sample_id)] for sample_id in sample_ids], dtype=int)


def _evenly_spaced_rows(rows: np.ndarray, maximum: int) -> np.ndarray:
    if maximum < 1:
        raise ValueError("Sample limits must be positive")
    if len(rows) <= maximum:
        return rows
    positions = np.linspace(0, len(rows) - 1, maximum, dtype=int)
    return rows[positions]


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    maximum = float(values.max(initial=0.0))
    return values / maximum if maximum > 0 else np.zeros_like(values)


def _integrated_gradients(
    pipeline,
    X_explain: pd.DataFrame,
    X_reference: pd.DataFrame,
    steps: int,
    epsilon: float,
) -> np.ndarray:
    if steps < 2:
        raise ValueError("Integrated gradients requires at least two integration steps")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    transformer = pipeline[:-1]
    model = pipeline.named_steps["model"]
    transformed = np.asarray(transformer.transform(X_explain), dtype=float)
    reference = np.asarray(transformer.transform(X_reference), dtype=float)
    if transformed.shape[1] != X_explain.shape[1]:
        raise ValueError("Interpretability requires a pipeline without feature removal")
    baseline = np.median(reference, axis=0)
    difference = transformed - baseline
    n_samples, n_features = transformed.shape
    feature_indices = np.tile(np.arange(n_features), n_samples)
    row_indices = np.arange(n_samples * n_features)
    accumulated = np.zeros_like(transformed)

    if hasattr(model, "predict_proba"):
        target_columns = np.argmax(model.predict_proba(transformed), axis=1)
        repeated_targets = np.repeat(target_columns, n_features)

        def selected_output(values: np.ndarray) -> np.ndarray:
            probabilities = np.asarray(model.predict_proba(values), dtype=float)
            return probabilities[row_indices, repeated_targets]

    else:

        def selected_output(values: np.ndarray) -> np.ndarray:
            return np.asarray(model.predict(values), dtype=float)

    for alpha in np.linspace(0.0, 1.0, steps):
        path = baseline + alpha * difference
        plus = np.repeat(path, n_features, axis=0)
        minus = plus.copy()
        plus[row_indices, feature_indices] += epsilon
        minus[row_indices, feature_indices] -= epsilon
        gradients = (selected_output(plus) - selected_output(minus)) / (2.0 * epsilon)
        accumulated += gradients.reshape(n_samples, n_features)
    return difference * accumulated / steps


def explain_saved_mlp(
    data: AnalysisData,
    model_path: str | Path,
    output: str | Path,
    max_test_samples: int = 100,
    max_background_samples: int = 20,
    integrated_gradient_steps: int = 32,
    epsilon: float = 1e-4,
) -> Path:
    """Explain a saved training-only MLP on its designated independent test rows."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "Install the optional interpretability dependencies with '.[explain]'"
        ) from exc

    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    feature_names = [str(item) for item in artifact["feature_names"]]
    if feature_names != list(data.X.columns.astype(str)):
        raise ValueError("The model artifact and input table have different feature columns")
    model = pipeline.named_steps.get("model")
    if model is None or not model.__class__.__name__.startswith("MLP"):
        raise ValueError("This command expects a saved MLP pipeline")
    output = prepare_output_directory(output)

    training_rows = _rows_for_ids(data, artifact["training_sample_ids"])
    test_rows = _rows_for_ids(data, artifact["test_sample_ids"])
    background_rows = _evenly_spaced_rows(training_rows, max_background_samples)
    explain_rows = _evenly_spaced_rows(test_rows, max_test_samples)
    X_background = data.X.iloc[background_rows]
    X_explain = data.X.iloc[explain_rows]

    def predict(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=feature_names)
        if hasattr(pipeline, "predict_proba"):
            return np.asarray(pipeline.predict_proba(frame), dtype=float)
        return np.asarray(pipeline.predict(frame), dtype=float)

    explainer = shap.Explainer(
        predict,
        X_background,
        algorithm="permutation",
        feature_names=feature_names,
    )
    explanation = explainer(X_explain, max_evals=2 * len(feature_names) + 1)
    shap_values = np.asarray(explanation.values, dtype=float)
    if shap_values.ndim == 3:
        shap_importance = np.mean(np.abs(shap_values), axis=(0, 2))
        shap_per_sample = np.mean(np.abs(shap_values), axis=2)
    elif shap_values.ndim == 2:
        shap_importance = np.mean(np.abs(shap_values), axis=0)
        shap_per_sample = np.abs(shap_values)
    else:
        raise ValueError(f"Unexpected SHAP value shape: {shap_values.shape}")

    integrated_values = _integrated_gradients(
        pipeline,
        X_explain,
        X_background,
        steps=integrated_gradient_steps,
        epsilon=epsilon,
    )
    integrated_importance = np.mean(np.abs(integrated_values), axis=0)
    combined = (_normalize(shap_importance) + _normalize(integrated_importance)) / 2.0

    sample_column = data.sample_ids[explain_rows]
    pd.DataFrame(shap_per_sample, columns=feature_names).assign(
        sample_id=sample_column
    ).set_index("sample_id").to_csv(output / "mean_absolute_shap_by_sample.csv")
    pd.DataFrame(integrated_values, columns=feature_names).assign(
        sample_id=sample_column
    ).set_index("sample_id").to_csv(output / "integrated_gradients.csv")
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap": shap_importance,
            "mean_absolute_integrated_gradient": integrated_importance,
            "combined_normalized_importance": combined,
        }
    ).sort_values("combined_normalized_importance", ascending=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    metadata = {
        "method": "permutation SHAP and finite-difference integrated gradients",
        "explanation_scope": "designated independent test partition",
        "model_path_name": Path(model_path).name,
        "n_test_samples_explained": len(explain_rows),
        "n_training_background_samples": len(background_rows),
        "integrated_gradient_steps": int(integrated_gradient_steps),
        "finite_difference_epsilon": float(epsilon),
        "input_sha256": data.input_sha256,
        "feature_importance_policy": (
            "mean absolute attribution; equal-weight mean after separate max normalization"
        ),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-test-samples", type=int, default=100)
    parser.add_argument("--max-background-samples", type=int, default=20)
    parser.add_argument("--integrated-gradient-steps", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=1e-4)
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
    output = explain_saved_mlp(
        data,
        args.model,
        args.output,
        max_test_samples=args.max_test_samples,
        max_background_samples=args.max_background_samples,
        integrated_gradient_steps=args.integrated_gradient_steps,
        epsilon=args.epsilon,
    )
    print(f"Saved interpretability outputs to {output}")


if __name__ == "__main__":
    main()
