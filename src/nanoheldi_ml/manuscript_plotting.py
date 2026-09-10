"""Plot manuscript classification, regression, and feature-count outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import MODEL_COLORS


def _save(fig, output_stem: str | Path) -> None:
    output = Path(output_stem)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_classification(run_directory: str | Path, output_directory: str | Path) -> None:
    run = Path(run_directory)
    output = Path(output_directory)
    metrics = pd.read_csv(run / "test_metrics.csv")
    curves = pd.read_csv(run / "test_roc_curves.csv")
    confusion = pd.read_csv(run / "test_confusion_matrices.csv")
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))

    folds = pd.read_csv(run / "training_cv_fold_metrics.csv")
    summary = pd.read_csv(run / "training_cv_summary.csv").sort_values(
        "accuracy_mean", ascending=True
    )
    fig_height = max(3.0, 0.32 * len(summary) + 0.8)
    fig, ax = plt.subplots(figsize=(4.0, fig_height))
    positions = np.arange(len(summary))
    colors = [MODEL_COLORS.get(name, "#808080") for name in summary["model"]]
    ax.barh(
        positions,
        summary["accuracy_mean"],
        xerr=summary["accuracy_std"],
        color=colors,
        edgecolor="#333333",
        linewidth=0.5,
        error_kw={"elinewidth": 0.7, "capsize": 2},
    )
    for position, model_name in zip(positions, summary["model"]):
        values = folds.loc[folds["model"].eq(model_name), "accuracy"].to_numpy()
        offsets = np.linspace(-0.09, 0.09, len(values)) if len(values) > 1 else [0]
        ax.scatter(
            values,
            position + np.asarray(offsets),
            s=11,
            facecolor="white",
            edgecolor="#222222",
            linewidth=0.5,
            zorder=3,
        )
    ax.set_yticks(positions, summary["model"])
    ax.set(xlabel="Training cross-validation accuracy", xlim=(0, 1.02))
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, output / "training_cv_model_comparison")

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    grid = np.linspace(0, 1, 201)
    for model_name, model_curves in curves.groupby("model", sort=False):
        class_curves = []
        for _, class_curve in model_curves.groupby("class_label", sort=False):
            ordered = class_curve.sort_values("fpr")
            class_curves.append(np.interp(grid, ordered["fpr"], ordered["tpr"]))
        mean_tpr = np.mean(class_curves, axis=0)
        auc = float(metrics.loc[metrics["model"].eq(model_name), "macro_ovr_auc"].iloc[0])
        ax.plot(
            grid,
            mean_tpr,
            color=MODEL_COLORS.get(model_name, "#808080"),
            linewidth=1.1,
            label=f"{model_name} (AUC={auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, output / "test_roc_curves")

    best_model = metadata["model_selected_by_training_cv_macro_ovr_auc"]
    selected = confusion.loc[confusion["model"].eq(best_model)].copy()
    selected["true_label"] = selected["true_label"].astype(str)
    selected["predicted_label"] = selected["predicted_label"].astype(str)
    labels = list(dict.fromkeys(selected["true_label"].astype(str)))
    matrix = selected.pivot(
        index="true_label", columns="predicted_label", values="count"
    ).reindex(index=labels, columns=labels, fill_value=0).to_numpy(float)
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row in range(len(labels)):
        for column in range(len(labels)):
            ax.text(
                column,
                row,
                f"{normalized[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if normalized[row, column] > 0.55 else "#222222",
            )
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="Predicted label", ylabel="True label", title=best_model)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized accuracy")
    _save(fig, output / "selected_model_confusion_matrix")


def plot_regression(
    run_directory: str | Path,
    output_stem: str | Path,
    model: str = "MLP",
) -> None:
    run = Path(run_directory)
    predictions = pd.read_csv(run / "test_predictions.csv")
    metrics = pd.read_csv(run / "test_metrics.csv")
    selected = predictions.loc[predictions["model"].eq(model)]
    metric = metrics.loc[metrics["model"].eq(model)].iloc[0]
    low = float(min(selected["observed"].min(), selected["predicted"].min()))
    high = float(max(selected["observed"].max(), selected["predicted"].max()))
    grid = np.linspace(low, high, 100)
    tolerance = float(metric["tolerance"])
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.fill_between(grid, grid - tolerance, grid + tolerance, color="#D9EFD3", alpha=0.7)
    ax.plot(grid, grid, color="#444444", linewidth=0.8)
    ax.scatter(
        selected["observed"],
        selected["predicted"],
        s=17,
        facecolors="none",
        edgecolors="#E64B35",
        linewidths=0.75,
    )
    ax.set(
        xlabel="True pain level",
        ylabel="Predicted pain level",
        xlim=(low, high),
        ylim=(low, high),
    )
    ax.text(
        0.03,
        0.97,
        f"R²={metric['r2']:.2f}\n±{tolerance:g} accuracy={metric['within_tolerance_accuracy']:.1%}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, output_stem)


def plot_feature_curve(
    summary_path: str | Path,
    output_stem: str | Path,
    metric: str,
) -> None:
    summary = pd.read_csv(summary_path).sort_values("n_features")
    mean_column = f"{metric}_mean"
    sd_column = f"{metric}_sd"
    if mean_column not in summary or sd_column not in summary:
        raise ValueError(f"The summary does not contain {mean_column} and {sd_column}")
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.errorbar(
        summary["n_features"],
        summary[mean_column],
        yerr=summary[sd_column],
        marker="o",
        markersize=3.5,
        linewidth=0.9,
        capsize=2,
        color="#55AAA7",
    )
    ax.set(xlabel="Number of m/z features", ylabel=metric.replace("_", " ").title())
    if "accuracy" in metric:
        ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, output_stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="plot_type", required=True)
    classification = subparsers.add_parser("classification")
    classification.add_argument("--run", required=True)
    classification.add_argument("--output", required=True)
    regression = subparsers.add_parser("regression")
    regression.add_argument("--run", required=True)
    regression.add_argument("--output", required=True)
    regression.add_argument("--model", default="MLP")
    feature_curve = subparsers.add_parser("feature-curve")
    feature_curve.add_argument("--summary", required=True)
    feature_curve.add_argument("--output", required=True)
    feature_curve.add_argument("--metric", default="accuracy")
    args = parser.parse_args()

    if args.plot_type == "classification":
        plot_classification(args.run, args.output)
    elif args.plot_type == "regression":
        plot_regression(args.run, args.output, args.model)
    else:
        plot_feature_curve(args.summary, args.output, args.metric)


if __name__ == "__main__":
    main()
