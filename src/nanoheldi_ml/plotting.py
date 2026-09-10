"""Publication-oriented model comparison plot from frozen fold metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_COLORS = {
    "Random": "#B7B7B7",
    "Mean": "#B7B7B7",
    "NB": "#9A5EB5",
    "DT": "#F2CA55",
    "LR": "#55AAA7",
    "GB": "#E7635F",
    "SVM": "#3E6DA5",
    "RF": "#64A85D",
    "AdaBoost": "#8C75B8",
    "KNN": "#C88A35",
    "XGBoost": "#9A685B",
    "MLP": "#F4A52F",
}


def metric_for(summary: pd.DataFrame) -> str:
    for candidate in ("balanced_accuracy", "accuracy", "within_tolerance_accuracy", "r2"):
        if f"{candidate}_mean" in summary.columns:
            return candidate
    raise ValueError("No supported metric found in summary table")


def plot(summary_path: str | Path, folds_path: str | Path, output_stem: str | Path) -> None:
    summary = pd.read_csv(summary_path)
    folds = pd.read_csv(folds_path)
    metric = metric_for(summary)
    summary = summary.sort_values(f"{metric}_mean", ascending=True).reset_index(drop=True)

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(3.5, max(2.5, 0.35 * len(summary) + 0.8)))
    y = np.arange(len(summary))
    colors = [MODEL_COLORS.get(name, "#808080") for name in summary["model"]]
    ax.barh(
        y,
        summary[f"{metric}_mean"],
        xerr=summary[f"{metric}_sd"],
        color=colors,
        edgecolor="#333333",
        linewidth=0.55,
        alpha=0.88,
        capsize=2,
    )
    for position, row in summary.iterrows():
        scores = folds.loc[folds["model"].eq(row["model"]), metric].to_numpy(float)
        jitter = np.linspace(-0.13, 0.13, len(scores))
        ax.scatter(
            scores,
            position + jitter,
            s=12,
            color=MODEL_COLORS.get(row["model"], "#808080"),
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    ax.set_yticks(y, summary["model"])
    ax.set_xlabel(metric.replace("_", " ").title())
    if metric != "r2":
        ax.set_xlim(0, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#D0D0D0", linewidth=0.45, alpha=0.6)
    n_participants = int(summary["n_participants"].iloc[0])
    n_folds = int(summary["n_folds"].iloc[0])
    ax.text(
        0.0,
        1.015,
        f"n={n_participants} participants; points={n_folds} grouped CV folds; bars=mean ± SD",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color="#444444",
    )
    fig.tight_layout()

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        output_stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(args.summary, args.folds, args.output)
    print(f"Saved plot files with stem {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
