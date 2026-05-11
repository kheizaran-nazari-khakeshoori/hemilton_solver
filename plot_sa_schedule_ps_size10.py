"""Plot approximation quality metrics for SA schedule grid search (size 10).

This script reads the CSV produced by sa_schedule_grid_search_size10.py and
creates heatmaps for:

- average relative error Ps = (best_cost - ground_cost) / ground_cost
- probability of reaching within 1% of optimal
- probability of reaching within 10% of optimal

These plots show solution quality instead of exact success probability.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_CSV = Path("sa_schedule_grid_size10.csv")
PLOTS_DIR = Path("plots")


def _plot_heatmap(matrix: np.ndarray, x_vals: np.ndarray, y_vals: np.ndarray, title: str, cbar_label: str, out_filename: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
        cmap="viridis",
    )

    ax.set_xlabel("Number of swaps N")
    ax.set_ylabel("Final inverse temperature beta")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    PLOTS_DIR.mkdir(exist_ok=True)
    out_path = PLOTS_DIR / out_filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not INPUT_CSV.is_file():
        raise SystemExit(
            f"Input CSV '{INPUT_CSV}' not found. Run sa_schedule_grid_search_size10.py first."
        )

    df = pd.read_csv(INPUT_CSV)
    required_cols = {
        "beta_final",
        "steps",
        "avg_ps",
        "prob_ps_le_0.01",
        "prob_ps_le_0.1",
    }
    if not required_cols.issubset(df.columns):
        raise SystemExit(
            f"CSV must contain columns {sorted(required_cols)}. "
            "Did the grid search script run with Ps metrics enabled?"
        )

    grouped = (
        df.groupby(["beta_final", "steps"], as_index=False)
        .agg(
            avg_ps=("avg_ps", "mean"),
            prob_ps_le_0_01=("prob_ps_le_0.01", "mean"),
            prob_ps_le_0_1=("prob_ps_le_0.1", "mean"),
        )
    )

    beta_vals = np.sort(grouped["beta_final"].unique())
    step_vals = np.sort(grouped["steps"].unique())

    beta_to_idx = {b: i for i, b in enumerate(beta_vals)}
    step_to_idx = {s: j for j, s in enumerate(step_vals)}

    avg_ps_matrix = np.full((len(beta_vals), len(step_vals)), np.nan, dtype=float)
    prob_ps01_matrix = np.full((len(beta_vals), len(step_vals)), np.nan, dtype=float)
    prob_ps10_matrix = np.full((len(beta_vals), len(step_vals)), np.nan, dtype=float)

    for _, row in grouped.iterrows():
        bi = beta_to_idx[float(row["beta_final"])]
        sj = step_to_idx[int(row["steps"])]
        avg_ps_matrix[bi, sj] = float(row["avg_ps"])
        prob_ps01_matrix[bi, sj] = float(row["prob_ps_le_0_01"])
        prob_ps10_matrix[bi, sj] = float(row["prob_ps_le_0_1"])

    _plot_heatmap(
        avg_ps_matrix,
        step_vals,
        beta_vals,
        "Average relative error Ps for size-10 SA schedules",
        "average Ps",
        "sa_schedule_avg_ps_heatmap_size10.png",
    )

    _plot_heatmap(
        prob_ps01_matrix,
        step_vals,
        beta_vals,
        "Probability of Ps <= 0.01 for size-10 SA schedules",
        "probability",
        "sa_schedule_prob_ps01_heatmap_size10.png",
    )

    _plot_heatmap(
        prob_ps10_matrix,
        step_vals,
        beta_vals,
        "Probability of Ps <= 0.1 for size-10 SA schedules",
        "probability",
        "sa_schedule_prob_ps10_heatmap_size10.png",
    )

    print(f"Saved Ps heatmaps to '{PLOTS_DIR}' directory.")


if __name__ == "__main__":
    main()
