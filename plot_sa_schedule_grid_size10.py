"""Plot 2D heatmaps for SA schedule grid search (size 10).

This script reads the CSV produced by sa_schedule_grid_search_size10.py
and generates coloured plots of the metrics as functions of
(final beta, total number of swaps).

For each of the three selected instances it produces, at least, a
heatmap of the probability of solving:

    x-axis: total swaps N
    y-axis: final beta
    colour: probability of reaching the ground state
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_CSV = Path("sa_schedule_grid_size10.csv")
PLOTS_DIR = Path("plots")


def _plot_instance_heatmap(df_inst: pd.DataFrame, instance_id: int) -> None:
    # Pivot the table so that rows are beta, columns are steps.
    beta_vals = np.sort(df_inst["beta_final"].unique())
    step_vals = np.sort(df_inst["steps"].unique())

    beta_to_idx = {b: i for i, b in enumerate(beta_vals)}
    step_to_idx = {s: j for j, s in enumerate(step_vals)}

    prob_matrix = np.zeros((len(beta_vals), len(step_vals)), dtype=float)

    for _, row in df_inst.iterrows():
        bi = beta_to_idx[row["beta_final"]]
        sj = step_to_idx[row["steps"]]
        prob_matrix[bi, sj] = row["prob_solved"]

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        prob_matrix,
        origin="lower",
        aspect="auto",
        extent=[step_vals[0], step_vals[-1], beta_vals[0], beta_vals[-1]],
        cmap="viridis",
    )

    ax.set_xlabel("Number of swaps N")
    ax.set_ylabel("Final inverse temperature beta")
    ax.set_title(f"SA solving probability (size=10, instance={instance_id})")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Probability of solving")

    PLOTS_DIR.mkdir(exist_ok=True)
    out_path = PLOTS_DIR / f"sa_schedule_prob_heatmap_size10_inst{instance_id}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not INPUT_CSV.is_file():
        raise SystemExit(
            f"Input CSV '{INPUT_CSV}' not found. Run sa_schedule_grid_search_size10.py first."
        )

    df = pd.read_csv(INPUT_CSV)

    for inst_id, df_inst in df.groupby("instance"):
        _plot_instance_heatmap(df_inst, int(inst_id))

    print(f"Saved schedule heatmaps to '{PLOTS_DIR}' directory.")


if __name__ == "__main__":
    main()
