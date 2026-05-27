"""Plot steps-to-solution (STS) from SA schedule grid search for size 10.

This script reads `sa_schedule_grid_size10.csv` produced by
`sa_schedule_grid_search_size10.py`, aggregates the probability of
solving over the three representative instances for each
(beta_final, steps) pair, computes the steps-to-solution (STS) metric
for 99% confidence, and visualises it as a heatmap.

STS is defined as:

    STS = N_steps * ln(1 - 0.99) / ln(1 - p),

where N_steps is the number of Metropolis swaps per run and p is the
probability of success for that schedule.

This script computes STS for three success definitions:
- exact solution probability (`prob_solved`)
- probability of reaching within 1% of optimal (`prob_ps_le_0.01`)
- probability of reaching within 10% of optimal (`prob_ps_le_0.1`)

Lower STS values correspond to more efficient schedules in terms of
steps required to reach a given confidence level.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The input CSV contains schedule results for several size-10 instances.
INPUT_CSV = Path("sa_schedule_grid_size10.csv")
PLOTS_DIR = Path("plots")

TARGET_CONFIDENCE = 0.99


def _compute_sts_matrix(grouped: pd.DataFrame, p_column: str) -> np.ndarray:
    beta_vals = np.sort(grouped["beta_final"].unique())
    step_vals = np.sort(grouped["steps"].unique())

    beta_to_idx = {b: i for i, b in enumerate(beta_vals)}
    step_to_idx = {s: j for j, s in enumerate(step_vals)}

    sts_matrix = np.full((len(beta_vals), len(step_vals)), np.nan, dtype=float)
    log_one_minus_conf = np.log(1.0 - TARGET_CONFIDENCE)

    for _, row in grouped.iterrows():
        beta = float(row["beta_final"])
        steps = int(row["steps"])
        p = float(row[p_column])

        if p <= 0.0 or p >= 1.0:
            sts = np.nan
        else:
            sts = steps * log_one_minus_conf / np.log(1.0 - p)

        bi = beta_to_idx[beta]
        sj = step_to_idx[steps]
        sts_matrix[bi, sj] = sts

    return beta_vals, step_vals, sts_matrix


def _plot_log_heatmap(matrix: np.ndarray, step_vals: np.ndarray, beta_vals: np.ndarray, title: str, out_filename: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))

    sts_for_plot = np.where(np.isfinite(matrix), matrix, np.nan)
    with np.errstate(invalid="ignore"):
        log_sts = np.log10(sts_for_plot)

    im = ax.imshow(
        log_sts,
        origin="lower",
        aspect="auto",
        extent=[step_vals[0], step_vals[-1], beta_vals[0], beta_vals[-1]],
        cmap="viridis",
    )

    ax.set_xlabel("Number of swaps N")
    ax.set_ylabel("Final inverse temperature beta")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(steps-to-solution)")

    fig.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    out_path = PLOTS_DIR / out_filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not INPUT_CSV.is_file():
        raise SystemExit(
            f"Input CSV '{INPUT_CSV}' not found. Run sa_schedule_grid_search_size10.py first."
        )

    df = pd.read_csv(INPUT_CSV)

    # We average success probability over the three size-10 instances in the grid.
    required_cols = {
        "beta_final",
        "steps",
        "prob_solved",
        "prob_ps_le_0.01",
        "prob_ps_le_0.1",
    }
    if not required_cols.issubset(df.columns):
        raise SystemExit(
            f"CSV must contain columns {sorted(required_cols)}. "
            "Did the grid search script run correctly?"
        )

    grouped = df.groupby(["beta_final", "steps"], as_index=False).mean()

    beta_vals, step_vals, sts_exact = _compute_sts_matrix(grouped, "prob_solved")
    _, _, sts_ps01 = _compute_sts_matrix(grouped, "prob_ps_le_0.01")
    _, _, sts_ps10 = _compute_sts_matrix(grouped, "prob_ps_le_0.1")

    _plot_log_heatmap(
        sts_exact,
        step_vals,
        beta_vals,
        "log10(STS) for exact success (size-10 SA schedules, 99% confidence)",
        "sa_schedule_sts_exact_heatmap_size10.png",
    )

    _plot_log_heatmap(
        sts_ps01,
        step_vals,
        beta_vals,
        "log10(STS) for Ps <= 0.01 (size-10 SA schedules, 99% confidence)",
        "sa_schedule_sts_ps01_heatmap_size10.png",
    )

    _plot_log_heatmap(
        sts_ps10,
        step_vals,
        beta_vals,
        "log10(STS) for Ps <= 0.1 (size-10 SA schedules, 99% confidence)",
        "sa_schedule_sts_ps10_heatmap_size10.png",
    )

    print(f"Saved STS heatmaps to '{PLOTS_DIR}' directory.")


if __name__ == "__main__":
    main()
