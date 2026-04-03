"""Plot steps-to-solution (STS) from SA schedule grid search for size 10.

This script reads `sa_schedule_grid_size10.csv` produced by
`sa_schedule_grid_search_size10.py`, aggregates the probability of
solving over the three representative instances for each
(beta_final, steps) pair, computes the steps-to-solution (STS) metric
for 99% confidence, and visualises it as a heatmap.

STS is defined as:

    STS = N_steps * ln(1 - 0.99) / ln(1 - p),

where N_steps is the number of Metropolis swaps per run and p is the
probability of solving (reaching the ground state) for that schedule.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_CSV = Path("sa_schedule_grid_size10.csv")
PLOTS_DIR = Path("plots")

TARGET_CONFIDENCE = 0.99


def main() -> None:
    if not INPUT_CSV.is_file():
        raise SystemExit(
            f"Input CSV '{INPUT_CSV}' not found. Run sa_schedule_grid_search_size10.py first."
        )

    df = pd.read_csv(INPUT_CSV)

    required_cols = {"beta_final", "steps", "prob_solved"}
    if not required_cols.issubset(df.columns):
        raise SystemExit(
            f"CSV must contain columns {sorted(required_cols)}. "
            "Did the grid search script run correctly?"
        )

    # Average probability of solving over instances for each (beta_final, steps).
    grouped = (
        df.groupby(["beta_final", "steps"], as_index=False)["prob_solved"]
        .mean()
        .rename(columns={"prob_solved": "mean_prob_solved"})
    )

    beta_vals = np.sort(grouped["beta_final"].unique())
    step_vals = np.sort(grouped["steps"].unique())

    beta_to_idx = {b: i for i, b in enumerate(beta_vals)}
    step_to_idx = {s: j for j, s in enumerate(step_vals)}

    sts_matrix = np.full((len(beta_vals), len(step_vals)), np.nan, dtype=float)

    log_one_minus_conf = np.log(1.0 - TARGET_CONFIDENCE)

    for _, row in grouped.iterrows():
        beta = float(row["beta_final"])
        steps = int(row["steps"])
        p = float(row["mean_prob_solved"])

        if p <= 0.0 or p >= 1.0:
            # STS is undefined if p <= 0 (never solves) or formally
            # equals N_steps if p == 1 (always solves). We treat
            # these edge cases as NaN in the heatmap.
            sts = np.nan
        else:
            sts = steps * log_one_minus_conf / np.log(1.0 - p)

        bi = beta_to_idx[beta]
        sj = step_to_idx[steps]
        sts_matrix[bi, sj] = sts

    PLOTS_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Use a log scale for STS to improve contrast, ignoring NaNs.
    sts_for_plot = np.where(np.isfinite(sts_matrix), sts_matrix, np.nan)
    # Avoid taking log of zeros or negatives.
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
    ax.set_title("log10(STS) for size-10 SA schedules (99% confidence)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(steps-to-solution)")

    fig.tight_layout()
    out_path = PLOTS_DIR / "sa_schedule_sts_heatmap_size10.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved STS heatmap to '{out_path}'")


if __name__ == "__main__":
    main()
