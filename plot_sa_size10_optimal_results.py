"""Plot results of SA evaluation with optimal schedule on size-10 instances.

This script reads `sa_size10_optimal_schedule_results.csv`, produced by
`evaluate_sa_size10_optimal_schedule.py`, and generates a couple of
simple diagnostic plots:

- Probability of solving vs instance index.
- Histogram of per-instance solving probabilities.

These are mainly for visualization; the key summary number remains the
overall average probability of solving printed by the evaluation script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_CSV = Path("sa_size10_optimal_schedule_results.csv")
PLOTS_DIR = Path("plots")


def main() -> None:
    if not INPUT_CSV.is_file():
        raise SystemExit(
            f"Input CSV '{INPUT_CSV}' not found. Run evaluate_sa_size10_optimal_schedule.py first."
        )

    df = pd.read_csv(INPUT_CSV)

    if "instance" not in df.columns or "prob_solved" not in df.columns:
        raise SystemExit(
            "CSV must contain 'instance' and 'prob_solved' columns. "
            "Did the evaluation script run correctly?"
        )

    PLOTS_DIR.mkdir(exist_ok=True)

    # Plot 1: probability of solving vs instance index
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(df["instance"], df["prob_solved"], marker="o", linestyle="none")
    ax1.set_xlabel("Instance index")
    ax1.set_ylabel("Probability of solving")
    ax1.set_title("SA solving probability per size-10 instance (optimal schedule)")
    ax1.set_ylim(-0.05, 1.05)
    fig1.tight_layout()
    out1 = PLOTS_DIR / "sa_size10_optimal_prob_by_instance.png"
    fig1.savefig(out1, dpi=150)
    plt.close(fig1)

    # Plot 2: histogram of per-instance probabilities
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(df["prob_solved"], bins=20, range=(0.0, 1.0), edgecolor="black")
    ax2.set_xlabel("Probability of solving")
    ax2.set_ylabel("Number of instances")
    ax2.set_title("Distribution of SA solving probability (size 10, optimal schedule)")
    fig2.tight_layout()
    out2 = PLOTS_DIR / "sa_size10_optimal_prob_histogram.png"
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)

    print(f"Saved plots to '{PLOTS_DIR}' directory:")
    print(f"  - {out1}")
    print(f"  - {out2}")


if __name__ == "__main__":
    main()
