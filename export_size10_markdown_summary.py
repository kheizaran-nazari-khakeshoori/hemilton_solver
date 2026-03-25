"""Export Markdown summary for size-10 ground-state vs SA experiment.

Reads ``size10_ground_vs_sa.csv`` (from run_size10_groundstate_experiment.py)
and prints a compact Markdown summary you can paste into your report.

Usage (from project root):

    python export_size10_markdown_summary.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


INPUT_CSV = Path("size10_ground_vs_sa.csv")


def load_results():
    instances = []
    gaps = []
    success_flags = []

    with INPUT_CSV.open("r", newline="") as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            instances.append(int(row["instance"]))
            gaps.append(float(row["gap_sa_minus_ground"]))
            success_flags.append(float(row["sa_reached_ground"]))

    instances_arr = np.asarray(instances, dtype=int)
    gaps_arr = np.asarray(gaps, dtype=float)
    success_arr = np.asarray(success_flags, dtype=float)

    return instances_arr, gaps_arr, success_arr


def main() -> None:
    if not INPUT_CSV.is_file():
        raise SystemExit(
            f"Input CSV '{INPUT_CSV}' not found. "
            "Run run_size10_groundstate_experiment.py first."
        )

    instances, gaps, success = load_results()

    mean_gap = float(np.mean(gaps))
    min_gap = float(np.min(gaps))
    max_gap = float(np.max(gaps))
    success_rate = float(np.mean(success))

    # Print a small Markdown section summarizing size-10 performance.
    print("## Size 10: Simulated Annealing vs exact ground state")
    print()
    print("Summary over 100 random instances of size 10.")
    print()
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| Mean gap (SA - exact) | {mean_gap:.6f} |")
    print(f"| Min gap | {min_gap:.6f} |")
    print(f"| Max gap | {max_gap:.6f} |")
    print(f"| SA success rate (reached ground) | {success_rate:.2%} |")


if __name__ == "__main__":
    main()
