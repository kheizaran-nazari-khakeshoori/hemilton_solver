"""Export Markdown table summarizing SA performance across sizes.

Reads ``batch_results.csv`` (from run_batch_experiments.py) and prints
an aggregate Markdown table for the Simulated Annealing algorithm,
one row per problem size.

Usage (from project root):

    python export_batch_sa_markdown_summary.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


INPUT_CSV = Path("batch_results.csv")
ALGO_NAME = "Simulated Annealing"


def load_sa_results():
    if not INPUT_CSV.is_file():
        raise SystemExit(
            f"Input CSV '{INPUT_CSV}' not found. "
            "Run run_batch_experiments.py first."
        )

    residual_by_size: dict[int, list[float]] = defaultdict(list)
    success_by_size: dict[int, list[float]] = defaultdict(list)

    with INPUT_CSV.open("r", newline="") as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            if row.get("algorithm") != ALGO_NAME:
                continue

            try:
                size = int(row["size"])
                residual = float(row["residual_energy"])
                success_prob = float(row["success_probability"])
            except (KeyError, ValueError) as exc:
                raise SystemExit(f"Malformed SA row in {INPUT_CSV}: {row}") from exc

            residual_by_size[size].append(residual)
            success_by_size[size].append(success_prob)

    if not residual_by_size:
        raise SystemExit(
            f"No rows for algorithm '{ALGO_NAME}' found in {INPUT_CSV}. "
            "Did run_batch_experiments.py finish successfully?"
        )

    sizes = sorted(residual_by_size.keys())

    stats = []
    for s in sizes:
        res_arr = np.asarray(residual_by_size[s], dtype=float)
        succ_arr = np.asarray(success_by_size[s], dtype=float)
        stats.append(
            {
                "size": s,
                "mean_residual": float(np.mean(res_arr)),
                "median_residual": float(np.median(res_arr)),
                "max_residual": float(np.max(res_arr)),
                "success_rate": float(np.mean(succ_arr)),
            }
        )

    return stats


def main() -> None:
    stats = load_sa_results()

    print("## Simulated Annealing performance across sizes")
    print()
    print("Summary over all instances and trials per size, based on batch_results.csv.")
    print()
    print("| Size | Mean residual | Median residual | Max residual | Mean success rate |")
    print("|------|---------------|-----------------|-------------|-------------------|")

    for row in stats:
        print(
            f"| {row['size']:>4d} | "
            f"{row['mean_residual']:.6f} | "
            f"{row['median_residual']:.6f} | "
            f"{row['max_residual']:.6f} | "
            f"{row['success_rate']:.2%} |"
        )


if __name__ == "__main__":
    main()
