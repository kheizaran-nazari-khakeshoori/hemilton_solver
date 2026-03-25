"""Visualize Simulated Annealing performance across all problem sizes.

This script reads ``batch_results.csv`` (produced by
``run_batch_experiments.py`` / ``main.py``), filters the
"Simulated Annealing" rows, and plots how performance changes
with problem size.

It produces simple figures such as:
- mean residual energy vs size
- mean success probability vs size
- boxplot of residual energy per size

Requires matplotlib and numpy:

    pip install matplotlib numpy

Usage (from project root):

    python plot_batch_sa_performance.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT_CSV = Path("batch_results.csv")
OUTPUT_DIR = Path("plots")
ALGO_NAME = "Simulated Annealing"


def load_sa_results():
    """Load Simulated Annealing rows grouped by size.

    Returns
    -------
    sizes : np.ndarray[int]
        Sorted array of unique sizes.
    residual_by_size : dict[int, np.ndarray]
        Mapping size -> residual_energy samples across all instances.
    success_prob_by_size : dict[int, np.ndarray]
        Mapping size -> success_probability samples across all instances.
    """

    if not INPUT_CSV.is_file():
        raise SystemExit(
            f"Input CSV '{INPUT_CSV}' not found. "
            "Run run_batch_experiments.py (or main.py) first."
        )

    residual_by_size: dict[int, list[float]] = defaultdict(list)
    success_prob_by_size: dict[int, list[float]] = defaultdict(list)

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
            success_prob_by_size[size].append(success_prob)

    if not residual_by_size:
        raise SystemExit(
            f"No rows for algorithm '{ALGO_NAME}' found in {INPUT_CSV}. "
            "Did run_batch_experiments.py finish successfully?"
        )

    sizes = np.array(sorted(residual_by_size.keys()), dtype=int)

    residual_by_size_np: dict[int, np.ndarray] = {}
    success_prob_by_size_np: dict[int, np.ndarray] = {}

    for s in sizes:
        residual_by_size_np[s] = np.asarray(residual_by_size[s], dtype=float)
        success_prob_by_size_np[s] = np.asarray(success_prob_by_size[s], dtype=float)

    return sizes, residual_by_size_np, success_prob_by_size_np


def plot_mean_residual_vs_size(sizes: np.ndarray, residual_by_size: dict[int, np.ndarray]) -> None:
    mean_residual = np.array([np.mean(residual_by_size[s]) for s in sizes], dtype=float)

    plt.figure(figsize=(6, 4))
    plt.plot(sizes, mean_residual, marker="o")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.title("Simulated Annealing: Mean residual energy vs size")
    plt.xlabel("Problem size N")
    plt.ylabel("Mean residual energy (best cost - reference best)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_mean_success_vs_size(sizes: np.ndarray, success_prob_by_size: dict[int, np.ndarray]) -> None:
    mean_success = np.array([np.mean(success_prob_by_size[s]) for s in sizes], dtype=float)

    plt.figure(figsize=(6, 4))
    plt.plot(sizes, mean_success, marker="o")
    plt.ylim(0.0, 1.05)
    plt.title("Simulated Annealing: Mean success probability vs size")
    plt.xlabel("Problem size N")
    plt.ylabel("Mean success probability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_residual_boxplot_by_size(sizes: np.ndarray, residual_by_size: dict[int, np.ndarray]) -> None:
    data = [residual_by_size[s] for s in sizes]

    plt.figure(figsize=(8, 4))
    plt.boxplot(data, positions=sizes, widths=3)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.title("Simulated Annealing: Residual energy distribution by size")
    plt.xlabel("Problem size N")
    plt.ylabel("Residual energy (best cost - reference best)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    sizes, residual_by_size, success_prob_by_size = load_sa_results()

    print("SA performance summary across sizes:")
    print("------------------------------------")
    for s in sizes:
        res = residual_by_size[s]
        succ = success_prob_by_size[s]
        print(
            f"Size {s:3d}: mean residual = {np.mean(res):.6f}, "
            f"median residual = {np.median(res):.6f}, "
            f"success (mean) = {np.mean(succ):.2%}"
        )

    # Create plots
    plot_mean_residual_vs_size(sizes, residual_by_size)
    plot_mean_success_vs_size(sizes, success_prob_by_size)
    plot_residual_boxplot_by_size(sizes, residual_by_size)

    # Save figures (current figures are numbered in creation order)
    mean_residual_path = OUTPUT_DIR / "sa_mean_residual_vs_size.png"
    mean_success_path = OUTPUT_DIR / "sa_mean_success_vs_size.png"
    residual_boxplot_path = OUTPUT_DIR / "sa_residual_boxplot_by_size.png"

    plt.figure(1)
    plt.savefig(mean_residual_path, dpi=150)

    plt.figure(2)
    plt.savefig(mean_success_path, dpi=150)

    plt.figure(3)
    plt.savefig(residual_boxplot_path, dpi=150)

    print("Saved plots:")
    print(f" - {mean_residual_path}")
    print(f" - {mean_success_path}")
    print(f" - {residual_boxplot_path}")


if __name__ == "__main__":
    main()
