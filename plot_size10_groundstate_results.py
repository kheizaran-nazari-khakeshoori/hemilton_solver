"""Plot results of the size-10 ground-state vs SA experiment.

This script reads ``size10_ground_vs_sa.csv`` (produced by
``run_size10_groundstate_experiment.py``) and generates simple plots
showing how close Simulated Annealing (SA) gets to the exact ground
state across the 100 instances.

Requires matplotlib:

    pip install matplotlib

Usage (from project root):

    python plot_size10_groundstate_results.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT_CSV = Path("size10_ground_vs_sa.csv")
OUTPUT_DIR = Path("plots")


def load_results():
	instances = []
	gaps = []
	success_flags = []

	with INPUT_CSV.open("r", newline="") as f_in:
		reader = csv.DictReader(f_in)
		for row in reader:
			instances.append(int(row["instance"]))
			gaps.append(float(row["gap_sa_minus_ground"]))
			# stored as 0/1 float
			success_flags.append(float(row["sa_reached_ground"]))

	instances_arr = np.asarray(instances, dtype=int)
	gaps_arr = np.asarray(gaps, dtype=float)
	success_arr = np.asarray(success_flags, dtype=float)

	return instances_arr, gaps_arr, success_arr


def plot_gap_by_instance(instances, gaps):
	plt.figure(figsize=(8, 4))
	plt.scatter(instances, gaps, s=20, alpha=0.7)
	plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
	plt.title("Size 10: SA best cost minus ground-state cost per instance")
	plt.xlabel("Instance index")
	plt.ylabel("SA best cost - ground-state cost")
	plt.tight_layout()


def plot_gap_histogram(gaps):
	plt.figure(figsize=(6, 4))
	plt.hist(gaps, bins=20, edgecolor="black", alpha=0.8)
	plt.axvline(0.0, color="black", linewidth=1, linestyle="--")
	plt.title("Size 10: Distribution of SA gap to ground state")
	plt.xlabel("SA best cost - ground-state cost")
	plt.ylabel("Frequency")
	plt.tight_layout()


def main() -> None:
	if not INPUT_CSV.is_file():
		raise SystemExit(
			f"Input CSV '{INPUT_CSV}' not found. "
			"Run run_size10_groundstate_experiment.py first."
		)

	OUTPUT_DIR.mkdir(exist_ok=True)

	instances, gaps, success = load_results()

	# Basic statistics
	mean_gap = float(np.mean(gaps))
	max_gap = float(np.max(gaps))
	min_gap = float(np.min(gaps))
	success_rate = float(np.mean(success))

	print("Summary for size 10 (SA vs exact ground state):")
	print("------------------------------------------------")
	print(f"Mean gap (SA - exact): {mean_gap:.6f}")
	print(f"Min gap               : {min_gap:.6f}")
	print(f"Max gap               : {max_gap:.6f}")
	print(f"Success rate          : {success_rate:.2%}")

	# Create plots
	plot_gap_by_instance(instances, gaps)
	plot_gap_histogram(gaps)

	# Save figures
	gap_by_instance_path = OUTPUT_DIR / "size10_gap_by_instance.png"
	gap_hist_path = OUTPUT_DIR / "size10_gap_histogram.png"

	plt.figure(1)
	plt.savefig(gap_by_instance_path, dpi=150)

	plt.figure(2)
	plt.savefig(gap_hist_path, dpi=150)

	print(f"Saved plots to '{gap_by_instance_path}' and '{gap_hist_path}'")


if __name__ == "__main__":
	main()
