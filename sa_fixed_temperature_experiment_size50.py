"""Fixed-temperature Metropolis (SA) experiment for size 50.

This is a size-50 variant of sa_fixed_temperature_experiment.py.
It sweeps over beta and number of Metropolis swaps at fixed
problem size N=50 for a few instances.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from pure_simulated_annealing import pure_simulated_annealing
from hemiltonian_energy import qap_cost


INSTANCES_DIR = Path("instances")

SIZE = 50
INSTANCE_IDS = [1, 2, 3]

BETA_MIN = 0.1
BETA_MAX = 2.0
N_BETA_VALUES = 10

# Scale steps range roughly proportional to problem size (50 vs 10)
STEPS_MIN = 500
STEPS_MAX = 15000
N_STEPS_VALUES = 10

RUNS_PER_CONFIG = 100

OUTPUT_CSV = Path("sa_fixed_temp_traces_size50.csv")


def test_sa_fixed_temperature(F: np.ndarray, D: np.ndarray, beta: float, steps: int, seed_offset: int = 0) -> np.ndarray:
	if beta <= 0:
		raise ValueError("beta must be > 0 for fixed-temperature SA")
	if steps <= 0:
		raise ValueError("steps must be > 0")

	n = F.shape[0]
	p0 = np.arange(n, dtype=int)

	result = pure_simulated_annealing(
		p0=p0,
		F=F,
		D=D,
		initial_temp=1.0 / beta,
		cooling_rate=1.0,
		steps=steps,
		seed=42 + int(seed_offset),
	)

	return result.cost_trace.copy()


def main() -> None:
	if not INSTANCES_DIR.is_dir():
		raise SystemExit(
			f"Instances directory '{INSTANCES_DIR}' not found. "
			"Run generate_instances.py first."
		)

	betas = np.linspace(BETA_MIN, BETA_MAX, N_BETA_VALUES)
	step_values = np.linspace(STEPS_MIN, STEPS_MAX, N_STEPS_VALUES, dtype=int)

	with OUTPUT_CSV.open("w", newline="") as f_out:
		fieldnames = [
			"size",
			"instance",
			"beta_index",
			"beta",
			"steps_index",
			"steps_max",
			"t",
			"mean_energy",
		]
		writer = csv.DictWriter(f_out, fieldnames=fieldnames)
		writer.writeheader()

		for inst_id in INSTANCE_IDS:
			fname = INSTANCES_DIR / f"{SIZE}-{inst_id}.npz"
			if not fname.is_file():
				raise SystemExit(f"Missing instance file: {fname}. Generate it first.")

			data = np.load(fname)
			if "F" not in data or "D" not in data:
				raise SystemExit(f"File {fname} must contain 'F' and 'D' arrays.")

			F = np.asarray(data["F"], dtype=float)
			D = np.asarray(data["D"], dtype=float)

			if F.shape != (SIZE, SIZE) or D.shape != (SIZE, SIZE):
				raise SystemExit(f"File {fname} has inconsistent shapes: F={F.shape}, D={D.shape}.")

			initial_perm = np.arange(SIZE, dtype=int)
			initial_cost = float(qap_cost(F, D, initial_perm))
			print(f"Instance size={SIZE}, id={inst_id}: initial cost = {initial_cost:.6f}")

			for b_idx, beta in enumerate(betas):
				for s_idx, steps in enumerate(step_values):
					energy_sum = np.zeros(steps, dtype=float)

					for run in range(RUNS_PER_CONFIG):
						seed_offset = (
							inst_id * 1_000_000
							+ b_idx * 10_000
							+ s_idx * 100
							+ run
						)
						trace = test_sa_fixed_temperature(F, D, beta=beta, steps=int(steps), seed_offset=seed_offset)
						energy_sum[: len(trace)] += trace

					mean_energy = energy_sum / float(RUNS_PER_CONFIG)

					for t, e in enumerate(mean_energy):
						writer.writerow(
							{
								"size": SIZE,
								"instance": inst_id,
								"beta_index": b_idx,
								"beta": float(beta),
								"steps_index": s_idx,
								"steps_max": int(steps),
								"t": int(t),
								"mean_energy": float(e),
							}
						)

	print(f"Wrote fixed-temperature SA traces (size={SIZE}) to {OUTPUT_CSV}")


if __name__ == "__main__":
	main()
