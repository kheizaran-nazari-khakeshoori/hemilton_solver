"""Fixed-temperature Metropolis (SA) experiment over beta and step ranges.

This script implements the loop your professor described:

    for beta in np.linspace(B0, B1, 10):
        for N_steps in np.linspace(N0, N1, 10):
            for i in range(100):
                test_sa(beta, N_steps)

Here, beta = 1 / T is the inverse temperature, and N_steps is the number
of Metropolis swap attempts (time horizon). We keep the problem size
fixed (by default N=10) and choose a few problem instances.

For each (instance, beta, N_steps) configuration we run multiple
independent fixed-temperature SA trajectories and store the *average*
energy E(t) as a function of t (Metropolis swaps).

Results are written to a CSV file:

    sa_fixed_temp_traces_size10.csv

with columns:

    size, instance, beta_index, beta, steps_index, steps_max, t, mean_energy

which can be plotted by a separate script.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from pure_simulated_annealing import pure_simulated_annealing
from hemiltonian_energy import qap_cost


INSTANCES_DIR = Path("instances")

# Problem size to study (QAP dimension)
SIZE = 10
# Choose a few representative instances for this size
INSTANCE_IDS = [1, 2, 3]

# Beta (inverse temperature) range: 10 values between BETA_MIN and BETA_MAX
BETA_MIN = 0.1
BETA_MAX = 2.0
N_BETA_VALUES = 10

# Number of Metropolis steps (swaps) per run: 10 values between these bounds
STEPS_MIN = 100
STEPS_MAX = 3000
N_STEPS_VALUES = 10

# Number of independent runs per (instance, beta, steps) configuration
RUNS_PER_CONFIG = 100

OUTPUT_CSV = Path("sa_fixed_temp_traces_size10.csv")


def test_sa_fixed_temperature(F: np.ndarray, D: np.ndarray, beta: float, steps: int, seed_offset: int = 0) -> np.ndarray:
	"""Run fixed-temperature Metropolis SA and return energy trace.

	This uses the existing pure_simulated_annealing implementation with
	constant temperature (cooling_rate = 1.0). The initial temperature is
	set to 1 / beta so that the acceptance probability is exp(-beta * dE).
	"""

	if beta <= 0:
		raise ValueError("beta must be > 0 for fixed-temperature SA")
	if steps <= 0:
		raise ValueError("steps must be > 0")

	n = F.shape[0]
	# Simple deterministic starting permutation; randomness comes from moves.
	p0 = np.arange(n, dtype=int)

	result = pure_simulated_annealing(
		p0=p0,
		F=F,
		D=D,
		initial_temp=1.0 / beta,
		cooling_rate=1.0,  # fixed temperature
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
					# Accumulate energies over RUNS_PER_CONFIG runs
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

	print(f"Wrote fixed-temperature SA traces to {OUTPUT_CSV}")


if __name__ == "__main__":
	main()
