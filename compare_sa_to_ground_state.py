"""Compare Simulated Annealing with exact QAP ground state on a single instance.

Usage example (from project root):

    python compare_sa_to_ground_state.py --size 10 --instance 1

This will load instances/10-1.npz, compute the exact ground state via
brute-force enumeration (only allowed for small N), then run the
Simulated Annealing algorithm starting from a simple seed permutation
and report both energies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from exact_qap_ground_state import brute_force_ground_state
from pure_simulated_annealing import pure_simulated_annealing
from hemiltonian_energy import qap_cost


INSTANCES_DIR = Path("instances")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Compare SA with exact QAP ground state on one instance")
	parser.add_argument("--size", type=int, required=True, help="Problem size N (e.g. 10)")
	parser.add_argument("--instance", type=int, required=True, help="Instance index (1..100)")
	parser.add_argument(
		"--max-n",
		type=int,
		default=10,
		help="Maximum N allowed for brute-force search (default: 10)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	N = args.size
	inst_idx = args.instance
	max_n = args.max_n

	fname = INSTANCES_DIR / f"{N}-{inst_idx}.npz"
	if not fname.is_file():
		raise SystemExit(f"Instance file not found: {fname}. Run generate_instances.py first.")

	data = np.load(fname)
	if "F" not in data or "D" not in data:
		raise SystemExit(f"File {fname} must contain 'F' and 'D' arrays.")

	F = np.asarray(data["F"], dtype=float)
	D = np.asarray(data["D"], dtype=float)

	if F.shape != D.shape or F.shape[0] != F.shape[1]:
		raise SystemExit(f"Inconsistent shapes in {fname}: F={F.shape}, D={D.shape}")

	if F.shape[0] != N:
		raise SystemExit(f"File {fname} has size {F.shape[0]}, but --size {N} was requested.")

	print(f"Loaded instance: size={N}, index={inst_idx}")
	print(f"Brute-force max_n = {max_n}")

	# Exact ground state (brute-force over all permutations).
	ground = brute_force_ground_state(F, D, max_n=max_n)

	print("\nExact ground state (brute-force):")
	print("---------------------------------")
	print(f"Best cost (ground state): {ground.best_cost:.6f}")
	print(f"Best permutation        : {ground.best_permutation.tolist()}")
	print(f"Permutations evaluated  : {ground.evaluations}")

	# Now run Simulated Annealing once from a simple seed permutation.
	n = F.shape[0]
	p0 = np.arange(n, dtype=int)

	# Match the step schedule used in calculation.run_all_calculations_bundle.
	sa_steps = max(1_500, min(8_000, n * 450))

	sa_result = pure_simulated_annealing(
		p0=p0,
		F=F,
		D=D,
		initial_temp=3.5,
		cooling_rate=0.998,
		steps=sa_steps,
		seed=42,
	)

	initial_cost = float(qap_cost(F, D, p0))

	print("\nSimulated Annealing result:")
	print("---------------------------")
	print(f"Initial cost           : {initial_cost:.6f}")
	print(f"SA best cost           : {sa_result.best_cost:.6f}")
	print(f"SA final cost          : {sa_result.final_cost:.6f}")
	print(f"SA accepted moves      : {sa_result.accepted_moves}/{sa_result.attempted_moves}")

	diff = sa_result.best_cost - ground.best_cost
	print("\nComparison:")
	print("-----------")
	print(f"Ground state cost      : {ground.best_cost:.6f}")
	print(f"SA best cost           : {sa_result.best_cost:.6f}")
	print(f"Difference (SA - exact): {diff:.6f}")

	if abs(diff) <= 1e-9:
		print("SA reached the exact ground state within numerical tolerance.")
	else:
		print("SA did NOT reach the exact ground state (for this single run).")


if __name__ == "__main__":
	main()
