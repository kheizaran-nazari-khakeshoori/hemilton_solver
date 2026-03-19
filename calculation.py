"""Calculation service for running optimization algorithms on GUI parameters."""

from __future__ import annotations

import time
from typing import Dict

import numpy as np

from hemiltonian_energy import hamiltonian_vectorized
from pure_simulated_annealing import pure_simulated_annealing
from genetic_algorithm import genetic_algorithm
from tabu_search import tabu_search


def _validate_inputs(spins: np.ndarray, J: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	spins = np.asarray(spins, dtype=float)
	J = np.asarray(J, dtype=float)
	h = np.asarray(h, dtype=float)

	n = len(spins)
	if n < 2:
		raise ValueError("Need at least 2 spins")
	if J.shape != (n, n):
		raise ValueError(f"J must have shape ({n}, {n}), got {J.shape}")
	if h.shape != (n,):
		raise ValueError(f"h must have shape ({n},), got {h.shape}")
	return spins, J, h


def _format_header(spins: np.ndarray, J: np.ndarray, h: np.ndarray) -> str:
	initial_energy = float(hamiltonian_vectorized(spins, J, h))
	nonzero_edges = int(np.count_nonzero(np.triu(J, k=1)))
	return (
		f"N spins        : {len(spins)}\n"
		f"Initial energy : {initial_energy:.6f}\n"
		f"Non-zero J_ij  : {nonzero_edges}\n"
	)


def _first_best_step(trace: np.ndarray) -> int:
	if trace.size == 0:
		return 0
	best = float(np.min(trace))
	return int(np.where(trace == best)[0][0])


def _success_probability(best_energies: np.ndarray, reference_best: float, tol: float = 1e-9) -> float:
	if best_energies.size == 0:
		return 0.0
	return float(np.mean(np.abs(best_energies - reference_best) <= tol))


def _format_metrics_block(
	name: str,
	steps: int,
	cost: float,
	total_runtime: float,
	best_energies: np.ndarray,
	step_to_best: np.ndarray,
	reference_best: float,
) -> str:
	best_energy = float(np.min(best_energies))
	median_energy = float(np.median(best_energies))
	std_energy = float(np.std(best_energies))
	success_prob = _success_probability(best_energies, reference_best)
	residual = float(best_energy - reference_best)
	avg_step_best = float(np.mean(step_to_best)) if step_to_best.size else 0.0

	if steps > 0:
		avg_time_to_best = total_runtime * (avg_step_best / steps)
	else:
		avg_time_to_best = 0.0

	return (
		f"Algorithm            : {name}\n"
		f"Time taken (total)   : {total_runtime:.6f} s\n"
		f"Steps                : {steps}\n"
		f"Computational cost   : {cost:.2f}\n"
		f"Best energy          : {best_energy:.6f}\n"
		f"How fast to best     : {avg_time_to_best:.6f} s (avg step {avg_step_best:.1f})\n"
		f"Min / Median energy  : {best_energy:.6f} / {median_energy:.6f}\n"
		f"Success probability  : {success_prob:.2%}\n"
		f"Standard deviation   : {std_energy:.6f}\n"
		f"Residual energy      : {residual:.6f}\n"
	)


def _compute_metrics_dict(
	steps: int,
	cost: float,
	total_runtime: float,
	best_energies: np.ndarray,
	step_to_best: np.ndarray,
	reference_best: float,
) -> dict[str, float]:
	best_energy = float(np.min(best_energies))
	median_energy = float(np.median(best_energies))
	std_energy = float(np.std(best_energies))
	success_prob = _success_probability(best_energies, reference_best)
	residual = float(best_energy - reference_best)
	avg_step_best = float(np.mean(step_to_best)) if step_to_best.size else 0.0
	if steps > 0:
		avg_time_to_best = total_runtime * (avg_step_best / steps)
	else:
		avg_time_to_best = 0.0

	return {
		"time_taken": float(total_runtime),
		"steps": float(steps),
		"computational_cost": float(cost),
		"best_energy": best_energy,
		"time_to_best": float(avg_time_to_best),
		"min_energy": best_energy,
		"median_energy": median_energy,
		"success_probability": float(success_prob),
		"standard_deviation": std_energy,
		"residual_energy": residual,
	}
