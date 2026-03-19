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


def run_all_calculations_bundle(spins: np.ndarray, J: np.ndarray, h: np.ndarray) -> dict[str, dict]:
	spins, J, h = _validate_inputs(spins, J, h)
	n = len(spins)

	trials = 5
	sa_steps = max(1_500, min(8_000, n * 450))
	ga_population = max(24, min(100, 4 * n))
	ga_generations = max(60, min(220, 12 * n))
	tabu_iterations = max(80, min(450, 18 * n))
	tabu_tenure = max(5, min(20, n // 2))

	header = _format_header(spins, J, h)
	initial_energy = float(hamiltonian_vectorized(spins, J, h))
	all_best_candidates: list[float] = [initial_energy]

	t0 = time.perf_counter()
	for _ in range(trials):
		_ = float(hamiltonian_vectorized(spins, J, h))
	h_total = time.perf_counter() - t0
	h_best_arr = np.full(trials, initial_energy, dtype=float)
	h_step_arr = np.zeros(trials, dtype=float)

	sa_best = np.empty(trials, dtype=float)
	sa_step_best = np.empty(trials, dtype=float)
	t0 = time.perf_counter()
	for r in range(trials):
		sa = pure_simulated_annealing(
			s0=spins,
			J=J,
			h=h,
			initial_temp=3.5,
			cooling_rate=0.998,
			steps=sa_steps,
			seed=42 + r,
		)
		sa_best[r] = float(sa.best_energy)
		sa_step_best[r] = float(_first_best_step(sa.energy_trace))
	sa_total = time.perf_counter() - t0
	all_best_candidates.append(float(np.min(sa_best)))

	ga_best = np.empty(trials, dtype=float)
	ga_step_best = np.empty(trials, dtype=float)
	t0 = time.perf_counter()
	for r in range(trials):
		ga = genetic_algorithm(
			base_spins=spins,
			J=J,
			h=h,
			population_size=ga_population,
			generations=ga_generations,
			elite_fraction=0.12,
			mutation_rate=0.25,
			tournament_k=4,
			seed=77 + r,
		)
		ga_best[r] = float(ga.best_energy)
		ga_step_best[r] = float(_first_best_step(ga.history_best_energy))
	ga_total = time.perf_counter() - t0
	all_best_candidates.append(float(np.min(ga_best)))

	ts_best = np.empty(trials, dtype=float)
	ts_step_best = np.empty(trials, dtype=float)
	t0 = time.perf_counter()
	for _ in range(trials):
		ts = tabu_search(
			s0=spins,
			J=J,
			h=h,
			iterations=tabu_iterations,
			tabu_tenure=tabu_tenure,
			max_no_improve=max(40, tabu_iterations // 3),
		)
		ts_best[_] = float(ts.best_energy)
		ts_step_best[_] = float(_first_best_step(ts.history_best_energy))
	ts_total = time.perf_counter() - t0
	all_best_candidates.append(float(np.min(ts_best)))

	reference_best = float(np.min(np.array(all_best_candidates, dtype=float)))

	h_cost = float(trials)
	sa_cost = float(trials * sa_steps)
	ga_cost = float(trials * ga_population * (ga_generations + 1))
	ts_cost = float(trials * tabu_iterations * (n * (n - 1) // 2))

	metrics = {
		"Hamiltonian Energy": _compute_metrics_dict(1, h_cost, h_total, h_best_arr, h_step_arr, reference_best),
		"Simulated Annealing": _compute_metrics_dict(sa_steps, sa_cost, sa_total, sa_best, sa_step_best, reference_best),
		"Genetic Algorithm": _compute_metrics_dict(ga_generations, ga_cost, ga_total, ga_best, ga_step_best, reference_best),
		"Tabu Search": _compute_metrics_dict(tabu_iterations, ts_cost, ts_total, ts_best, ts_step_best, reference_best),
	}

	texts = {
		"Hamiltonian Energy": header + "\n" + _format_metrics_block("Hamiltonian Energy", 1, h_cost, h_total, h_best_arr, h_step_arr, reference_best),
		"Simulated Annealing": header + "\n" + _format_metrics_block("Pure Simulated Annealing", sa_steps, sa_cost, sa_total, sa_best, sa_step_best, reference_best),
		"Genetic Algorithm": header + "\n" + _format_metrics_block("Genetic Algorithm", ga_generations, ga_cost, ga_total, ga_best, ga_step_best, reference_best),
		"Tabu Search": header + "\n" + _format_metrics_block("Tabu Search", tabu_iterations, ts_cost, ts_total, ts_best, ts_step_best, reference_best),
	}

	return {"texts": texts, "metrics": metrics}


def run_all_calculations(spins: np.ndarray, J: np.ndarray, h: np.ndarray) -> Dict[str, str]:
	return run_all_calculations_bundle(spins, J, h)["texts"]
