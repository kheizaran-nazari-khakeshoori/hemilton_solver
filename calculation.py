"""Calculation service for running optimization algorithms on GUI parameters."""

from __future__ import annotations

import time
from typing import Dict

import numpy as np

from hemiltonian_energy import qap_cost
from pure_simulated_annealing import pure_simulated_annealing
# from genetic_algorithm import genetic_algorithm
# from tabu_search import tabu_search


def _validate_inputs(permutation_seed: np.ndarray, F: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	permutation_seed = np.asarray(permutation_seed, dtype=float)
	F = np.asarray(F, dtype=float)
	D = np.asarray(D, dtype=float)

	n = len(permutation_seed)
	if n < 2:
		raise ValueError("Need at least 2 facilities")
	if F.shape != (n, n):
		raise ValueError(f"F must have shape ({n}, {n}), got {F.shape}")
	if D.shape != (n, n):
		raise ValueError(f"D must have shape ({n}, {n}), got {D.shape}")
	return permutation_seed, F, D


def _seed_to_permutation(seed: np.ndarray) -> np.ndarray:
	# Convert any numeric seed vector to a valid permutation [0..N-1].
	idx = np.arange(len(seed), dtype=int)
	return np.lexsort((idx, seed)).astype(int)


def _format_header(p0: np.ndarray, F: np.ndarray, D: np.ndarray) -> str:
	initial_cost = float(qap_cost(F, D, p0))
	nonzero_flow = int(np.count_nonzero(F))
	nonzero_dist = int(np.count_nonzero(D))
	return (
		f"N facilities   : {len(p0)}\n"
		f"Initial cost   : {initial_cost:.6f}\n"
		f"Non-zero F_ij  : {nonzero_flow}\n"
		f"Non-zero D_ij  : {nonzero_dist}\n"
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
	best_costs: np.ndarray,
	step_to_best: np.ndarray,
	reference_best: float,
) -> str:
	best_cost = float(np.min(best_costs))
	median_cost = float(np.median(best_costs))
	std_cost = float(np.std(best_costs))
	success_prob = _success_probability(best_costs, reference_best)
	residual = float(best_cost - reference_best)
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
		f"Best cost            : {best_cost:.6f}\n"
		f"How fast to best     : {avg_time_to_best:.6f} s (avg step {avg_step_best:.1f})\n"
		f"Min / Median cost    : {best_cost:.6f} / {median_cost:.6f}\n"
		f"Success probability  : {success_prob:.2%}\n"
		f"Standard deviation   : {std_cost:.6f}\n"
		f"Residual cost        : {residual:.6f}\n"
	)


def _compute_metrics_dict(
	steps: int,
	cost: float,
	total_runtime: float,
	best_costs: np.ndarray,
	step_to_best: np.ndarray,
	reference_best: float,
) -> dict[str, float]:
	best_cost = float(np.min(best_costs))
	median_cost = float(np.median(best_costs))
	std_cost = float(np.std(best_costs))
	success_prob = _success_probability(best_costs, reference_best)
	residual = float(best_cost - reference_best)
	avg_step_best = float(np.mean(step_to_best)) if step_to_best.size else 0.0
	if steps > 0:
		avg_time_to_best = total_runtime * (avg_step_best / steps)
	else:
		avg_time_to_best = 0.0

	return {
		"time_taken": float(total_runtime),
		"steps": float(steps),
		"computational_cost": float(cost),
		"best_energy": best_cost,
		"time_to_best": float(avg_time_to_best),
		"min_energy": best_cost,
		"median_energy": median_cost,
		"success_probability": float(success_prob),
		"standard_deviation": std_cost,
		"residual_energy": residual,
	}


def run_all_calculations_bundle(permutation_seed: np.ndarray, F: np.ndarray, D: np.ndarray) -> dict[str, dict]:
	permutation_seed, F, D = _validate_inputs(permutation_seed, F, D)
	p0 = _seed_to_permutation(permutation_seed)
	n = len(p0)

	trials = 5
	sa_steps = max(1_500, min(8_000, n * 450))
	# ga_population = max(24, min(100, 4 * n))
	# ga_generations = max(60, min(220, 12 * n))
	# tabu_iterations = max(80, min(450, 18 * n))
	# tabu_tenure = max(5, min(20, n // 2))

	header = _format_header(p0, F, D)
	initial_cost = float(qap_cost(F, D, p0))
	all_best_candidates: list[float] = [initial_cost]

	t0 = time.perf_counter()
	for _ in range(trials):
		_ = float(qap_cost(F, D, p0))
	h_total = time.perf_counter() - t0
	h_best_arr = np.full(trials, initial_cost, dtype=float)
	h_step_arr = np.zeros(trials, dtype=float)

	sa_best = np.empty(trials, dtype=float)
	sa_step_best = np.empty(trials, dtype=float)
	t0 = time.perf_counter()
	for r in range(trials):
		sa = pure_simulated_annealing(
			p0=np.roll(p0, r),
			F=F,
			D=D,
			initial_temp=3.5,
			cooling_rate=0.998,
			steps=sa_steps,
			seed=42 + r,
		)
		sa_best[r] = float(sa.best_cost)
		sa_step_best[r] = float(_first_best_step(sa.cost_trace))
	sa_total = time.perf_counter() - t0
	all_best_candidates.append(float(np.min(sa_best)))
	# Genetic Algorithm and Tabu Search are disabled for this configuration.
	#
	# ga_best = np.empty(trials, dtype=float)
	# ga_step_best = np.empty(trials, dtype=float)
	# t0 = time.perf_counter()
	# for r in range(trials):
	# 	ga = genetic_algorithm(
	# 		p0=np.roll(p0, r),
	# 		F=F,
	# 		D=D,
	# 		population_size=ga_population,
	# 		generations=ga_generations,
	# 		elite_fraction=0.12,
	# 		mutation_rate=0.25,
	# 		tournament_k=4,
	# 		seed=77 + r,
	# 	)
	# 	ga_best[r] = float(ga.best_cost)
	# 	ga_step_best[r] = float(_first_best_step(ga.history_best_cost))
	# ga_total = time.perf_counter() - t0
	# all_best_candidates.append(float(np.min(ga_best)))
	#
	# ts_best = np.empty(trials, dtype=float)
	# ts_step_best = np.empty(trials, dtype=float)
	# t0 = time.perf_counter()
	# for _ in range(trials):
	# 	ts = tabu_search(
	# 		p0=np.roll(p0, _),
	# 		F=F,
	# 		D=D,
	# 		iterations=tabu_iterations,
	# 		tabu_tenure=tabu_tenure,
	# 		max_no_improve=max(40, tabu_iterations // 3),
	# 	)
	# 	ts_best[_] = float(ts.best_cost)
	# 	ts_step_best[_] = float(_first_best_step(ts.history_best_cost))
	# ts_total = time.perf_counter() - t0
	# all_best_candidates.append(float(np.min(ts_best)))

	reference_best = float(np.min(np.array(all_best_candidates, dtype=float)))

	h_cost = float(trials)
	sa_cost = float(trials * sa_steps)
	# ga_cost = float(trials * ga_population * (ga_generations + 1))
	# ts_cost = float(trials * tabu_iterations * (n * (n - 1) // 2))

	metrics = {
		"QAP Objective": _compute_metrics_dict(1, h_cost, h_total, h_best_arr, h_step_arr, reference_best),
		"Simulated Annealing": _compute_metrics_dict(sa_steps, sa_cost, sa_total, sa_best, sa_step_best, reference_best),
	}

	texts = {
		"QAP Objective": header + "\n" + _format_metrics_block("QAP Objective", 1, h_cost, h_total, h_best_arr, h_step_arr, reference_best),
		"Simulated Annealing": header + "\n" + _format_metrics_block("Pure Simulated Annealing", sa_steps, sa_cost, sa_total, sa_best, sa_step_best, reference_best),
	}

	return {"texts": texts, "metrics": metrics}


def run_all_calculations(permutation_seed: np.ndarray, F: np.ndarray, D: np.ndarray) -> Dict[str, str]:
	return run_all_calculations_bundle(permutation_seed, F, D)["texts"]
