"""Tabu Search for QAP minimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from hemiltonian_energy import qap_cost


Move = Tuple[int, int]


@dataclass
class TabuResult:
	best_permutation: np.ndarray
	best_cost: float
	final_permutation: np.ndarray
	final_cost: float
	history_best_cost: np.ndarray
	accepted_worse_moves: int


def _canonical_move(i: int, j: int) -> Move:
	return (i, j) if i < j else (j, i)


def tabu_search(
	p0: np.ndarray,
	F: np.ndarray,
	D: np.ndarray,
	iterations: int = 500,
	tabu_tenure: int = 12,
	max_no_improve: Optional[int] = None,
) -> TabuResult:
	"""Run swap-based tabu search."""
	P = np.asarray(p0, dtype=int).copy()
	F = np.asarray(F, dtype=float)
	D = np.asarray(D, dtype=float)

	n = len(P)
	if n < 2:
		raise ValueError("Need at least 2 facilities")
	if F.shape != (n, n):
		raise ValueError(f"F must have shape ({n}, {n}), got {F.shape}")
	if D.shape != (n, n):
		raise ValueError(f"D must have shape ({n}, {n}), got {D.shape}")
	if set(P.tolist()) != set(range(n)):
		raise ValueError("p0 must be a valid permutation")
	if iterations < 1:
		raise ValueError("iterations must be >= 1")
	if tabu_tenure < 1:
		raise ValueError("tabu_tenure must be >= 1")

	current_cost = float(qap_cost(F, D, P))
	best_cost = current_cost
	best_perm = P.copy()

	tabu_until: Dict[Move, int] = {}

	history = np.empty(iterations, dtype=float)
	no_improve_count = 0
	accepted_worse = 0

	for t in range(iterations):
		best_candidate_cost = np.inf
		best_candidate_move: Optional[Move] = None

		for i in range(n - 1):
			for j in range(i + 1, n):
				move = (i, j)

				P[i], P[j] = P[j], P[i]
				cand_cost = float(qap_cost(F, D, P))
				P[i], P[j] = P[j], P[i]

				is_tabu = tabu_until.get(move, -1) >= t
				improves_global = cand_cost < best_cost

				if is_tabu and not improves_global:
					continue

				if cand_cost < best_candidate_cost:
					best_candidate_cost = cand_cost
					best_candidate_move = move

		if best_candidate_move is None:
			tabu_until.clear()
			history[t] = best_cost
			continue

		i, j = best_candidate_move
		P[i], P[j] = P[j], P[i]
		prev_cost = current_cost
		current_cost = best_candidate_cost

		if current_cost > prev_cost:
			accepted_worse += 1

		tabu_until[_canonical_move(i, j)] = t + tabu_tenure

		if current_cost < best_cost:
			best_cost = current_cost
			best_perm = P.copy()
			no_improve_count = 0
		else:
			no_improve_count += 1

		history[t] = best_cost

		if max_no_improve is not None and no_improve_count >= max_no_improve:
			history = history[: t + 1]
			break

	return TabuResult(
		best_permutation=best_perm,
		best_cost=float(best_cost),
		final_permutation=P.copy(),
		final_cost=float(current_cost),
		history_best_cost=history,
		accepted_worse_moves=accepted_worse,
	)


if __name__ == "__main__":
	N = 18
	rng = np.random.default_rng(13)
	p0 = rng.permutation(N)
	F = rng.integers(0, 10, size=(N, N)).astype(float)
	D = rng.integers(0, 10, size=(N, N)).astype(float)
	np.fill_diagonal(F, 0)
	np.fill_diagonal(D, 0)

	initial_cost = float(qap_cost(F, D, p0))
	result = tabu_search(
		p0=p0,
		F=F,
		D=D,
		iterations=500,
		tabu_tenure=10,
		max_no_improve=120,
	)

	print("Tabu Search (QAP)")
	print(f"Initial cost        : {initial_cost:.6f}")
	print(f"Final cost          : {result.final_cost:.6f}")
	print(f"Best cost           : {result.best_cost:.6f}")
	print(f"Accepted worse moves: {result.accepted_worse_moves}")
