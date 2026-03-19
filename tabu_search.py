"""Tabu Search for Ising energy minimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from hemiltonian_energy import hamiltonian_vectorized


Move = Tuple[int, int]


@dataclass
class TabuResult:
	best_spins: np.ndarray
	best_energy: float
	final_spins: np.ndarray
	final_energy: float
	history_best_energy: np.ndarray
	accepted_worse_moves: int


def _canonical_move(i: int, j: int) -> Move:
	return (i, j) if i < j else (j, i)


def tabu_search(
	s0: np.ndarray,
	J: np.ndarray,
	h: np.ndarray,
	iterations: int = 500,
	tabu_tenure: int = 12,
	max_no_improve: Optional[int] = None,
) -> TabuResult:
	"""Run swap-based tabu search."""
	s = np.asarray(s0, dtype=float).copy()
	J = np.asarray(J, dtype=float)
	h = np.asarray(h, dtype=float)

	n = len(s)
	if n < 2:
		raise ValueError("Need at least 2 spins")
	if J.shape != (n, n):
		raise ValueError(f"J must have shape ({n}, {n}), got {J.shape}")
	if h.shape != (n,):
		raise ValueError(f"h must have shape ({n},), got {h.shape}")
	if iterations < 1:
		raise ValueError("iterations must be >= 1")
	if tabu_tenure < 1:
		raise ValueError("tabu_tenure must be >= 1")

	current_energy = float(hamiltonian_vectorized(s, J, h))
	best_energy = current_energy
	best_spins = s.copy()

	tabu_until: Dict[Move, int] = {}

	history = np.empty(iterations, dtype=float)
	no_improve_count = 0
	accepted_worse = 0

	for t in range(iterations):
		best_candidate_energy = np.inf
		best_candidate_move: Optional[Move] = None

		for i in range(n - 1):
			for j in range(i + 1, n):
				move = (i, j)

				s[i], s[j] = s[j], s[i]
				cand_energy = float(hamiltonian_vectorized(s, J, h))
				s[i], s[j] = s[j], s[i]

				is_tabu = tabu_until.get(move, -1) >= t
				improves_global = cand_energy < best_energy

				if is_tabu and not improves_global:
					continue

				if cand_energy < best_candidate_energy:
					best_candidate_energy = cand_energy
					best_candidate_move = move

		if best_candidate_move is None:
			tabu_until.clear()
			history[t] = best_energy
			continue

		i, j = best_candidate_move
		s[i], s[j] = s[j], s[i]
		prev_energy = current_energy
		current_energy = best_candidate_energy

		if current_energy > prev_energy:
			accepted_worse += 1

		tabu_until[_canonical_move(i, j)] = t + tabu_tenure

		if current_energy < best_energy:
			best_energy = current_energy
			best_spins = s.copy()
			no_improve_count = 0
		else:
			no_improve_count += 1

		history[t] = best_energy

		if max_no_improve is not None and no_improve_count >= max_no_improve:
			history = history[: t + 1]
			break

	return TabuResult(
		best_spins=best_spins,
		best_energy=float(best_energy),
		final_spins=s.copy(),
		final_energy=float(current_energy),
		history_best_energy=history,
		accepted_worse_moves=accepted_worse,
	)
