"""Exact ground-state search for small QAP instances.

This module provides a brute-force solver that enumerates all
permutations using ``itertools.permutations`` and returns the
permutation with minimum QAP cost (the "ground state").

It is only practical for very small problem sizes, because the
number of permutations grows as N!.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Tuple

import numpy as np

from hemiltonian_energy import qap_cost


@dataclass
class GroundStateResult:
	"""Result of an exact ground-state search.

	Attributes
	----------
	best_permutation:
		Permutation (shape (N,)) that achieves the minimum QAP cost.
	best_cost:
		Minimum QAP cost (ground-state energy).
	evaluations:
		Number of permutations evaluated (should equal N!).
	"""

	best_permutation: np.ndarray
	best_cost: float
	evaluations: int


def brute_force_ground_state(F: np.ndarray, D: np.ndarray, max_n: int = 10) -> GroundStateResult:
	"""Exhaustively search all permutations and return the ground state.

	Parameters
	----------
	F, D:
		Flow and distance matrices of shape (N, N).
	max_n:
		Maximum N to allow. For N > max_n a ValueError is raised to
		protect against accidental calls on large instances.

	Returns
	-------
	GroundStateResult
		The best permutation found, its cost, and evaluation count.
	"""

	F = np.asarray(F, dtype=float)
	D = np.asarray(D, dtype=float)

	if F.shape != D.shape or F.shape[0] != F.shape[1]:
		raise ValueError(f"F and D must be square with the same shape, got F={F.shape}, D={D.shape}")

	n = F.shape[0]
	if n == 0:
		raise ValueError("Empty matrices are not allowed")
	if n > max_n:
		raise ValueError(f"Brute-force search only supported for n <= {max_n}, got n={n}")

	best_cost = float("inf")
	best_perm: np.ndarray | None = None
	evaluations = 0

	for perm in permutations(range(n)):
		P = np.fromiter(perm, dtype=int, count=n)
		cost = qap_cost(F, D, P)
		evaluations += 1

		if cost < best_cost:
			best_cost = float(cost)
			best_perm = P.copy()

	if best_perm is None:
		raise RuntimeError("No permutation evaluated; this should not happen for n >= 1")

	return GroundStateResult(
		best_permutation=best_perm,
		best_cost=best_cost,
		evaluations=evaluations,
	)
