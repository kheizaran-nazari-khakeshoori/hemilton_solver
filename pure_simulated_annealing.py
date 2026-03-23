"""Pure Simulated Annealing for QAP permutation swaps."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from hemiltonian_energy import qap_cost


def delta_swap(F: np.ndarray, D: np.ndarray, P: np.ndarray, a: int, b: int) -> float:
	"""Compute QAP cost delta for swapping facilities a and b in permutation P."""
	N = len(P)
	ra, rb = int(P[a]), int(P[b])
	delta = 0.0

	for t in range(N):
		if t == a or t == b:
			continue
		rt = int(P[t])

		delta += (
			F[a, t] * (D[rb, rt] - D[ra, rt])
			+ F[t, a] * (D[rt, rb] - D[rt, ra])
			+ F[b, t] * (D[ra, rt] - D[rb, rt])
			+ F[t, b] * (D[rt, ra] - D[rt, rb])
		)

	delta += F[a, b] * (D[rb, ra] - D[ra, rb])
	delta += F[b, a] * (D[ra, rb] - D[rb, ra])

	delta += F[a, a] * (D[rb, rb] - D[ra, ra])
	delta += F[b, b] * (D[ra, ra] - D[rb, rb])

	return float(delta)


@dataclass
class AnnealResult:
	best_permutation: np.ndarray
	best_cost: float
	final_permutation: np.ndarray
	final_cost: float
	accepted_moves: int
	attempted_moves: int
	cost_trace: np.ndarray


def pure_simulated_annealing(
	p0: np.ndarray,
	F: np.ndarray,
	D: np.ndarray,
	initial_temp: float = 5.0,
	cooling_rate: float = 0.995,
	steps: int = 10_000,
	seed: Optional[int] = None,
) -> AnnealResult:
	"""Run pair-swap simulated annealing over QAP permutations."""
	if steps <= 0:
		raise ValueError("steps must be > 0")
	if initial_temp <= 0:
		raise ValueError("initial_temp must be > 0")
	if not (0 < cooling_rate <= 1):
		raise ValueError("cooling_rate must be in (0, 1]")

	P = np.asarray(p0, dtype=int).copy()
	F = np.asarray(F, dtype=float)
	D = np.asarray(D, dtype=float)

	n = len(P)
	if n < 2:
		raise ValueError("Need at least 2 facilities for swaps")
	if F.shape != (n, n):
		raise ValueError(f"F must have shape ({n}, {n}), got {F.shape}")
	if D.shape != (n, n):
		raise ValueError(f"D must have shape ({n}, {n}), got {D.shape}")
	if set(P.tolist()) != set(range(n)):
		raise ValueError("p0 must be a valid permutation")

	rng = np.random.default_rng(seed)

	current_cost = float(qap_cost(F, D, P))
	best_cost = current_cost
	best_perm = P.copy()

	accepted = 0
	temp = float(initial_temp)
	trace = np.empty(steps, dtype=float)

	for t in range(steps):
		i = int(rng.integers(0, n))
		j = int(rng.integers(0, n - 1))
		if j >= i:
			j += 1

		delta = delta_swap(F, D, P, i, j)

		if delta <= 0:
			accept = True
		else:
			accept_prob = math.exp(-delta / max(temp, 1e-12))
			accept = bool(rng.random() < accept_prob)

		if accept:
			P[i], P[j] = P[j], P[i]
			current_cost += delta
			accepted += 1

			if current_cost < best_cost:
				best_cost = current_cost
				best_perm = P.copy()

		trace[t] = current_cost
		temp *= cooling_rate

	return AnnealResult(
		best_permutation=best_perm,
		best_cost=float(best_cost),
		final_permutation=P.copy(),
		final_cost=float(current_cost),
		accepted_moves=accepted,
		attempted_moves=steps,
		cost_trace=trace,
	)


if __name__ == "__main__":
	N = 12
	rng = np.random.default_rng(7)
	p0 = rng.permutation(N)
	F = rng.integers(0, 10, size=(N, N)).astype(float)
	D = rng.integers(0, 10, size=(N, N)).astype(float)
	np.fill_diagonal(F, 0)
	np.fill_diagonal(D, 0)

	result = pure_simulated_annealing(
		p0=p0,
		F=F,
		D=D,
		initial_temp=3.0,
		cooling_rate=0.999,
		steps=20_000,
		seed=42,
	)

	print("Pure Simulated Annealing (QAP)")
	print(f"Initial cost: {qap_cost(F, D, p0):.6f}")
	print(f"Final cost  : {result.final_cost:.6f}")
	print(f"Best cost   : {result.best_cost:.6f}")
	print(f"Accepted moves: {result.accepted_moves}/{result.attempted_moves}")
