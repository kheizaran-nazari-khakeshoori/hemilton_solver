"""Pure Simulated Annealing for Ising spin swaps."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from hemiltonian_energy import hamiltonian_vectorized


def _coupling(J: np.ndarray, a: int, b: int) -> float:
	"""Return J_ab assuming couplings are stored in upper-triangle form."""
	if a == b:
		return 0.0
	return float(J[a, b] if a < b else J[b, a])


def delta_h_for_swap(s: np.ndarray, J: np.ndarray, h: np.ndarray, i: int, j: int) -> float:
	"""Compute DeltaH = H(after swapping i,j) - H(before) in O(N)."""
	if i == j:
		return 0.0

	si = float(s[i])
	sj = float(s[j])

	if si == sj:
		return 0.0

	delta = 0.0

	old_field = -(h[i] * si + h[j] * sj)
	new_field = -(h[i] * sj + h[j] * si)
	delta += float(new_field - old_field)

	for k in range(len(s)):
		if k == i or k == j:
			continue

		sik = float(s[k])
		Jik = _coupling(J, i, k)
		Jjk = _coupling(J, j, k)

		old_pair = -(Jik * si * sik + Jjk * sj * sik)
		new_pair = -(Jik * sj * sik + Jjk * si * sik)
		delta += float(new_pair - old_pair)

	return delta


@dataclass
class AnnealResult:
	best_spins: np.ndarray
	best_energy: float
	final_spins: np.ndarray
	final_energy: float
	accepted_moves: int
	attempted_moves: int
	energy_trace: np.ndarray


def pure_simulated_annealing(
	s0: np.ndarray,
	J: np.ndarray,
	h: np.ndarray,
	initial_temp: float = 5.0,
	cooling_rate: float = 0.995,
	steps: int = 10_000,
	seed: Optional[int] = None,
) -> AnnealResult:
	"""Run pair-swap simulated annealing with one random proposal per step."""
	if steps <= 0:
		raise ValueError("steps must be > 0")
	if initial_temp <= 0:
		raise ValueError("initial_temp must be > 0")
	if not (0 < cooling_rate <= 1):
		raise ValueError("cooling_rate must be in (0, 1]")

	s = np.asarray(s0, dtype=float).copy()
	J = np.asarray(J, dtype=float)
	h = np.asarray(h, dtype=float)

	n = len(s)
	if n < 2:
		raise ValueError("Need at least 2 spins for pair swaps")
	if J.shape != (n, n):
		raise ValueError(f"J must have shape ({n}, {n}), got {J.shape}")
	if h.shape != (n,):
		raise ValueError(f"h must have shape ({n},), got {h.shape}")

	rng = np.random.default_rng(seed)

	current_energy = float(hamiltonian_vectorized(s, J, h))
	best_energy = current_energy
	best_spins = s.copy()

	accepted = 0
	temp = float(initial_temp)
	trace = np.empty(steps, dtype=float)

	for t in range(steps):
		i = int(rng.integers(0, n))
		j = int(rng.integers(0, n - 1))
		if j >= i:
			j += 1

		delta_h = delta_h_for_swap(s, J, h, i, j)

		if delta_h <= 0:
			accept = True
		else:
			accept_prob = math.exp(-delta_h / max(temp, 1e-12))
			accept = bool(rng.random() < accept_prob)

		if accept:
			s[i], s[j] = s[j], s[i]
			current_energy += delta_h
			accepted += 1

			if current_energy < best_energy:
				best_energy = current_energy
				best_spins = s.copy()

		trace[t] = current_energy
		temp *= cooling_rate

	return AnnealResult(
		best_spins=best_spins,
		best_energy=float(best_energy),
		final_spins=s.copy(),
		final_energy=float(current_energy),
		accepted_moves=accepted,
		attempted_moves=steps,
		energy_trace=trace,
	)


if __name__ == "__main__":
	N = 12
	rng = np.random.default_rng(7)
	s0 = rng.choice([-1.0, 1.0], size=N)

	J = np.zeros((N, N), dtype=float)
	for idx in range(N - 1):
		J[idx, idx + 1] = 1.0

	h = np.zeros(N, dtype=float)

	result = pure_simulated_annealing(
		s0=s0,
		J=J,
		h=h,
		initial_temp=3.0,
		cooling_rate=0.999,
		steps=20_000,
		seed=42,
	)

	print("Pure Simulated Annealing (random-pair proposals)")
	print(f"Initial energy: {hamiltonian_vectorized(s0, J, h):.6f}")
	print(f"Final energy  : {result.final_energy:.6f}")
	print(f"Best energy   : {result.best_energy:.6f}")
	print(f"Accepted moves: {result.accepted_moves}/{result.attempted_moves}")
