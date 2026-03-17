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
