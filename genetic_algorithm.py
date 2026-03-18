"""Genetic Algorithm for Ising energy minimization via permutations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from hemiltonian_energy import hamiltonian_vectorized


@dataclass
class GAResult:
	best_permutation: np.ndarray
	best_spins: np.ndarray
	best_energy: float
	history_best_energy: np.ndarray


def _energy_of_permutation(
	permutation: np.ndarray,
	base_spins: np.ndarray,
	J: np.ndarray,
	h: np.ndarray,
) -> float:
	candidate = base_spins[permutation]
	return float(hamiltonian_vectorized(candidate, J, h))
