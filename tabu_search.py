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
