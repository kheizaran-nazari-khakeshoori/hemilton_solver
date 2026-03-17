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
