"""Tabu Search for Ising energy minimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from hemiltonian_energy import hamiltonian_vectorized


Move = Tuple[int, int]
