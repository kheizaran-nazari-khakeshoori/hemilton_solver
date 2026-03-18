"""Genetic Algorithm for Ising energy minimization via permutations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from hemiltonian_energy import hamiltonian_vectorized
