"""Pure Simulated Annealing for Ising spin swaps."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from hemiltonian_energy import hamiltonian_vectorized
