"""Calculation service for running optimization algorithms on GUI parameters."""

from __future__ import annotations

import time
from typing import Dict

import numpy as np

from hemiltonian_energy import hamiltonian_vectorized
from pure_simulated_annealing import pure_simulated_annealing
from genetic_algorithm import genetic_algorithm
from tabu_search import tabu_search
