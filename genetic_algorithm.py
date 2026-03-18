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


def _tournament_select(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
	contenders = rng.integers(0, len(fitness), size=k)
	return int(contenders[np.argmin(fitness[contenders])])


def _crossover_unique(dad: np.ndarray, mom: np.ndarray, rng: np.random.Generator) -> np.ndarray:
	"""Half-from-dad + fill-from-mom while preserving uniqueness."""
	n = len(dad)
	cut = n // 2

	child = np.full(n, -1, dtype=int)
	child[:cut] = dad[:cut]

	used = set(int(x) for x in child[:cut])
	write_idx = cut
	for gene in mom:
		g = int(gene)
		if g not in used:
			child[write_idx] = g
			used.add(g)
			write_idx += 1
			if write_idx == n:
				break

	return child


def _mutate_swap(chromosome: np.ndarray, rng: np.random.Generator) -> None:
	"""In-place mutation: swap two random positions."""
	n = len(chromosome)
	i = int(rng.integers(0, n))
	j = int(rng.integers(0, n - 1))
	if j >= i:
		j += 1
	chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
