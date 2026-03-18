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


def genetic_algorithm(
	base_spins: np.ndarray,
	J: np.ndarray,
	h: np.ndarray,
	population_size: int = 80,
	generations: int = 300,
	elite_fraction: float = 0.1,
	mutation_rate: float = 0.2,
	tournament_k: int = 3,
	seed: Optional[int] = None,
) -> GAResult:
	"""Minimize Ising energy by evolving permutations of a base spin vector."""
	base_spins = np.asarray(base_spins, dtype=float)
	J = np.asarray(J, dtype=float)
	h = np.asarray(h, dtype=float)

	n = len(base_spins)
	if n < 2:
		raise ValueError("Need at least 2 spins")
	if J.shape != (n, n):
		raise ValueError(f"J must have shape ({n}, {n}), got {J.shape}")
	if h.shape != (n,):
		raise ValueError(f"h must have shape ({n},), got {h.shape}")
	if population_size < 4:
		raise ValueError("population_size must be >= 4")
	if generations < 1:
		raise ValueError("generations must be >= 1")
	if not (0 <= mutation_rate <= 1):
		raise ValueError("mutation_rate must be in [0, 1]")
	if not (0 < elite_fraction < 1):
		raise ValueError("elite_fraction must be in (0, 1)")

	rng = np.random.default_rng(seed)
	elite_count = max(1, int(population_size * elite_fraction))

	population = np.array([rng.permutation(n) for _ in range(population_size)], dtype=int)
	fitness = np.array([
		_energy_of_permutation(p, base_spins, J, h) for p in population
	])

	history = np.empty(generations, dtype=float)

	for g in range(generations):
		order = np.argsort(fitness)
		population = population[order]
		fitness = fitness[order]
		history[g] = fitness[0]

		elites = population[:elite_count].copy()
		new_population = [e.copy() for e in elites]
		while len(new_population) < population_size:
			dad_idx = _tournament_select(fitness, tournament_k, rng)
			mom_idx = _tournament_select(fitness, tournament_k, rng)
			dad = population[dad_idx]
			mom = population[mom_idx]

			child = _crossover_unique(dad, mom, rng)
			if rng.random() < mutation_rate:
				_mutate_swap(child, rng)
			new_population.append(child)

		population = np.array(new_population, dtype=int)
		fitness = np.array([
			_energy_of_permutation(p, base_spins, J, h) for p in population
		])

	best_idx = int(np.argmin(fitness))
	best_perm = population[best_idx].copy()
	best_spins = base_spins[best_perm]
	best_energy = float(fitness[best_idx])

	return GAResult(
		best_permutation=best_perm,
		best_spins=best_spins,
		best_energy=best_energy,
		history_best_energy=history,
	)
