"""Genetic Algorithm for QAP minimization via permutations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from hemiltonian_energy import qap_cost


@dataclass
class GAResult:
	best_permutation: np.ndarray
	best_cost: float
	history_best_cost: np.ndarray


def _energy_of_permutation(
	permutation: np.ndarray,
	F: np.ndarray,
	D: np.ndarray,
) -> float:
	return float(qap_cost(F, D, permutation))


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
	p0: np.ndarray,
	F: np.ndarray,
	D: np.ndarray,
	population_size: int = 80,
	generations: int = 300,
	elite_fraction: float = 0.1,
	mutation_rate: float = 0.2,
	tournament_k: int = 3,
	seed: Optional[int] = None,
) -> GAResult:
	"""Minimize QAP cost by evolving assignment permutations."""
	p0 = np.asarray(p0, dtype=int)
	F = np.asarray(F, dtype=float)
	D = np.asarray(D, dtype=float)

	n = len(p0)
	if n < 2:
		raise ValueError("Need at least 2 facilities")
	if F.shape != (n, n):
		raise ValueError(f"F must have shape ({n}, {n}), got {F.shape}")
	if D.shape != (n, n):
		raise ValueError(f"D must have shape ({n}, {n}), got {D.shape}")
	if set(p0.tolist()) != set(range(n)):
		raise ValueError("p0 must be a valid permutation")
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
	population[0] = p0.copy()
	fitness = np.array([
		_energy_of_permutation(p, F, D) for p in population
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
		fitness = np.array([_energy_of_permutation(p, F, D) for p in population])

	best_idx = int(np.argmin(fitness))
	best_perm = population[best_idx].copy()
	best_cost = float(fitness[best_idx])

	return GAResult(
		best_permutation=best_perm,
		best_cost=best_cost,
		history_best_cost=history,
	)


if __name__ == "__main__":
	N = 20
	rng = np.random.default_rng(12)
	p0 = rng.permutation(N)
	F = rng.integers(0, 10, size=(N, N)).astype(float)
	D = rng.integers(0, 10, size=(N, N)).astype(float)
	np.fill_diagonal(F, 0)
	np.fill_diagonal(D, 0)

	initial_cost = qap_cost(F, D, p0)
	result = genetic_algorithm(
		p0=p0,
		F=F,
		D=D,
		population_size=100,
		generations=350,
		elite_fraction=0.12,
		mutation_rate=0.25,
		tournament_k=4,
		seed=77,
	)

	print("Genetic Algorithm (QAP)")
	print(f"Initial cost : {initial_cost:.6f}")
	print(f"Best cost    : {result.best_cost:.6f}")
	print(f"Best permutation head: {result.best_permutation[:10].tolist()}")
