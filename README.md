This project explores different optimization approaches (currently Simulated Annealing)
for Quadratic Assignment Problem (QAP) style instances.

## Exact ground state vs Simulated Annealing

For small instances (e.g. N = 10) you can compute the exact
"ground state" (global minimum energy) by enumerating all
permutations and then compare that to the result of the
Simulated Annealing (SA) algorithm.

From the project root, after generating instances with
`python generate_instances.py`, run for example:

	python compare_sa_to_ground_state.py --size 10 --instance 1

This will:

- Load `instances/10-1.npz` (matrices F and D)
- Compute the exact ground state using exhaustive search
- Run SA on the same instance
- Print both energies and report whether SA reached the ground state

The exact solver is only intended for very small N (default N ≤ 10),
because the number of permutations grows as N!.