This project explores different optimization approaches (currently Simulated Annealing)
for Quadratic Assignment Problem (QAP) style instances.

## Exact ground state vs Simulated Annealing

For small instances (e.g. N = 10) you can compute the exact
"ground state" (global minimum energy) by enumerating all
permutations and then compare that to the result of the
Simulated Annealing (SA) algorithm.

From the project root, after generating instances with
`python generate_instances.py`, you have two main workflows:

### 1) Single-instance comparison

Run for example:

	python compare_sa_to_ground_state.py --size 10 --instance 1

This will:

- Load `instances/10-1.npz` (matrices F and D)
- Compute the exact ground state using exhaustive search
- Run SA on the same instance
- Print both energies and report whether SA reached the ground state

### 2) All 100 instances of size 10 + plots

First, run the batch experiment to compare SA with the exact ground
state on all 100 size-10 instances:

	python run_size10_groundstate_experiment.py

This writes a CSV `size10_ground_vs_sa.csv` that contains, for each
instance, the exact ground-state cost, SA costs, the gap, and a
success flag.

Then, to generate basic graphs (requires `matplotlib`):

	python plot_size10_groundstate_results.py

This creates a `plots/` folder with PNG figures showing the gap
`SA best cost - ground-state cost` per instance and its distribution,
and prints summary statistics including the SA success rate.

The exact solver is only intended for very small N (default N ≤ 10),
because the number of permutations grows as N!.