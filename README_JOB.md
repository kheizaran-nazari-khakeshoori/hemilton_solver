# Hamiltonian Solver — Job-Level Project Summary

## Project overview
This repository implements and evaluates optimization methods for Quadratic Assignment Problem (QAP) style instances, with a focus on Simulated Annealing (SA) and exact ground-state comparison for small problems.

## What the project does
- Generates random QAP-style instances containing two matrices (`F` and `D`).
- Computes exact ground-state solutions for small instances by exhaustive search.
- Runs Simulated Annealing to solve the same instances and compare performance.
- Aggregates results across many instances and problem sizes.
- Produces visualizations that show how SA performance scales and where it matches or misses the exact solution.

## Why it matters
- Demonstrates strong understanding of combinatorial optimization and metaheuristics.
- Shows ability to compare heuristic algorithms against exact baselines.
- Provides reproducible experiment pipelines for metrics, analysis, and plotting.

## Key capabilities
- `generate_instances.py`: constructs instance datasets used for experiments.
- `compare_sa_to_ground_state.py`: compares SA results against exhaustive optimal solutions for a single instance.
- `run_size10_groundstate_experiment.py`: evaluates SA across 100 small instances and compares to exact ground state.
- `run_batch_experiments.py`: runs experiments across multiple sizes, capturing SA scaling behavior.
- `plot_*`: generates plots for experiment results and algorithm analysis.

## Data and outputs
- `instances/`: stores generated problem instances in `.npz` format.
- `batch_results.csv`: stores SA results for a range of problem sizes.
- `size10_ground_vs_sa.csv`: stores exact vs SA comparison data for size-10 instances.
- `plots/`: stores generated charts and figures.

## Technical skills demonstrated
- Python scripting and experiment orchestration.
- Numerical optimization, energy minimization, and search heuristics.
- Statistical comparison of algorithmic performance.
- Data processing, CSV reporting, and visualization.

## How to run
1. Create instances:
   ```bash
   python generate_instances.py
   ```
2. Compare SA vs exact ground state for a single size-10 instance:
   ```bash
   python compare_sa_to_ground_state.py --size 10 --instance 1
   ```
3. Run the full size-10 ground-state experiment:
   ```bash
   python run_size10_groundstate_experiment.py
   ```
4. Run broader SA batch experiments across sizes:
   ```bash
   python run_batch_experiments.py
   ```
5. Generate plots after experiments:
   ```bash
   python plot_size10_groundstate_results.py
   python plot_batch_sa_performance.py
   ```

## Summary for hiring managers
This repository is a concise research-focused project that showcases the ability to design, implement, and evaluate optimization algorithms in Python. It is valuable evidence of practical experience with algorithm comparison, performance tracking, and experiment-driven analysis.
