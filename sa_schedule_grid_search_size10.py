"""Grid search over SA schedules (beta_final, N_steps) for size-10 QAP.

This script implements the experiment described in Andrea's email:

- Fix problem size N = 10.
- Select a few representative instances (here: three out of the 100).
- Choose 10 values of final inverse temperature beta between BETA_MIN and
  BETA_MAX.
- Choose 10 values of total Metropolis swaps N between STEPS_MIN and
  STEPS_MAX.
- For each pair (beta_final, N_steps) and each chosen instance, run many
  stochastic SA trials starting at very high temperature (beta ~ 0) and
  cooling down to beta_final over N_steps swaps.
- For each configuration, record:
    * average best cost reached,
    * average final cost,
    * probability of solving (reaching the exact ground state).

Results are written to a CSV file which can be visualised as 2D
heatmaps (x-axis = N_steps, y-axis = beta_final, colour = metric).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from pure_simulated_annealing import pure_simulated_annealing
from hemiltonian_energy import qap_cost
from exact_qap_ground_state import brute_force_ground_state


INSTANCES_DIR = Path("instances")
SIZE = 10

# For the schedule grid search, use three representative instances.
# These can be adjusted if desired (e.g., randomly chosen with a
# fixed seed), but using the first three keeps things deterministic
# and keeps the runtime manageable while tuning hyperparameters.
INSTANCE_IDS = [1, 2, 3]

# Grid over final inverse temperature beta and total steps N.
# After the initial coarse scan (0.1--2.0) indicated that the best
# performance occurs at the smallest tested beta, we refine the range
# to smaller values as suggested by the supervisor.
BETA_MIN = 0.01
BETA_MAX = 0.11
N_BETA_VALUES = 10

STEPS_MIN = 100
STEPS_MAX = 3000
N_STEPS_VALUES = 10

# Number of independent SA trials per (instance, beta_final, N_steps).
RUNS_PER_CONFIG = 100

OUTPUT_CSV = Path("sa_schedule_grid_size10.csv")


def _compute_schedule_parameters(beta_final: float, steps: int, initial_temp: float = 100.0) -> float:
    """Return cooling_rate for geometric schedule reaching beta_final.

    We use the existing pure_simulated_annealing implementation, which
    assumes a geometric cooling schedule:

        T_{t+1} = cooling_rate * T_t

    with T_0 = initial_temp. The inverse temperature is beta_t = 1 / T_t,
    so after `steps` iterations we have approximately

        beta_final ≈ 1 / (initial_temp * cooling_rate**steps).

    Given a target beta_final > 0, we solve for cooling_rate:

        T_final = 1 / beta_final
        cooling_rate = (T_final / initial_temp) ** (1 / steps).

    Choosing a large initial_temp makes the starting beta close to zero,
    which approximates the "beta = 0" high-temperature start in the email.
    """

    if beta_final <= 0:
        raise ValueError("beta_final must be > 0")
    if steps <= 0:
        raise ValueError("steps must be > 0")

    T_final = 1.0 / float(beta_final)
    cooling_rate = (T_final / float(initial_temp)) ** (1.0 / float(steps))
    return float(cooling_rate)


def run_grid_search() -> None:
    if not INSTANCES_DIR.is_dir():
        raise SystemExit(
            f"Instances directory '{INSTANCES_DIR}' not found. "
            "Run generate_instances.py first."
        )

    betas = np.linspace(BETA_MIN, BETA_MAX, N_BETA_VALUES)
    step_values = np.linspace(STEPS_MIN, STEPS_MAX, N_STEPS_VALUES, dtype=int)

    rng = np.random.default_rng(12345)

    with OUTPUT_CSV.open("w", newline="") as f_out:
        fieldnames = [
            "size",
            "instance",
            "beta_index",
            "beta_final",
            "steps_index",
            "steps",
            "avg_best_cost",
            "avg_final_cost",
            "prob_solved",
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for inst_id in INSTANCE_IDS:
            fname = INSTANCES_DIR / f"{SIZE}-{inst_id}.npz"
            if not fname.is_file():
                raise SystemExit(f"Missing instance file: {fname}. Generate it first.")

            data = np.load(fname)
            if "F" not in data or "D" not in data:
                raise SystemExit(f"File {fname} must contain 'F' and 'D' arrays.")

            F = np.asarray(data["F"], dtype=float)
            D = np.asarray(data["D"], dtype=float)

            if F.shape != (SIZE, SIZE) or D.shape != (SIZE, SIZE):
                raise SystemExit(
                    f"File {fname} has inconsistent shapes: F={F.shape}, D={D.shape}."
                )

            # Compute exact ground-state cost once for this instance.
            gs = brute_force_ground_state(F, D, max_n=SIZE)
            ground_cost = float(gs.best_cost)
            print(
                f"Instance size={SIZE}, id={inst_id}: ground cost = {ground_cost:.6f}"
            )

            for b_idx, beta_final in enumerate(betas):
                for s_idx, steps in enumerate(step_values):
                    cooling_rate = _compute_schedule_parameters(
                        beta_final=beta_final,
                        steps=int(steps),
                        initial_temp=100.0,
                    )

                    best_sum = 0.0
                    final_sum = 0.0
                    solved_count = 0

                    for run in range(RUNS_PER_CONFIG):
                        # Random initial permutation for each trial.
                        p0 = rng.permutation(SIZE)

                        seed = (
                            inst_id * 1_000_000
                            + b_idx * 10_000
                            + s_idx * 100
                            + run
                        )

                        result = pure_simulated_annealing(
                            p0=p0,
                            F=F,
                            D=D,
                            initial_temp=100.0,
                            cooling_rate=cooling_rate,
                            steps=int(steps),
                            seed=int(seed),
                        )

                        best_cost = float(result.best_cost)
                        final_cost = float(result.final_cost)

                        best_sum += best_cost
                        final_sum += final_cost

                        if best_cost <= ground_cost + 1e-9:
                            solved_count += 1

                    avg_best = best_sum / float(RUNS_PER_CONFIG)
                    avg_final = final_sum / float(RUNS_PER_CONFIG)
                    prob_solved = solved_count / float(RUNS_PER_CONFIG)

                    writer.writerow(
                        {
                            "size": SIZE,
                            "instance": inst_id,
                            "beta_index": b_idx,
                            "beta_final": float(beta_final),
                            "steps_index": s_idx,
                            "steps": int(steps),
                            "avg_best_cost": float(avg_best),
                            "avg_final_cost": float(avg_final),
                            "prob_solved": float(prob_solved),
                        }
                    )

    print(f"Wrote SA schedule grid search results to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_grid_search()
