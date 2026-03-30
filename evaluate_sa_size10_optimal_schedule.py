"""Evaluate SA on all size-10 instances using an "optimal" schedule.

This script implements Andrea's point 3 for size 10:

- Take the schedule grid search results from sa_schedule_grid_search_size10.py.
- For each (beta_final, steps) pair, average the probability of solving
  across the three selected instances.
- Select the pair (beta_final*, steps*) with the highest average
  probability of solving.
- Using that fixed schedule, run SA on *all* 100 size-10 instances.
  For each instance, perform many trials and compute:
    * average best cost,
    * average final cost,
    * probability of solving.
- At the end, report the overall average probability of solving across
  all instances as the main summary number.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from pure_simulated_annealing import pure_simulated_annealing
from exact_qap_ground_state import brute_force_ground_state


INSTANCES_DIR = Path("instances")
SIZE = 10
INSTANCES_PER_SIZE = 100

GRID_CSV = Path("sa_schedule_grid_size10.csv")
OUTPUT_CSV = Path("sa_size10_optimal_schedule_results.csv")

# Number of SA trials per instance when evaluating the chosen schedule.
RUNS_PER_INSTANCE = 100

INITIAL_TEMP = 100.0


def _compute_cooling_rate(beta_final: float, steps: int, initial_temp: float = INITIAL_TEMP) -> float:
    """Return cooling_rate for a geometric schedule reaching beta_final.

    This mirrors the logic used in sa_schedule_grid_search_size10.py.
    """

    if beta_final <= 0:
        raise ValueError("beta_final must be > 0")
    if steps <= 0:
        raise ValueError("steps must be > 0")

    T_final = 1.0 / float(beta_final)
    cooling_rate = (T_final / float(initial_temp)) ** (1.0 / float(steps))
    return float(cooling_rate)


def _select_optimal_schedule_from_grid() -> tuple[float, int]:
    """Select (beta_final, steps) that maximizes mean prob_solved over instances.

    Reads the grid search CSV and, for each (beta_final, steps) pair,
    computes the average probability of solving across all instances
    present in the file (expected: 3 instances). Returns the pair with
    the highest average probability.
    """

    if not GRID_CSV.is_file():
        raise SystemExit(
            f"Grid search CSV '{GRID_CSV}' not found. Run sa_schedule_grid_search_size10.py first."
        )

    df = pd.read_csv(GRID_CSV)

    # Group by (beta_final, steps) and average prob_solved over instances.
    grouped = (
        df.groupby(["beta_final", "steps"], as_index=False)["prob_solved"]
        .mean()
        .rename(columns={"prob_solved": "mean_prob_solved"})
    )

    # Find the row with maximum mean_prob_solved.
    best_row = grouped.loc[grouped["mean_prob_solved"].idxmax()]
    beta_opt = float(best_row["beta_final"])
    steps_opt = int(best_row["steps"])
    best_prob = float(best_row["mean_prob_solved"])

    print(
        "Selected optimal schedule from grid: "
        f"beta_final={beta_opt:.6f}, steps={steps_opt}, "
        f"mean_prob_solved_over_3_instances={best_prob:.4f}"
    )

    return beta_opt, steps_opt


def evaluate_optimal_schedule() -> None:
    if not INSTANCES_DIR.is_dir():
        raise SystemExit(
            f"Instances directory '{INSTANCES_DIR}' not found. "
            "Run generate_instances.py first."
        )

    beta_opt, steps_opt = _select_optimal_schedule_from_grid()
    cooling_rate = _compute_cooling_rate(beta_final=beta_opt, steps=steps_opt)

    rng = np.random.default_rng(98765)

    with OUTPUT_CSV.open("w", newline="") as f_out:
        fieldnames = [
            "size",
            "instance",
            "beta_final",
            "steps",
            "avg_best_cost",
            "avg_final_cost",
            "prob_solved",
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        total_trials = 0
        total_solved = 0

        for inst_id in range(1, INSTANCES_PER_SIZE + 1):
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

            gs = brute_force_ground_state(F, D, max_n=SIZE)
            ground_cost = float(gs.best_cost)

            best_sum = 0.0
            final_sum = 0.0
            solved_count = 0

            for run in range(RUNS_PER_INSTANCE):
                p0 = rng.permutation(SIZE)

                seed = inst_id * 1_000_000 + run

                result = pure_simulated_annealing(
                    p0=p0,
                    F=F,
                    D=D,
                    initial_temp=INITIAL_TEMP,
                    cooling_rate=cooling_rate,
                    steps=steps_opt,
                    seed=int(seed),
                )

                best_cost = float(result.best_cost)
                final_cost = float(result.final_cost)

                best_sum += best_cost
                final_sum += final_cost

                if best_cost <= ground_cost + 1e-9:
                    solved_count += 1

            avg_best = best_sum / float(RUNS_PER_INSTANCE)
            avg_final = final_sum / float(RUNS_PER_INSTANCE)
            prob_solved = solved_count / float(RUNS_PER_INSTANCE)

            total_trials += RUNS_PER_INSTANCE
            total_solved += solved_count

            writer.writerow(
                {
                    "size": SIZE,
                    "instance": inst_id,
                    "beta_final": float(beta_opt),
                    "steps": int(steps_opt),
                    "avg_best_cost": float(avg_best),
                    "avg_final_cost": float(avg_final),
                    "prob_solved": float(prob_solved),
                }
            )

            print(
                f"Instance {inst_id:3d}: prob_solved={prob_solved:.4f}, "
                f"avg_best_cost={avg_best:.3f}, avg_final_cost={avg_final:.3f}"
            )

    overall_prob = total_solved / float(total_trials)
    print(
        "\nOverall average probability of solving across all "
        f"{INSTANCES_PER_SIZE} instances = {overall_prob:.4f}"
    )
    print(f"Per-instance results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    evaluate_optimal_schedule()
