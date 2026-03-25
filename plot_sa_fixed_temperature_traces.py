"""Plot energy vs Metropolis swaps for fixed-temperature SA experiment.

Reads ``sa_fixed_temp_traces_size10.csv`` produced by
``sa_fixed_temperature_experiment.py`` and generates plots with
energy on the y-axis and Metropolis swap index t on the x-axis.

Two types of plots are produced:

1. For a chosen instance and the largest steps_max, plot average
   energy vs t for several beta values (to show temperature effect).
2. A contrasting plot where beta is very large (almost frozen) or
   steps_max is very small, illustrating "bad" parameter choices
   (either stuck or not equilibrated).

Requires matplotlib and numpy.

Usage (from project root):

    python plot_sa_fixed_temperature_traces.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT_CSV = Path("sa_fixed_temp_traces_size10.csv")
OUTPUT_DIR = Path("plots")


def load_traces():
    """Load mean energy traces grouped by (instance, beta_index, steps_index)."""

    if not INPUT_CSV.is_file():
        raise SystemExit(
            f"Input CSV '{INPUT_CSV}' not found. "
            "Run sa_fixed_temperature_experiment.py first."
        )

    # Key: (instance, beta_index, steps_index) -> dict with beta, steps_max, arrays
    traces: dict[tuple[int, int, int], dict[str, np.ndarray | float | int]] = {}

    # Temporary storage for energies per key
    energy_by_key: dict[tuple[int, int, int], list[float]] = defaultdict(list)

    with INPUT_CSV.open("r", newline="") as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            inst = int(row["instance"])
            b_idx = int(row["beta_index"])
            s_idx = int(row["steps_index"])
            beta = float(row["beta"])
            steps_max = int(row["steps_max"])
            t = int(row["t"])
            e = float(row["mean_energy"])

            key = (inst, b_idx, s_idx)

            if key not in traces:
                traces[key] = {
                    "instance": inst,
                    "beta_index": b_idx,
                    "beta": beta,
                    "steps_index": s_idx,
                    "steps_max": steps_max,
                }

            energy_by_key[key].append((t, e))

    # Convert lists to sorted arrays by t
    for key, pairs in energy_by_key.items():
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        t_arr = np.array([p[0] for p in pairs_sorted], dtype=int)
        e_arr = np.array([p[1] for p in pairs_sorted], dtype=float)
        traces[key]["t"] = t_arr
        traces[key]["mean_energy"] = e_arr

    return traces


def plot_energy_vs_swaps(traces):
    """Plot energy vs swaps for several betas at largest steps_max."""

    # Choose first instance present
    instances = sorted({info["instance"] for info in traces.values()})
    if not instances:
        raise SystemExit("No traces loaded from CSV.")
    inst = instances[0]

    # Find largest steps_max for that instance
    keys_for_inst = [k for k, info in traces.items() if info["instance"] == inst]
    steps_candidates = {traces[k]["steps_max"] for k in keys_for_inst}
    max_steps = max(steps_candidates)

    # Collect betas for this instance and max_steps
    configs = [
        (k, traces[k])
        for k in keys_for_inst
        if traces[k]["steps_max"] == max_steps
    ]

    # Sort by beta
    configs.sort(key=lambda item: item[1]["beta"])

    # Pick a few representative betas: min, middle, max
    if len(configs) >= 3:
        chosen = [configs[0], configs[len(configs)//2], configs[-1]]
    else:
        chosen = configs

    plt.figure(figsize=(7, 4))
    for _, info in chosen:
        t = info["t"]
        e = info["mean_energy"]
        beta = info["beta"]
        plt.plot(t, e, label=f"beta={beta:.2f}")

    plt.title(f"Fixed-T SA (instance={inst}): energy vs swaps, steps_max={max_steps}")
    plt.xlabel("Metropolis swap index t")
    plt.ylabel("Mean energy E(t)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_bad_parameters(traces):
    """Plot example of too large beta or too few steps ("bad" parameters)."""

    instances = sorted({info["instance"] for info in traces.values()})
    inst = instances[0]
    keys_for_inst = [k for k, info in traces.items() if info["instance"] == inst]

    # Choose smallest steps_max and largest beta as "bad" examples
    steps_candidates = {traces[k]["steps_max"] for k in keys_for_inst}
    min_steps = min(steps_candidates)
    max_steps = max(steps_candidates)

    # For min steps, pick mid-range beta
    configs_min_steps = [
        (k, traces[k])
        for k in keys_for_inst
        if traces[k]["steps_max"] == min_steps
    ]
    configs_min_steps.sort(key=lambda item: item[1]["beta"])
    cfg_short = configs_min_steps[len(configs_min_steps)//2]

    # For max steps, pick largest beta (almost frozen)
    configs_max_steps = [
        (k, traces[k])
        for k in keys_for_inst
        if traces[k]["steps_max"] == max_steps
    ]
    cfg_frozen = max(configs_max_steps, key=lambda item: item[1]["beta"])

    plt.figure(figsize=(7, 4))

    # Short run: not enough steps to equilibrate
    _, info_short = cfg_short
    plt.plot(info_short["t"], info_short["mean_energy"], label=f"short run, steps={min_steps}, beta={info_short['beta']:.2f}")

    # Frozen run: too large beta
    _, info_frozen = cfg_frozen
    plt.plot(info_frozen["t"], info_frozen["mean_energy"], label=f"frozen, steps={max_steps}, beta={info_frozen['beta']:.2f}")

    plt.title(f"Fixed-T SA (instance={inst}): examples of bad parameters")
    plt.xlabel("Metropolis swap index t")
    plt.ylabel("Mean energy E(t)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    traces = load_traces()

    plot_energy_vs_swaps(traces)
    plot_bad_parameters(traces)

    # Save figures
    fig1_path = OUTPUT_DIR / "sa_fixedT_energy_vs_swaps.png"
    fig2_path = OUTPUT_DIR / "sa_fixedT_bad_parameters.png"

    plt.figure(1)
    plt.savefig(fig1_path, dpi=150)

    plt.figure(2)
    plt.savefig(fig2_path, dpi=150)

    print("Saved fixed-temperature SA plots:")
    print(f" - {fig1_path}")
    print(f" - {fig2_path}")


if __name__ == "__main__":
    main()
