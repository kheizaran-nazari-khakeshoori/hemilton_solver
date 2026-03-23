"""Generate QAP instances as (F, D) matrices saved in .npz files.

Each instance file is named "{size}-{instance}.npz" and contains two
arrays with keys "F" (flow matrix) and "D" (distance matrix).

Sizes: [10, 15, 20, 25, 50, 75, 100]
Instances per size: 100

Run this script once to populate the "instances/" directory:

    python generate_instances.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


SIZES = [10, 15, 20, 25, 50, 75, 100]
INSTANCES_PER_SIZE = 100
OUTPUT_DIR = Path("instances")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    for N in SIZES:
        # Use a size-dependent seed so runs are reproducible but different per N.
        rng = np.random.default_rng(42 + N)

        for k in range(1, INSTANCES_PER_SIZE + 1):
            # Simple random integer matrices; you can adjust ranges if needed.
            F = rng.integers(0, 10, size=(N, N)).astype(float)
            D = rng.integers(0, 10, size=(N, N)).astype(float)

            # Zero out diagonals (no self-flow / self-distance) as in demo code.
            np.fill_diagonal(F, 0)
            np.fill_diagonal(D, 0)

            filename = OUTPUT_DIR / f"{N}-{k}.npz"
            np.savez(filename, F=F, D=D)
            # Optional: print progress; comment out if noisy.
            # print(f"Saved {filename}")


if __name__ == "__main__":
    main()
