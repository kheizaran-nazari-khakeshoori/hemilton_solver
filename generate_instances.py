"""Generate QAP instances as (F, D) matrices saved in .npz files.

Each instance file is named "{size}-{instance}.npz" and contains two
arrays with keys "F" (flow matrix) and "D" (distance matrix).

Sizes: [10, 15, 20, 25, 50, 75, 100]
Instances per size: 100

F is a synthetic symmetric flow matrix (non-negative, zero diagonal).
D is a Euclidean distance matrix computed from random 2D coordinates.

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
            # Generate symmetric, non-negative flow matrix with zero diagonal.
            F_upper = rng.integers(0, 10, size=(N, N)).astype(float)
            F = np.triu(F_upper, k=1)
            F = F + F.T
            np.fill_diagonal(F, 0.0)

            # Generate Euclidean distance matrix D from random points in a square.
            # Sample N points (x_i, y_i) uniformly in [0, 1] x [0, 1], then set
            # D_ij = sqrt((x_i - x_j)^2 + (y_i - y_j)^2).
            coords = rng.random((N, 2))
            dx = coords[:, 0][:, None] - coords[:, 0][None, :]
            dy = coords[:, 1][:, None] - coords[:, 1][None, :]
            D = np.sqrt(dx * dx + dy * dy)

            # Ensure exact zeros on the diagonal (no self-distance).
            np.fill_diagonal(D, 0.0)

            filename = OUTPUT_DIR / f"{N}-{k}.npz"
            np.savez(filename, F=F, D=D)
            # Optional: print progress; comment out if noisy.
            # print(f"Saved {filename}")


if __name__ == "__main__":
    main()
