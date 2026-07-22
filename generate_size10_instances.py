"""Generate only the size-10 QAP instances used by the ground-state workflow.

This is a focused variant of ``generate_instances.py`` that creates
``instances/10-1.npz`` through ``instances/10-100.npz`` using the same
deterministic generation rule as the main dataset generator.

Run from the project root:

    python generate_size10_instances.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


SIZE = 10
INSTANCES_PER_SIZE = 100
OUTPUT_DIR = Path("instances")


def main() -> None:
	OUTPUT_DIR.mkdir(exist_ok=True)

	rng = np.random.default_rng(42 + SIZE)

	for instance_index in range(1, INSTANCES_PER_SIZE + 1):
		# Flow matrix: symmetric, non-negative, zero diagonal.
		flow_upper = rng.integers(0, 10, size=(SIZE, SIZE)).astype(float)
		flow = np.triu(flow_upper, k=1)
		flow = flow + flow.T
		np.fill_diagonal(flow, 0.0)

		# Distance matrix: Euclidean distances between random 2D coordinates.
		coords = rng.random((SIZE, 2))
		dx = coords[:, 0][:, None] - coords[:, 0][None, :]
		dy = coords[:, 1][:, None] - coords[:, 1][None, :]
		distance = np.sqrt(dx * dx + dy * dy)
		np.fill_diagonal(distance, 0.0)

		filename = OUTPUT_DIR / f"{SIZE}-{instance_index}.npz"
		np.savez(filename, F=flow, D=distance)


if __name__ == "__main__":
	main()