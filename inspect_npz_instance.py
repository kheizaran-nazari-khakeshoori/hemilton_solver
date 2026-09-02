"""Inspect a NumPy .npz instance file from the command line.

Usage:

    python3 inspect_npz_instance.py instances/10-1.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Inspect a .npz instance file")
	parser.add_argument("path", type=Path, help="Path to the .npz file")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	path = args.path

	if not path.is_file():
		raise SystemExit(f"File not found: {path}")

	data = np.load(path)
	print(f"File: {path}")
	print(f"Keys: {list(data.files)}")

	for key in data.files:
		array = np.asarray(data[key])
		print(f"\n[{key}] shape={array.shape} dtype={array.dtype}")
		print(array)


if __name__ == "__main__":
	main()