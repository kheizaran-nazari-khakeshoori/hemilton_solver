"""QAP objective utilities.

This module provides the canonical Quadratic Assignment Problem (QAP) cost:

    cost(P) = sum_{i,j} F[i,j] * D[P[i], P[j]]

where P is a permutation that assigns facility i to location P[i].
"""

import numpy as np


def permutation_to_assignment(P: np.ndarray) -> np.ndarray:
    """Convert permutation vector P to binary assignment matrix X."""
    P = np.asarray(P, dtype=int)
    N = len(P)
    X = np.zeros((N, N), dtype=int)
    X[np.arange(N), P] = 1
    return X


def qap_cost(F: np.ndarray, D: np.ndarray, P: np.ndarray) -> float:
    """Compute canonical QAP cost for a permutation assignment."""
    F = np.asarray(F, dtype=float)
    D = np.asarray(D, dtype=float)
    P = np.asarray(P, dtype=int)

    N = len(P)
    if F.shape != (N, N):
        raise ValueError(f"F must be ({N}, {N}), got {F.shape}")
    if D.shape != (N, N):
        raise ValueError(f"D must be ({N}, {N}), got {D.shape}")
    if set(P.tolist()) != set(range(N)):
        raise ValueError("P must be a valid permutation of [0..N-1]")

    return float(np.sum(F * D[np.ix_(P, P)]))


def qap_cost_matrix(F: np.ndarray, D: np.ndarray, P: np.ndarray) -> float:
    """Matrix-form QAP cost using assignment matrix X."""
    X = permutation_to_assignment(P)
    return float(np.sum((F @ X @ D) * X))


def hamiltonian_vectorized(P: np.ndarray, F: np.ndarray, D: np.ndarray) -> float:
    """Backward-compatible alias for callers still using old function name."""
    return qap_cost(F, D, P)


if __name__ == "__main__":
    N = 5
    rng = np.random.default_rng(42)
    F = rng.integers(0, 10, size=(N, N)).astype(float)
    D = rng.integers(0, 10, size=(N, N)).astype(float)
    np.fill_diagonal(F, 0)
    np.fill_diagonal(D, 0)
    P = rng.permutation(N)

    c1 = qap_cost(F, D, P)
    c2 = qap_cost_matrix(F, D, P)

    print("QAP Objective Demo")
    print("=" * 40)
    print(f"Permutation P  : {P.tolist()}")
    print(f"QAP cost       : {c1:.4f}")
    print(f"Matrix formula : {c2:.4f}")
