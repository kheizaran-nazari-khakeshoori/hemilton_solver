"""
Ising Model Hamiltonian
=======================
Computes the energy of a spin configuration under the Ising model:

    H(s) = -sum_{i<j} J_ij * s_i * s_j  -  sum_i h_i * s_i

where:
    s   - spin configuration, each s_i in {-1, +1}
    J   - coupling matrix (J[i][j] = interaction strength between spins i and j)
    h   - external magnetic field vector (h[i] = local field at site i)
"""

import numpy as np


def hamiltonian(s: np.ndarray, J: np.ndarray, h: np.ndarray) -> float:
    """
    Calculate the Ising model Hamiltonian for a given spin configuration.

    Parameters
    ----------
    s : np.ndarray, shape (N,)
        Spin configuration; each element must be +1 or -1.
    J : np.ndarray, shape (N, N)
        Coupling matrix. Only the upper triangle (i < j) is used.
    h : np.ndarray, shape (N,)
        External magnetic field at each spin site.

    Returns
    -------
    float
        The total energy H(s).
    """
    s = np.asarray(s, dtype=float)
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)

    N = len(s)
    if J.shape != (N, N):
        raise ValueError(f"J must be ({N}, {N}), got {J.shape}")
    if h.shape != (N,):
        raise ValueError(f"h must be ({N},), got {h.shape}")

    interaction_energy = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            interaction_energy += J[i, j] * s[i] * s[j]

    field_energy = np.dot(h, s)

    return -interaction_energy - field_energy
