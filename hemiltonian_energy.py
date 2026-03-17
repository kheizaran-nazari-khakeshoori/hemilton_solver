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
