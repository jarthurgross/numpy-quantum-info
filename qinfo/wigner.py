"""Tools for working with quasiprobability distributions.

"""

import itertools as it
import numpy as np
from sympy.physics.wigner import clebsch_gordan

def CG_coeff(j1, j2, j3, m1, m2, m3):
    return float(clebsch_gordan(j1, j2, j3, m1, m2, m3))

def Delta_nz(s, epsilons=None):
    if epsilons is None:
        epsilons = it.repeat(1)
    dim = int(2*s + 1)
    ms = [s - n for n in range(dim)]
    ls = list(range(dim))
    return np.diag([sum([epsilon*(2*l + 1)/(2*s + 1)*CG_coeff(s, l, s, m, 0, m)
                         for epsilon, l in zip(epsilons, ls)]) for m in ms])

def rotation_matrix(phi, theta, Jz, Rx_pi_2):
    Jz_diag = np.diag(Jz)
    Rz_phi = np.diagflat(np.exp(-1.j*phi*Jz_diag))
    Rz_theta = np.diagflat(np.exp(-1.j*theta*Jz_diag))
    return Rz_phi @ Rx_pi_2.conj().T @ Rz_theta @ Rx_pi_2

def make_Delta(phi, theta, Delta_0, Jz, Rx_pi_2):
    # Assume Jz is diagonal
    R = rotation_matrix(phi, theta, Jz, Rx_pi_2)
    return R @ Delta_0 @ R.conj().T

def calc_spin_wigner(rho, Deltas):
    dim = rho.shape[0]
    s = (dim - 1)/2
    Ws = np.tensordot(rho, Deltas, ([0, 1], [-1, -2])).real
    return Ws
