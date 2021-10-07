# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import qiskit
from qclib import PauliOperators, abs2, fill_diagonal, get_ground_state
from qclib.circuit import QuantumCircuit
from qclib.vqe import VariationalSolver, VQEFitter


def compute_ground_state_com(num_sites, eps=0., hop=1., periodic=True):
    diag = np.full(num_sites, fill_value=eps)
    off_diag = np.full(num_sites - 1, fill_value=hop)
    if periodic:
        ham = np.zeros((num_sites, num_sites))
        fill_diagonal(ham, diag)
        fill_diagonal(ham, off_diag, -1)
        fill_diagonal(ham, off_diag, +1)
        ham[0, -1] = ham[-1, 0] = hop
        eigvals, eigvecs = la.eigh(ham)
    else:
        eigvals, eigvecs = la.eigh_tridiagonal(diag, off_diag)
    return get_ground_state(eigvals, eigvecs)


def hamiltonian_dir(num_sites, eps=0., hop=1.0):
    size = int(2 ** num_sites)
    s = PauliOperators(num_sites)
    ham = np.zeros((size, size), dtype=np.complex64)
    if hop:
        for i in range(num_sites - 1):
            sign = (-1) ** i
            ham += sign * hop * (s.x[i] @ s.x[i + 1] + s.y[i] @ s.y[i + 1])
    if eps:
        for i in range(num_sites):
            ham += - eps * s.z[i]
    return ham / 2


def compute_ground_state_dir(num_sites, eps=0., hop=1., periodic=True):
    ham = hamiltonian_dir(num_sites, eps, hop)
    eigvals, eigvecs = la.eigh(ham)
    return get_ground_state(eigvals, eigvecs)


def main():
    num_sites = 4
    eps, hop = 0.0, 1.0

    gs_energy_dir, gs_dir = compute_ground_state_dir(num_sites, eps, hop, periodic=False)
    gs_energy_com, gs_com = compute_ground_state_com(num_sites, eps, hop, periodic=False)
    print(gs_energy_dir, gs_energy_com)


if __name__ == "__main__":
    main()
