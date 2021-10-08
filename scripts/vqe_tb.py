# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from qclib import PauliOperators, abs2, fill_diagonal, get_ground_state, compute_ground_state
from qclib.circuit import QuantumCircuit
from qclib.vqe import VariationalSolver, VQEFitter
from qclib import jordan_wigner as jw


def main():
    num_sites = 5
    eps, hop = 1.0, 1.0

    ham = jw.onsite_terms(num_sites, eps) + jw.hopping_terms_chain(num_sites, hop)
    e_gs, gs = compute_ground_state(ham)
    probs = abs2(gs)

    vqe = VQEFitter(probs, layers=3, gates="rx ry rz", entangle="linear", shots=10 * 1024)
    sol = vqe.optimize(tol=1e-10, method="COBYLA", catol=0.00005, maxiter=10_000)
    probs_vqe = vqe.get_optimized_probabilities()
    print(sol)

    fig, ax = plt.subplots()
    x = np.arange(probs.shape[0])
    ax.bar(x, probs, alpha=0.5)
    ax.bar(x, probs_vqe, alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()
