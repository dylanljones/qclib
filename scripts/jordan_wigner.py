# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
from typing import Union, Sequence
from qclib import pauli, kron, empty_single_qubit_layer, PauliOperators, compute_ground_state
from qclib import jordan_wigner as jw


si, sx, sy, sz = pauli
sp = 0.5 * (sx + 1j * sy)
sm = 0.5 * (sx - 1j * sy)


def creation_operator_jw(num_qubits, idx):
    arrays = empty_single_qubit_layer(num_qubits)
    arrays[:idx] = sz   # pauli string
    arrays[idx] = sm        # S⁻
    return kron(arrays)


def annihilation_operator_jw(num_qubits, idx):
    arrays = empty_single_qubit_layer(num_qubits)
    arrays[:idx] = sz   # pauli string
    arrays[idx] = sp        # S⁺
    return kron(arrays)


def hopping_term_full(num_qubits, i, j):
    copi_dag = creation_operator_jw(num_qubits, i)
    copj = annihilation_operator_jw(num_qubits, j)

    copj_dag = creation_operator_jw(num_qubits, j)
    copi = annihilation_operator_jw(num_qubits, i)

    return np.matmul(copi_dag, copj) + np.matmul(copj_dag, copi)


def main():
    num_qubits = 4
    eps, hop = 1.0, 1.0
    ham = jw.onsite_terms(num_qubits, eps) + jw.hopping_terms_chain(num_qubits, hop)
    print(ham)


if __name__ == "__main__":
    main()
