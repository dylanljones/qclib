# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
from qclib import pauli, kron, empty_single_qubit_layer

si, sx, sy, sz = pauli
sp = 0.5 * (sx + 1j * sy)
sm = 0.5 * (sx - 1j * sy)


def creation_operator_jw(num_qubits, idx):
    arrays = empty_single_qubit_layer(num_qubits)
    arrays[:idx - 1] = sz   # pauli string
    arrays[idx] = sm        # S⁻
    return kron(arrays)


def annihilation_operator_jw(num_qubits, idx):
    arrays = empty_single_qubit_layer(num_qubits)
    arrays[:idx - 1] = sz   # pauli string
    arrays[idx] = sp        # S⁺
    return kron(arrays)


def hopping_term_full(num_qubits, i, j):
    copi_dag = creation_operator_jw(num_qubits, i)
    copj = annihilation_operator_jw(num_qubits, j)

    copj_dag = creation_operator_jw(num_qubits, j)
    copi = annihilation_operator_jw(num_qubits, i)

    return (copi_dag @ copj) + (copj_dag @ copi)


def hopping_term(num_qubits, i, j):
    xx_terms = empty_single_qubit_layer(num_qubits)
    yy_terms = empty_single_qubit_layer(num_qubits)
    xx_terms[[i, j]] = sx
    yy_terms[[i, j]] = sy

    indices = np.arange(i+1, j)
    xx_terms[indices] = sz
    yy_terms[indices] = sz

    return 0.5 * (kron(xx_terms) + kron(yy_terms))


def main():
    num_qubits = 3
    i, j = 1, 2
    hop1 = hopping_term_full(num_qubits, i, j)
    hop2 = hopping_term(num_qubits, i, j)
    print(hop1.real)
    print(hop2.real)


if __name__ == "__main__":
    main()
