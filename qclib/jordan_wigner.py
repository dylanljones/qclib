# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
from .math import kron, empty_single_qubit_layer, sx, sy, sz


def hopping_term(num_qubits, hop, i, j):
    xx_terms = empty_single_qubit_layer(num_qubits)
    yy_terms = empty_single_qubit_layer(num_qubits)
    xx_terms[[i, j]] = sx
    yy_terms[[i, j]] = sy
    if abs(i-j) > 1:
        # Jordan wigner string
        indices = np.arange(i+1, j)
        xx_terms[indices] = sz
        yy_terms[indices] = sz

    return hop / 2 * (kron(xx_terms) + kron(yy_terms)).real


def hopping_terms_chain(num_qubits, hop):
    return sum(hopping_term(num_qubits, hop, i, i+1) for i in range(num_qubits - 1))


def onsite_terms(num_qubits, eps):
    terms = empty_single_qubit_layer(num_qubits)
    terms[:] = - eps * sz / 2
    return kron(terms).real
