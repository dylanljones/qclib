# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
from .math import kron, empty_single_qubit_layer, si, sx, sy, sz


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


def tight_binding_chain(num_sites, eps=0., hop=1.):
    return onsite_terms(num_sites, eps) + hopping_terms_chain(num_sites, hop)


class OpView:

    def __init__(self, ops, indices, op):
        self._ops = ops
        self.indices = [indices] if isinstance(indices, int) else indices
        if isinstance(op, str):
            op = [op for _ in range(len(self.indices))]
        for i, idx in enumerate(self.indices):
            self._ops[idx] = op[i]

    @property
    def str(self):
        return "".join(self._ops)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for i in self.indices:
            self._ops[i] = "I"


def spin2list(spin, spinless=False):
    if spin is None:
        spin = [0] if spinless else [0, 1]
    elif isinstance(spin, int):
        spin = [spin]
    return spin


def build_op_matrix(ham):
    num_qubits = len(ham[0][1])
    size = 2 ** num_qubits

    primitives = {"I": si, "X": sx, "Y": sy, "Z": sz}

    arr = np.zeros((size, size), dtype=np.complex128)
    for coeff, opstr in ham:
        items = [primitives[char] for char in opstr]
        arr += coeff * kron(*items)
    return arr


class HamiltonianPauliOperator:

    def __init__(self, num_sites, order="spin", spinless=False):
        self.spinless = spinless
        self.order = order
        self.num_sites = num_sites
        self.size = num_sites if spinless else 2 * num_sites

        self._ops = ["I"] * self.size
        self._ham_ops = list()

    def _op_view(self, indices, op):
        return OpView(self._ops, indices, op)

    def index(self, site: int, spin: int = 0):
        if self.spinless:
            return site
        if self.order == "spin":
            return site + spin * self.num_sites
        elif self.order == "site":
            return 2 * site + spin

    def onsite(self, eps: float, site: int, spin: int = None):
        if not eps:
            return
        spin = spin2list(spin, self.spinless)
        for s in spin:
            i = self.index(site, s)
            with self._op_view(i, "Z") as op:
                self._ham_ops.append((-eps / 2, op.str))

    def hopping(self, hop, site1, site2, spin: int = None):
        if not hop:
            return
        spin = spin2list(spin, self.spinless)
        for s in spin:
            i, j = self.index(site1, s), self.index(site2, s)
            indices = list(range(i, j+1))
            # X-X term
            op = ["X"] + ["Z" for _ in range(j-i-1)] + ["X"]
            with self._op_view(indices, op) as op:
                self._ham_ops.append((hop / 2, op.str))
            # Y-Y term
            op = ["Y"] + ["Z" for _ in range(j-i-1)] + ["Y"]
            with self._op_view(indices, op) as op:
                self._ham_ops.append((hop / 2, op.str))

    def inter(self, u: float, site: int):
        if not u or self.spinless:
            return
        i = self.index(site, spin=0)
        j = self.index(site, spin=1)
        with self._op_view([i, j], "Z") as op:
            self._ham_ops.append((u / 4, op.str))
        with self._op_view(i, "Z") as op:
            self._ham_ops.append((-u / 4, op.str))
        with self._op_view(j, "Z") as op:
            self._ham_ops.append((-u / 4, op.str))

    def to_list(self):
        return self._ham_ops.copy()

    def to_matrix(self):
        return build_op_matrix(self._ham_ops)
