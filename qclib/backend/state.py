# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
from ..math import apply_statevec, apply_density, density_matrix
from .measure import measure_qubit, measure_qubit_rho


class DensityMatrix:

    def __init__(self, mat):
        self._data = np.asarray(mat)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def __str__(self):
        return str(self._data)

    def copy(self):
        return self.__class__(self._data.copy())

    def apply(self, operator):
        self._data = apply_density(self._data, operator)

    def measure(self, qubit, eigvals=None, eigvecs=None):
        res, self._data = measure_qubit_rho(self._data, qubit, eigvals, eigvecs)
        return res


class StateVector:

    def __init__(self, vec):
        self._data = np.asarray(vec)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def __str__(self):
        return str(self._data)

    def copy(self):
        return self.__class__(self._data.copy())

    def apply(self, operator):
        self._data = apply_statevec(self._data, operator)

    def measure(self, qubit, eigvals=None, eigvecs=None):
        res, self._data = measure_qubit(self._data, qubit, eigvals, eigvecs)
        return res

    def to_density(self):
        return DensityMatrix(density_matrix(self._data))
