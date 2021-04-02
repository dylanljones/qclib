# coding: utf-8
#
# Created on 22 Oct 2020
# Author: Dylan Jones

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister, ClassicalRegister
from itertools import count
from .circuit import run


class VqeSolver:

    MODE_FULL = "full"
    MODE_LINEAR = "linear"

    def __init__(self, num_qubits: int, mode="linear", depth: int = 2):
        self.num_qubits = num_qubits
        self.num_params = 2 * num_qubits * (depth + 1)
        self.mode = mode
        self.depth = depth

        self.circuit = None
        self.params = list()
        self._index = count()
        self._target = None
        self._sol = None

        self._build_circuit()

    @property
    def sol(self):
        return

    @property
    def x(self):
        return self._sol.x

    def __repr__(self):
        return f"{self.__class__.__name__}(qubits: {self.num_qubits}, entanglement: {self.mode})"

    def _next_param(self):
        index = next(self._index)
        param = Parameter(str(index))
        self.params.append(param)
        return param

    def _control_layer(self, qc: QuantumCircuit):
        for i in range(self.num_qubits):
            qc.ry(self._next_param(), i)
            qc.rz(self._next_param(), i)

    def _build_circuit(self):
        qbits = QuantumRegister(self.num_qubits)
        qc = QuantumCircuit(qbits, name="VQE")

        self._control_layer(qc)
        for _ in range(self.depth):
            # Entanglement layer
            if self.mode == self.MODE_LINEAR:
                for i in range(self.num_qubits - 1):
                    qc.cx(i, i + 1)
            elif self.mode == self.MODE_FULL:
                for i in range(self.num_qubits - 1):
                    for j in range(i + 1):
                        qc.cx(j, i + 1)
            self._control_layer(qc)

        self.circuit = qc

    def get_circuit(self, args) -> QuantumCircuit:
        value_dict = dict(zip(self.params, args))
        return self.circuit.bind_parameters(value_dict)

    def draw(self, show=True):
        self.circuit.draw("mpl")
        if show:
            plt.show()

    def cost_function(self, res):
        state = res.probabilities
        return np.sum(np.abs(state - self._target))

    def func(self, args):
        qc = self.get_circuit(args)
        creg = ClassicalRegister(self.num_qubits, "c")
        qc.add_register(creg)

        qc.measure(range(self.num_qubits), range(self.num_qubits))  # noqa
        res = run(qc, shots=1024)
        return self.cost_function(res)

    def optimize(self, target, tol=None, method="COBYLA", x0=None):
        self._target = target
        if x0 is None:
            x0 = np.random.uniform(0, np.pi, size=self.num_params)
        self._sol = optimize.minimize(self.func, x0=x0, method=method, tol=tol)
        return self._sol
