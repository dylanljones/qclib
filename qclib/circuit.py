# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
import qiskit
from .result import Result
from ._utils import Binary


def run(qc, shots=1024, backend=None):
    if backend is None:
        backend = qiskit.Aer.get_backend("qasm_simulator")
    job = qiskit.execute(qc, backend, shots=shots)
    return Result(job.result().get_counts())


class QuantumRegister(qiskit.QuantumRegister):

    def __init__(self, size, name=None):
        super().__init__(size, name)


class ClassicalRegister(qiskit.ClassicalRegister):

    def __init__(self, size, name=None):
        super().__init__(size, name)


class QuantumCircuit(qiskit.QuantumCircuit):

    def __init__(self, *regs, name=None):
        super().__init__(*regs, name=name)

    @property
    def registers(self):
        return *self.qregs, *self.cregs

    def init_binary(self, value, order=+1):
        b = Binary(value, width=self.num_qubits)
        binarr = b.binarr()[::order]
        regs = np.where(binarr == 1)[0]
        for i in regs:
            self.x(i)

    def measure(self, qregs=None, cregs=None):
        if qregs is None:
            qregs = np.arange(self.num_qubits)
        if cregs is None:
            cregs = np.arange(len(qregs))
        return super().measure(qregs, cregs)  # noqa

    def run(self, shots=1024, backend=None):
        return run(self, shots, backend)

    def append_circuit(self, circuit, qargs=None, cargs=None):
        self.append(circuit.to_instruction(), qargs, cargs)
