# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import qiskit
import numpy as np
from collections.abc import Mapping
from typing import Iterator, Dict, Union, Sequence
import matplotlib.pyplot as plt
from qiskit.circuit import Qubit, Clbit, AncillaQubit
from qiskit.circuit import QuantumRegister, ClassicalRegister, AncillaRegister
from .visualization import plot_histogram


class Result(Mapping):

    MAPPING = np.array([+1, -1])

    def __init__(self, data: Dict[str, int]):
        super().__init__()
        self._data = data
        self.num_samples = sum(x for x in data.values())
        self.num_qubits = len(list(data.keys())[0])
        self.size = 2 ** self.num_qubits
        self.labels = [f"{x:0{self.num_qubits}b}" for x in range(self.size)]

    @property
    def raw(self):
        return self._data

    @property
    def probabilities(self):
        return {s: c / self.num_samples for s, c in self._data.items()}

    @property
    def integers(self):
        total = self.num_samples
        return {int(s, 2): count / total for s, count in self._data.items()}

    @property
    def integer(self):
        integer, count = 0, 0.
        for s, c in self._data.items():
            if c > count:
                integer = int(s, 2)
                count = c
        return integer

    @property
    def count_vector(self):
        arr = np.zeros(self.size, dtype=np.int64)
        for k, v in self._data.items():
            arr[int(k, 2)] = v
        return arr

    @property
    def probability_vector(self):
        return self.count_vector / self.num_samples

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, k: str) -> int:
        return self._data[k]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._data})"

    def max(self):
        return max(list(self._data.values()))

    def argmax(self, index: bool = False):
        arg, max_count = 0, 0
        for key, counts in self._data.items():
            if counts > max_count:
                arg = key
        if index:
            arg = int(arg, 2)
        return arg

    def accumulate_counts(self, qubit: int):
        counts = {"0": 0, "1": 0}
        for state, num in self._data.items():
            counts[state[qubit]] += num
        return np.array([counts["0"], counts["1"]])

    def accumulate_probabilities(self, qubit: int = 0):
        return self.accumulate_counts(qubit) / self.num_samples

    def expectation(self, qubit: int = 0):
        probs = self.accumulate_probabilities(qubit)
        return np.sum(self.MAPPING * probs)

    def expectation_values(self):
        return np.array([self.expectation(i) for i in range(self.num_qubits)])

    def plot_histogram(self, **kwargs):
        return plot_histogram(self._data, **kwargs)


# noinspection PyBroadException
def transpile(qc, backend: Union[str, qiskit.providers.Backend] = None, gpu: bool = False):
    if backend is None:
        backend = qiskit.Aer.get_backend("qasm_simulator")
    elif isinstance(backend, str):
        if backend == "qasm":
            backend = "qasm_simulator"
        else:
            backend = "aer_simulator_" + backend
        backend = qiskit.Aer.get_backend(backend)

    # Use gpu for simulators
    if gpu:
        try:
            backend.set_options(device='GPU')
        except Exception:
            pass

    # transpile the circuit
    transpiled = qiskit.transpile(qc, backend=backend)
    return transpiled


# noinspection PyBroadException
def init_backend(backend: Union[str, qiskit.providers.Backend] = None,
                 gpu: bool = False) -> qiskit.providers.Backend:
    if backend is None:
        backend = qiskit.Aer.get_backend("qasm_simulator")
    elif isinstance(backend, str):
        if backend == "qasm":
            backend = "qasm_simulator"
        else:
            backend = "aer_simulator_" + backend
        backend = qiskit.Aer.get_backend(backend)

    # Use gpu for simulators
    if gpu:
        try:
            backend.set_options(device='GPU')
        except Exception:
            pass
    return backend


def run_transpiled(transpiled, backend: qiskit.providers.Backend, shots: int = 1024,
                   params: Union[dict, Sequence[float]] = None):
    """Run a transpiled `QuantumCircuit`."""
    if params is not None:
        transpiled = transpiled.bind_parameters(params)

    # noinspection PyUnresolvedReferences
    result = backend.run(transpiled, shots=shots).result()
    return Result(result.get_counts(transpiled))


# noinspection PyBroadException
def run(qc, shots: int = 1024, backend: Union[str, qiskit.providers.Backend] = None,
        params: Union[dict, Sequence[float]] = None, gpu: bool = False):
    if backend is None:
        backend = qiskit.Aer.get_backend("qasm_simulator")
    elif isinstance(backend, str):
        if backend == "qasm":
            backend = "qasm_simulator"
        else:
            backend = "aer_simulator_" + backend
        backend = qiskit.Aer.get_backend(backend)

    # Use gpu for simulators
    if gpu:
        try:
            backend.set_options(device='GPU')
        except Exception:
            pass

    # transpile the circuit
    transpiled = qiskit.transpile(qc, backend=backend)
    if params is not None:
        transpiled = transpiled.bind_parameters(params)

    # result = qiskit.execute(qc, backend, shots=shots).result()

    # assemble and run the circuit on the backend
    # qobj = qiskit.assemble(transpiled, shots=shots, backend=backend)
    result = backend.run(transpiled, shots=shots).result()
    return Result(result.get_counts(transpiled))


# noinspection PyUnresolvedReferences
def measure(qc: qiskit.QuantumCircuit, qbits, cbits, basis="z"):
    if not hasattr(qbits, "__len__"):
        qbits = [qbits]
    if not hasattr(cbits, "__len__"):
        cbits = [cbits]

    if basis == "x":
        # Measure in x (hadamard) basis
        for bit in qbits:
            qc.h(bit)
    elif basis == "y":
        # Measure in y (circular) basis
        for bit in qbits:
            qc.sdg(bit)
            qc.h(bit)
    elif basis == "z":
        # Measure in z (computational) basis
        pass
    else:
        raise ValueError(f"Can't measure in {basis} basis! Valid arguments: x, y, z")

    qc.measure(qbits, cbits)


# noinspection PyUnresolvedReferences
def measure_all(qc: qiskit.QuantumCircuit, basis="z"):
    if basis == "x":
        # Measure in x (hadamard) basis
        for bit in qc.qubits:
            qc.h(bit)
    elif basis == "y":
        # Measure in y (circular) basis
        for bit in qc.qubits:
            qc.sdg(bit)
            qc.h(bit)
    elif basis == "z":
        # Measure in z (computational) basis
        pass
    else:
        raise ValueError(f"Can't measure in {basis} basis! Valid arguments: x, y, z")

    qc.measure_all()


class QuantumCircuit(qiskit.QuantumCircuit):

    def measure_basis(self, qbits, cbits, basis="z"):
        return measure(self, qbits, cbits, basis)

    def measureall_basis(self, basis="z"):
        return measure_all(self, basis)

    def run(self, shots: int = 1024, backend: Union[str, qiskit.providers.Backend] = None,
            params: dict = None, gpu: bool = False):
        return run(self, shots, backend, params, gpu)

    def plot(self, show=True, **kwargs):
        self.draw("mpl", **kwargs)
        if show:
            plt.show()
