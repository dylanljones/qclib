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
import mthree
import mthree.utils
from .visualization import plot_histogram


def param_generator(name="θ"):
    count = 0
    while True:
        param_name = f"${name}_{count}$"
        p = qiskit.circuit.Parameter(param_name)
        count += 1
        yield p


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


def check_measurement(qc):
    """Checks if one or more measurements are performed on the circuit."""
    instructions = list(qc.to_instruction().definition)
    for ins in instructions:
        obj = ins[0]
        if isinstance(obj, qiskit.circuit.Measure):
            return True
    return False


def ensure_measurement(qc, inplace=True):
    """Adds measurements to all qubits if no other measurmeents are performed."""
    if not check_measurement(qc):
        qc.measure_all(inplace=inplace)


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


def transpile(circuits, backend):
    if isinstance(circuits, qiskit.QuantumCircuit):
        ensure_measurement(circuits)
    else:
        for qc in circuits:
            ensure_measurement(qc)
    return qiskit.transpile(circuits, backend)


def run_transpiled(transpiled, backend: qiskit.providers.Backend, shots: int = 1024,
                   params: Union[dict, Sequence[float]] = None):
    """Run a transpiled `QuantumCircuit`."""
    if params is not None:
        if isinstance(transpiled, Sequence):
            transpiled = [t.bind_parameters(p) for t, p in zip(transpiled, params)]
        else:
            transpiled = transpiled.bind_parameters(params)

    # noinspection PyUnresolvedReferences
    job = backend.run(transpiled, shots=shots)
    counts = job.result().get_counts()
    if isinstance(transpiled, qiskit.QuantumCircuit):
        results = Result(counts)
    else:
        results = [Result(x) for x in counts]
    return results


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


# =========================================================================
# Expectation values
# =========================================================================


def _opstr_to_meas_circ(op_strings):
    """Constructs circuits with the correct post-rotations for measurements.

    Parameters
    ----------
    op_strings : list of str
        List of strings representing the operators needed for measurements.

    Returns
    -------
    circs : list of qiskit.QuantumCircuit
        List of circuits for measurement post-rotations.
    """
    num_qubits = len(op_strings[0])
    circs = []
    for op in op_strings:
        qc = qiskit.QuantumCircuit(num_qubits)
        for idx, item in enumerate(op):
            if item == 'X':
                qc.h(num_qubits-idx-1)
            elif item == 'Y':
                qc.sdg(num_qubits-idx-1)
                qc.h(num_qubits-idx-1)
        circs.append(qc)
    return circs


def _prepare_circuits(backend, ham, ansatz_circ):
    # Split the Hamiltonian into two arrays, one for coefficients, the other for operator strings
    coeffs = np.array([item[0] for item in ham], dtype=complex)
    op_strings = [item[1] for item in ham]

    # Construct the measurement circuits with the correct single-qubit rotation gates.
    meas_circs = _opstr_to_meas_circ(op_strings)

    # When computing the expectation value for the energy, we need to know if we
    # evaluate a Z measurement or and identity measurement.
    # Here we take and X and Y operator in the strings and convert it to a Z since
    # we added the rotations with the meas_circs.
    meas_strings = [string.replace('X', 'Z').replace('Y', 'Z') for string in op_strings]

    # Take the ansatz circuits, add the single-qubit measurement basis rotations from
    # meas_circs, and finally append the measurements themselves.
    full_circs = [ansatz_circ.compose(mcirc).measure_all(inplace=False) for mcirc in meas_circs]

    # Because we are in general targeting a real quantum system, our circuits must be transpiled
    # to match the system topology and, hopefully, optimize them.
    # Here we will set the transpiler to the most optimal settings where 'sabre' layout and
    # routing are used, along with full O3 optimization.

    # This works around a bug in Qiskit where Sabre routing fails for simulators (Issue #7098)
    trans_dict = {}
    if not backend.configuration().simulator:
        trans_dict = {'layout_method': 'sabre', 'routing_method': 'sabre'}
    trans_circs = qiskit.transpile(full_circs, backend, optimization_level=3, **trans_dict)

    return coeffs, meas_strings, trans_circs


def _measure_expvals(backend, trans_circs, exp_ops, params, shots, use_meas_mitigation=False):
    # Attach (bind) parameters in params vector to the transpiled circuits.
    bound_circs = [circ.bind_parameters(params) for circ in trans_circs]
    # Submit the job and get the resultant counts back
    counts = backend.run(bound_circs, shots=shots).result().get_counts()

    # If using measurement mitigation we need to find out which physical qubits our transpiled
    # circuits actually measure, construct a mitigation object targeting our backend, and
    # finally calibrate our mitgation by running calibration circuits on the backend.
    if use_meas_mitigation:
        maps = mthree.utils.final_measurement_mapping(trans_circs)
        mit = mthree.M3Mitigation(backend)
        mit.cals_from_system(maps)
        quasi_collection = mit.apply_correction(counts, maps)
        expvals = quasi_collection.expval(exp_ops)
    else:
        expvals = mthree.utils.expval(counts, exp_ops)
    return expvals


def measure_expval(params, ham, ansatz, backend, shots):
    coeffs, meas_strings, trans_circs = _prepare_circuits(backend, ham, ansatz)
    expvals = _measure_expvals(backend, trans_circs, meas_strings, params, shots)
    return np.sum(coeffs * expvals).real
