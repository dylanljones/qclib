# coding: utf-8
#
# This code is part of qclib.
# 
# Copyright (c) 2021, Dylan Jones

import random
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import XGate, CXGate, CCXGate
from . import libary, run


def random_permutation(qc: QuantumCircuit, depth: int):
    gates = [XGate, CXGate] if qc.num_qubits == 2 else [XGate, CCXGate]
    num_ctrl = int(qc.num_qubits / 2)
    for _ in range(depth):
        gate = random.choice(gates)()
        if hasattr(gate, "num_ctrl_qubits"):
            ctrl_qubits = qc.qubits[:num_ctrl]
            target_qubits = qc.qubits[num_ctrl:]
            qargs = list(random.sample(ctrl_qubits, k=gate.num_ctrl_qubits))
            qargs.append(random.choice(target_qubits))
        else:
            qargs = [random.choice(qc.qubits)]
        qc.append(gate, qargs)


class PRNGCircuit(QuantumCircuit):

    def __init__(self, *regs, depth: int = 100, name: str = "P"):
        super().__init__(*regs, name=name)
        self._build(depth)

    def _build(self, depth):
        gates = [XGate, CXGate] if self.num_qubits == 2 else [XGate, CCXGate]
        num_ctrl = int(self.num_qubits / 2)
        for _ in range(depth):
            gate = random.choice(gates)()
            if hasattr(gate, "num_ctrl_qubits"):
                ctrl_qubits = self.qubits[:num_ctrl]
                target_qubits = self.qubits[num_ctrl:]
                qargs = list(random.sample(ctrl_qubits, k=gate.num_ctrl_qubits))
                qargs.append(random.choice(target_qubits))
            else:
                qargs = [random.choice(self.qubits)]

            self.append(gate, qargs)


class TBTimeEvolutionCircuit(libary.AbstractTimeEvolutionCircuit):

    def __init__(self, *regs, dt: float, w=0.0, num_perm: int = 10, mode: str = "noancilla",
                 reverse: bool = False, ):
        super().__init__(*regs, dt=dt, name="Trotter")
        self.w = w
        self.reverse = reverse
        self.num_perm = num_perm
        self.mode = mode

    def define_trotter_step(self, qc: QuantumCircuit, dt: float) -> None:
        # Kinetic hamiltonian
        # ===================
        index = qc.num_qubits - 1 if self.reverse else 0

        # Even part of the kinetic hamiltonian
        qc.rx(dt, index)
        # Odd part of the kinetic hamiltonian
        libary.increment(qc, self.reverse, self.mode)  # Translate lattice by +1
        qc.rx(dt, index)
        libary.decrement(qc, self.reverse, self.mode)  # Translate lattice by -1

        # Potential hamiltonian
        # =====================
        if self.w:
            p = PRNGCircuit(qc.num_qubits, depth=self.num_perm)
            # perform random permutation
            qc.append(p.to_instruction(), qc.qubits[:])
            # diagonal phase rotation around z-axis
            for i in range(qc.num_qubits):
                n = i + 1
                phi = self.w * dt / (2**n)
                qc.rz(phi, qc.qubits[i])
            # perform inverse random permutation for next step
            qc.append(p.inverse().to_instruction(), qc.qubits[:])


def run_time_evolution(num_sites, tmax, dt=0.2, w=0.0, x0=None, num_perm=10, shots=1024, header=""):
    num_qubits = np.log2(num_sites)
    qreg = QuantumRegister(num_qubits)
    creg = ClassicalRegister(num_qubits)

    steps = int(tmax / dt) + 1
    probs = np.zeros((steps, num_sites), dtype="float")
    x0 = x0 if x0 is not None else num_sites // 2

    for i in range(steps):
        prog = (i + 1) / steps
        string = f"{i + 1}/{steps}  {100 * prog:4.1f}% -- t: {i * dt:.2f}"
        print("\r" + header + string, end="", flush=True)

        # define circuit for Trotter step `i`
        qc = TBTimeEvolutionCircuit(qreg, creg, dt=dt, w=w, num_perm=num_perm)
        libary.binary(qc, x0)
        qc.trotter_evolution(i, qreg[:])
        # measure results
        qc.measure(qreg[:], creg[:])  # noqa

        # run circuit and store results
        res = run(qc, shots=shots)
        probs[i] = res.probabilities
    print()
    return probs


def get_positions(probs, full=False):
    """Computes the positions of the particle during the time evolution.

    The position of the particle is defined by the maximum of the probability distribution.

    Parameters
    ----------
    probs : (..., N, M) np.ndarray
        The array containing the probability distributions for each time step.
    full : bool, optional
        Wheter to use the full probability distribution in both directions.
        If ``False`` only one direction from the starting point is used.
        The default is ``False``.

    Returns
    -------
    positions : (..., N) np.ndarray
        The positions of the particle for each time step.
    """
    num_sites = probs.shape[-1]
    x0 = int(num_sites // 2)
    if full:
        return np.argmax(probs, axis=-1) - x0
    return np.argmax(probs[..., x0:], axis=-1)


def get_max_distance(positions):
    """Computes the maximal distance reached by the particle"""
    distances = np.abs(positions - positions[0])
    return np.max(distances)


def get_final_distance(probs, mode="mean"):
    positions = get_positions(probs, full=False)
    if mode == "mean":
        dist = np.mean(positions, axis=0)[-1]
    elif mode == "maxmean":
        dist = np.max(np.mean(positions, axis=0))
    elif mode == "max":
        dist = np.max(positions)
    else:
        raise ValueError("Mode " + mode + " not supported!")
    return dist
