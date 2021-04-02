# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

from qiskit import QuantumCircuit


def binary(qc: QuantumCircuit, num: int, order: int = -1) -> QuantumCircuit:
    w = qc.num_qubits
    indices = list(sorted(i for i, char in enumerate(f"{num:0{w}b}"[::order]) if char == "1"))
    for i in indices:
        qc.x(i)
    return qc


def increment(qc: QuantumCircuit, reverse: bool = False, mode: str = "noancilla"):
    """ Increments the binary value of the qubits in the given `QuantumCircuit` by 1. """
    if reverse:
        for i in range(qc.num_qubits - 1):
            qc.mcx(list(range(i+1, qc.num_qubits)), i, mode=mode)
        qc.x(qc.num_qubits - 1)
    else:
        for i in reversed(range(1, qc.num_qubits)):
            qc.mcx(list(range(0, i)), i, mode=mode)
        qc.x(0)


def decrement(qc: QuantumCircuit, reverse: bool = False, mode: str = "noancilla"):
    """ Decrements the binary value of the qubits in the given `QuantumCircuit` by 1. """
    if reverse:
        qc.x(qc.num_qubits - 1)
        for i in reversed(range(qc.num_qubits - 1)):
            qc.mcx(list(range(i+1, qc.num_qubits)), i, mode=mode)
    else:
        qc.x(0)
        for i in range(1, qc.num_qubits):
            qc.mcx(list(range(0, i)), i, mode=mode)
