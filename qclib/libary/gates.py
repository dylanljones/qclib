# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import qiskit


def xy_gate(phi):
    qc = qiskit.QuantumCircuit(2, name="XY")
    qc.cx(1, 0)
    qc.crx(4 * phi, 0, 1)
    qc.cx(1, 0)
    return qc.to_instruction()
