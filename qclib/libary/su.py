# coding: utf-8
#
# This code is part of qclib.
# 
# Copyright (c) 2021, Dylan Jones

import warnings
import qiskit


def rotation_layer(qc, qubits, params, axis="y"):
    for qbit in qubits:
        getattr(qc, "r" + axis)(next(params), qbit)


def entanglement_layer(qc, mode="linear", qubits=None):
    if qubits is None:
        qubits = qc.qubits
    num_qubits = len(qubits)
    if mode == "linear":
        for i in range(num_qubits - 1):
            qc.cx(qubits[i], qubits[i + 1])
    else:
        for con in range(num_qubits - 1):
            for target in range(con + 1, num_qubits):
                qc.cx(qubits[con], qubits[target])


def gate_layer(qc, gate, param_iter=None, qubits=None):
    if qubits is None:
        qubits = qc.qubits
    for qbit in qubits:
        if gate.startswith("r"):
            if param_iter is not None:
                param = next(param_iter)
            else:
                param = qiskit.circuit.Parameter(f"θ_{len(qc.parameters)}")
            getattr(qc, gate)(param, qbit)
        else:
            getattr(qc, gate)(qbit)


def su_layer(qc, gates, param_iter=None, qubits=None):
    for gate in gates:
        gate_layer(qc, gate, param_iter, qubits)


def efficient_su(qc, layers, gates, entangle="linear", final=True, params=None, qubits=None):
    param_iter = None if params is None else iter(params)
    for _ in range(layers):
        su_layer(qc, gates, param_iter, qubits)
        entanglement_layer(qc, entangle, qubits)
    if final:
        su_layer(qc, gates, param_iter, qubits)


def su_circuit(qc, params, layers, axes, entangle="linear", final_rot=True, barriers=False):
    warnings.warn("`su_circuit' is deprecated, use `efficient_su` instead!", DeprecationWarning)
    for _ in range(layers):
        for ax in axes:
            rotation_layer(qc, qc.qubits, params, axis=ax)
        entanglement_layer(qc, mode=entangle)
        if barriers:
            qc.barrier()

    if final_rot:
        for ax in axes:
            rotation_layer(qc, qc.qubits, params, axis=ax)
        if barriers:
            qc.barrier()


def su1_circuit(qc, params, layers, entangle="linear", final_rot=True, axis="y", barriers=False):
    warnings.warn("`su1_circuit' is deprecated, use `efficient_su` instead!", DeprecationWarning)
    axes = [axis]
    su_circuit(qc, params, layers, axes, entangle, final_rot, barriers)


def su2_circuit(qc, params, layers, entangle="linear", final_rot=True, axis1="x", axis2="y",
                barriers=False):
    warnings.warn("`su2_circuit' is deprecated, use `efficient_su` instead!", DeprecationWarning)
    axes = [axis1, axis2]
    su_circuit(qc, params, layers, axes, entangle, final_rot, barriers)


def su3_circuit(qc, params, layers, entangle="linear", final_rot=True, barriers=False):
    warnings.warn("`su3_circuit' is deprecated, use `efficient_su` instead!", DeprecationWarning)
    axes = ["x", "y", "z"]
    su_circuit(qc, params, layers, axes, entangle, final_rot, barriers)
