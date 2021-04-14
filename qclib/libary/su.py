# coding: utf-8
#
# This code is part of qclib.
# 
# Copyright (c) 2021, Dylan Jones

from qiskit import QuantumCircuit


def rotation_layer(qc, qubits, params, axis="y"):
    for qbit in qubits:
        getattr(qc, "r" + axis)(next(params), qbit)


def entanglement_layer(qc, mode="linear"):
    if mode == "linear":
        for i in range(qc.num_qubits - 1):
            qc.cx(i, i + 1)
    else:
        for con in range(qc.num_qubits - 1):
            for target in range(con + 1, qc.num_qubits):
                qc.cx(con, target)


def su_circuit(qc, params, layers, axes, entangle="linear", final_rot=True, barriers=False):
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
    axes = [axis]
    su_circuit(qc, params, layers, axes, entangle, final_rot, barriers)


def su2_circuit(qc, params, layers, entangle="linear", final_rot=True, axis1="x", axis2="y",
                barriers=False):
    axes = [axis1, axis2]
    su_circuit(qc, params, layers, axes, entangle, final_rot, barriers)


def su3_circuit(qc, params, layers, entangle="linear", final_rot=True, barriers=False):
    axes = ["x", "y", "z"]
    su_circuit(qc, params, layers, axes, entangle, final_rot, barriers)


class TwoLocal(QuantumCircuit):

    def __init__(self, *regs,
                 reps: int = 1,
                 insert_barriers: bool = False,
                 skip_final_rot: bool = False,
                 name=None):
        super().__init__(*regs, name=name)
        self._insert_barriers = insert_barriers
        self._skip_final_rotation_layer = skip_final_rot
        self.reps = reps
        self.num_control_layers = reps
        if not self._skip_final_rotation_layer:
            self.num_control_layers += 1

    def _build_rotation_layer(self, param_iter, lvl) -> QuantumCircuit:
        pass

    def _build_entanglement_layer(self, param_iter, lvl) -> QuantumCircuit:
        pass

    def build(self, params):
        param_iter = iter(params)
        for i in range(self.reps):
            if self._insert_barriers:
                self.barrier()
            self += self._build_rotation_layer(param_iter, i)  # noqa

            if self._insert_barriers:
                self.barrier()

            self += self._build_entanglement_layer(param_iter, i)  # noqa

        # add the final rotation layer
        if not self._skip_final_rotation_layer:
            if self._insert_barriers:
                self.barrier()
            self += self._build_rotation_layer(param_iter, self.reps)  # noqa


class SU2Circuit(TwoLocal):

    def __init__(self, *regs,
                 reps: int = 1,
                 insert_barriers: bool = False,
                 skip_final_rot: bool = False,
                 entangle: str = "linear",
                 name=None):
        super().__init__(*regs, reps=reps, insert_barriers=insert_barriers,
                         skip_final_rot=skip_final_rot, name=name)
        self.entangle = entangle

    @property
    def num_control_gates(self):
        return 2 * self.num_qubits * self.num_control_layers

    def _build_rotation_layer(self, param_iter, lvl):
        layer = QuantumCircuit(*self.qregs)

        for i in range(self.num_qubits):
            layer.ry(next(param_iter), i)
        for i in range(self.num_qubits):
            layer.rz(next(param_iter), i)

        return layer

    def _build_entanglement_layer(self, param_iter, lvl):
        layer = QuantumCircuit(*self.qregs)
        if self.entangle == "linear":
            for i in range(self.num_qubits - 1):
                layer.cx(i, i+1)
        else:
            for con in range(self.num_qubits - 1):
                for target in range(con+1, self.num_qubits):
                    layer.cx(con, target)
        return layer


class SU3Circuit(TwoLocal):

    def __init__(self, *regs,
                 reps: int = 1,
                 insert_barriers: bool = False,
                 skip_final_rot: bool = False,
                 entangle: str = "linear",
                 name=None):
        super().__init__(*regs, reps=reps, insert_barriers=insert_barriers,
                         skip_final_rot=skip_final_rot, name=name)
        self.entangle = entangle

    @property
    def num_control_gates(self):
        return 3 * self.num_qubits * self.num_control_layers

    def _build_rotation_layer(self, param_iter, lvl):
        layer = QuantumCircuit(*self.qregs)

        for i in range(self.num_qubits):
            layer.rx(next(param_iter), i)
        for i in range(self.num_qubits):
            layer.ry(next(param_iter), i)
        for i in range(self.num_qubits):
            layer.rz(next(param_iter), i)
        return layer

    def _build_entanglement_layer(self, param_iter, lvl):
        layer = QuantumCircuit(*self.qregs)
        if self.entangle == "linear":
            for i in range(self.num_qubits - 1):
                layer.cx(i, i+1)
        else:
            for con in range(self.num_qubits - 1):
                for target in range(con+1, self.num_qubits):
                    layer.cx(con, target)
        return layer
