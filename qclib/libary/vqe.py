# coding: utf-8
#
# This code is part of qclib.
# 
# Copyright (c) 2021, Dylan Jones

from qiskit import QuantumCircuit


class TwoLocal(QuantumCircuit):

    def __init__(self, *regs,
                 reps: int = 1,
                 insert_barriers: bool = False,
                 skip_final_rot: bool = False,
                 name=None):
        super().__init__(*regs, name=name)
        self.reps = reps
        self._insert_barriers = insert_barriers
        self._skip_final_rotation_layer = skip_final_rot

    def _build_rotation_layer(self, param_iter, lvl):
        pass

    def _build_entanglement_layer(self, param_iter, lvl):
        pass

    def build(self, param_iter):
        for i in range(self.reps):
            if self._insert_barriers:
                self.barrier()
            self._build_rotation_layer(param_iter, i)

            if self._insert_barriers:
                self.barrier()

            self._build_entanglement_layer(param_iter, i)

        # add the final rotation layer
        if not self._skip_final_rotation_layer:
            if self._insert_barriers:
                self.barrier()
            self._build_rotation_layer(param_iter, self.reps)


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

    def _build_rotation_layer(self, param_iter, lvl):
        layer = QuantumCircuit(*self.qregs)

        for i in range(self.num_qubits):
            layer.ry(next(param_iter), i)
        for i in range(self.num_qubits):
            layer.rz(next(param_iter), i)

        # add the layer to the circuit
        self += layer  # noqa

    def _build_entanglement_layer(self, param_iter, lvl):
        layer = QuantumCircuit(*self.qregs)
        if self.entangle == "linear":
            for i in range(self.num_qubits - 1):
                layer.cx(i, i+1)
        else:
            for con in range(self.num_qubits - 1):
                for target in range(con+1, self.num_qubits):
                    layer.cx(con, target)

        # add the layer to the circuit
        self += layer  # noqa
