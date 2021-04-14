# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

from abc import ABC, abstractmethod
from qiskit.circuit import QuantumCircuit, Instruction


class AbstractTimeEvolutionCircuit(QuantumCircuit, ABC):

    def __init__(self, *regs, dt: float, name: str = "Trotter", **kwargs):
        super().__init__(*regs, name=name)
        self._trotter = None
        self.dt = dt
        self._name = name
        self._num_steps = 0

    @property
    def num_trotter_steps(self) -> int:
        """int: The current number of Trotter steps included in the circuit."""
        return self._num_steps

    @abstractmethod
    def define_trotter_step(self, qc: QuantumCircuit, dt: float) -> None:
        """Defines the circuit of a single Trotter step.

        Parameters
        ----------
        qc: QuantumCircuit
            The ``QuantumCircuit`` for defining the Trotter step.
        dt: float
            The time interval of the step
        """
        ...

    def _get_trotter_step(self) -> Instruction:
        """Returns the (buffered) sub-circuit of a single Trotter step.

        Returns
        -------
        instructions: Instruction
            The instructions of a single Trotter step.
        """
        if self._trotter is None:
            qc = QuantumCircuit(self.num_qubits, name="Trotter step")
            self.define_trotter_step(qc, self.dt)
            self._trotter = qc.to_instruction()
        return self._trotter

    def trotter_step(self, qubits=None) -> None:
        """Builds and appends a single Trotter step.

        Parameters
        ----------
        qubits: Qubit or QuantumRegister, optional
            The step will include these qubits. If None, all qubits
            of the circuits are used. The default is None
        """
        if qubits is None:
            qubits = self.qubits[:]

        # Create and append sub-circuit
        qc = self._get_trotter_step().copy(f"Trotter {self._num_steps}")
        self.append(qc, qubits)
        # Update number of Trotter steps
        self._num_steps += 1
        # Rename circuit
        self.name = f"{self._name}({self._num_steps})"

    def trotter_evolution(self, num_steps, qubits=None) -> None:
        """Builds and appends multiple Trotter steps.

        Parameters
        ----------
        num_steps: int
            The number of Trotter steps.
        qubits: Qubit or QuantumRegister, optional
            The step will include these qubits. If None, all qubits
            of the circuits are used. The default is None
        """
        if qubits is None:
            qubits = self.qubits[:]
        for _ in range(num_steps):
            self.trotter_step(qubits)
