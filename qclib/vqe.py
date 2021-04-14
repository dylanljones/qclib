# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import qiskit
import numpy as np
import scipy.optimize
from scipy import optimize
from abc import ABC, abstractmethod
from typing import Union, Optional, Sequence, Iterable
from .circuit import run, Result
from .libary.su import su_circuit


class OptimizationError(Exception):

    def __init__(self, sol):
        self.status = sol.status
        self.nfev = sol.nfev
        super().__init__("Optimization unsuccessful: " + sol.message)


class VariationalSolver(ABC):
    """Base class for variational quantum solvers

    Parameters
    ----------
    shots : int, optional
        The number of shots for running the experiments.
    backend : str or qiskit.providers.Backend, optional
        The backend for running the experiments. See ``run`` for the default backend.

    """
    def __init__(self, shots: Optional[int] = 1024,
                 backend: Optional[qiskit.providers.Backend] = None):
        self.backend = backend
        self.shots = shots
        self._sol = None

    @property
    def sol(self) -> scipy.optimize.OptimizeResult:
        """scipy.optimize.OptimizeResult : The optimization result. """
        if self._sol is None:
            raise ValueError("Circuit has not been optimized yet!")
        return self._sol

    def set_shots(self, shots: int) -> None:
        """Sets the number of shots for running the circuit.

        Parameters
        ----------
        shots : int
            The number of times the circuit is run.
        """
        self.shots = shots

    def set_backend(self, backend: qiskit.providers.Backend):
        """Sets the backend used for running the circuit.

        Parameters
        ----------
        backend : str or qiskit.providers.Backend, optional
            The backend for running the experiments. See ``run`` for the default backend.
        """
        self.backend = backend

    @abstractmethod
    def num_params(self) -> int:
        """int : Retunrs the number of controllable parameters of the circuit."""
        pass

    @abstractmethod
    def build_circuit(self, args: Iterable[float]) -> qiskit.QuantumCircuit:
        """Constructs the ``QuantumCircuit``-ansatz of the solver.

        Parameters
        ----------
        args : Iterable of float
            An iterator of the circuit parameters. These parameters are used by the optimizer.

        Returns
        -------
        qc : QuantumCircuit
            The parameterized ``QuantumCircuit``.
        """
        pass

    @abstractmethod
    def cost_function(self, probabilities) -> Union[np.ndarray, float]:
        """Computes the cost function that is minimized by the optimizer.

        Parameters
        ----------
        probabilities : np.ndarray
            The probability distribution of the measurement result of the ``QuantumCircuit``.

        Returns
        -------
        cost : np.ndarray or float
            The cost value.
        """
        pass

    def run(self, args: Sequence[float]) -> Result:
        """Runs the ``QuantumCircuit``-ansatz with the passed arguments.

        Parameters
        ----------
        args : Sequence of float
            The parameters used in the ``QuantumCircuit``.

        Returns
        -------
        result : Result
            The result of running the experiment.
        """
        qc = self.build_circuit(iter(args))
        return run(qc, shots=self.shots, backend=self.backend)

    def _func(self, args):
        """Runs the ``QuantumCircuit`` and computes the cost value.

        This method is used by the optimizer and shouldn't be called by the user.
        """
        res = self.run(args)
        return self.cost_function(res.probabilities)

    def optimize(self, tol: Optional[float] = None,
                 x0: Optional[Union[Sequence[float], np.ndarray]] = None,
                 bounds: Optional[Union[Sequence[float], np.ndarray]] = None,
                 maxiter: Optional[int] = 1000,
                 method: Optional[str] = "COBYLA",
                 raise_error: Optional[bool] = True,
                 **options) -> scipy.optimize.OptimizeResult:
        """Optimizes the ``QuantumCircuit``-ansatz defined by the ``build_circuit``-method.

        Parameters
        ----------
        tol : float, optional
            Tolerance for termination. For detailed control, use solver-specific options.
        x0: (N, ) array_like, optional
            Initial guess. Array of real elements of size (n,), where ‘n’ is the number
            of independent variables. If ``None`` a random array is used.
        bounds : array_like, optional
            Bounds of the variables to optimize. The default is ``None``.
        maxiter : int, optional
            Maximum number of iterations to perform. Depending on the method each iteration
            may use several function evaluations.
        method : str, optional
            Type of solver used for optimizing.
        raise_error : bool, optional
            If ``True`` an exception is raised if the optimization was unsuccessful.
        **options
            Solver specific options.

        Raises
        ------
        OptimizationError
            Raised if the optimization was unsuccessful and ``raise_error==True``.

        Returns
        -------
        sol : scipy.optimize.OptimizeResult
            The optimization result of the ``QuantumCircuit``-anatz.
        """
        # Prepare arguments
        if x0 is None:
            x0 = np.random.uniform(0, np.pi, size=self.num_params())

        # Optimize circuit
        sol = optimize.minimize(self._func, x0=x0, method=method,
                                tol=tol, bounds=bounds,
                                options={"maxiter": maxiter, **options})
        if raise_error and not sol.success:
            raise OptimizationError(sol)

        # Save and return optimization result
        self._sol = sol
        return sol

    def run_optimized(self) -> Result:
        """Result : Runs the circuit and returns the result using the optimized parameters."""
        return self.run(self.sol.x)

    def get_optimized_circuit(self) -> qiskit.QuantumCircuit:
        """QuantumCircuit : Construct the ``QuantumCircuit`` with the optimized parameters."""
        return self.build_circuit(self.sol.x)

    def get_optimized_probabilities(self) -> np.ndarray:
        """np.ndarray : Runs the optimized circuit and returns the probability distribution."""
        return self.run_optimized().probabilities


class VQE(VariationalSolver):

    def __init__(self, target, rot, layers=2, entangle="linear", final_rot=False,
                 shots=1024, backend=None):
        super().__init__(shots, backend)
        self.num_qubits = int(np.log2(target.shape[0]))
        self.layers = layers
        self.entangle_mode = entangle
        self.rot_axis = list(rot)
        self.final_rot = final_rot

        self.target = target

    def set_target(self, target):
        self.target = target
        self._sol = None

    def num_params(self) -> int:
        layers = self.layers
        if self.final_rot:
            layers += 1
        return len(self.rot_axis) * layers * self.num_qubits

    def cost_function(self, probabilities):
        return np.sum(np.abs(probabilities - self.target))

    def build_circuit(self, args) -> qiskit.QuantumCircuit:
        params = iter(args)
        qc = qiskit.QuantumCircuit(self.num_qubits)
        su_circuit(qc, params, self.layers, self.rot_axis, self.entangle_mode, self.final_rot)
        qc.measure_all()
        return qc
