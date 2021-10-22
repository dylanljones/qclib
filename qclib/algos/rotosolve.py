# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import math
import numpy as np
from scipy.optimize import OptimizeResult
from typing import List, Sequence, Callable


def measure_expvals(params, idx, func, args, phi=0.):
    params[idx] = phi
    m_0 = func(params, *args)
    params[idx] = phi + np.pi / 2
    m_p = func(params, *args)
    params[idx] = phi - np.pi / 2
    m_m = func(params, *args)
    return m_0, m_p, m_m


def compute_coefficients(expvals, phi=0.):
    m_0, m_p, m_m = expvals
    term_1 = 2 * m_0 - m_p - m_m
    term_2 = m_p - m_m
    coeff_a = 0.5 * math.sqrt(term_1 ** 2 + term_2 ** 2)
    coeff_b = math.atan2(term_1, term_2) - phi
    coeff_c = 0.5 * (m_p + m_m)


class RotosolveOptimizer:

    def __init__(self, maxiter: int = 100, tol: float = None, phi: float = 0.,
                 callback: Callable = None):
        self.maxiter = maxiter
        self.tol = tol
        self.phi = phi
        self.callback = callback
        self._coeffs = [0., 0., 0.]        # A, B, C
        self._expectations = [0., 0., 0.]  # M_0, M_+, M_-
        self.costs = list()
        self.sub_costs = list()

    def _measure_expvals(self, params: np.ndarray, idx: int, func: Callable, args: tuple = ()):
        """Measures the three expectation values used to compute the rotosolve coefficients.

        Parameters
        ----------
        params : array_like of float
        idx : int
        func : callable
        args : tuple

        Returns
        -------
        expvals : list of float
            The expectation values used in the rotosolve procedure.
        """
        # compute epectation values
        params[idx] = self.phi
        m_0 = func(params, *args)
        params[idx] = self.phi + np.pi / 2
        m_p = func(params, *args)
        params[idx] = self.phi - np.pi / 2
        m_m = func(params, *args)
        # Store and return expectations
        self._expectations = [m_0, m_p, m_m]
        return self._expectations

    def _compute_coefficients(self) -> List[float]:
        r"""Computes the coefficients used by the rotosolve procedure.

        The parameters are defined as:
        ..math::
            A = \sqrt{(2⟨M⟩_{0} - ⟨M⟩_{π/2} - ⟨M⟩_{-π/2})² + (⟨M⟩_{π/2} - ⟨M⟩_{-π/2})²} / 2
            B = artctan2(2⟨M⟩_{0} - ⟨M⟩_{π/2} - ⟨M⟩_{-π/2}; ⟨M⟩_{π/2} - ⟨M⟩_{-π/2})
            C = (⟨M⟩_{π/2} + ⟨M⟩_{-π/2}) / 2

        Returns
        -------
        coeffs : list of float
            The coefficients used in the rotosolve procedure.
        """
        m_0, m_p, m_m = self._expectations
        term_1 = 2 * m_0 - m_p - m_m
        term_2 = m_p - m_m
        self._coeffs[0] = 0.5 * math.sqrt(term_1**2 + term_2**2)
        self._coeffs[1] = math.atan2(term_1, term_2) - self.phi
        self._coeffs[2] = 0.5 * (m_p + m_m)
        return self._coeffs

    def _expectation(self) -> float:
        """Returns the expectation value ⟨M⟩ using the most recent parameters.

        The expectation value can be computed using the coefficiens A and B:
        ..math::
            ⟨M⟩_{min} = -A + C

        Returns
        -------
        expval : float
            The expectation value of the operator.
        """
        return -self._coeffs[0] + self._coeffs[2]

    def _rotosolve(self, params: np.ndarray, idx: int, func: Callable, args: tuple = ()):
        # compute epectation values and rotosolve coefficients
        self._measure_expvals(params, idx, func, args)
        self._compute_coefficients()

        # Compute new parameter
        theta = self.phi - np.pi / 2 - self._coeffs[1]
        # Fix parameter to (pi, pi]
        if theta <= -np.pi:
            theta += 2 * np.pi
        elif theta > np.pi:
            theta -= 2 * np.pi
        # update parameter
        params[idx] = theta
        # Return new params and output of objective function
        return params, self._expectation()

    def step_and_cost(self, func: Callable, params: Sequence[float], args: tuple = ()):
        # Copy parameters
        x_min = np.array(params)
        # iterate through params and optimize
        sub_cost = list()
        for idx in range(len(x_min)):
            x_min, y_min = self._rotosolve(x_min, idx, func, args)
            sub_cost.append(y_min)
        y_min = func(params, *args)
        return x_min, y_min, sub_cost

    def step(self, func: Callable, params: Sequence[float], args: tuple = ()):
        x_min, *_ = self.step_and_cost(func, params, args)
        return x_min

    def minimize(self, func: Callable, args: tuple = (), x0: Sequence[float] = None,
                 num_params: int = 0):
        # Initialize parameters if not given
        if x0 is None:
            x0 = np.random.uniform(-np.pi, np.pi, num_params)

        x_min = list(x0)
        # Compute unoptimized objective function
        prev_cost = func(x_min, *args)

        # Run rotosolve
        cost = prev_cost
        self.costs = [prev_cost]
        self.sub_costs = [prev_cost]
        it, status = 0, 0
        message = "Optimization terminated successfully: Maximum iterations reached!"
        for it in range(self.maxiter):
            x_min, cost, sub_cost = self.step_and_cost(func, x_min, args)
            delta = abs(cost - prev_cost)
            # rdelta = delta / abs(prev_cost)

            self.costs.append(cost)
            self.sub_costs.extend(sub_cost)
            if self.callback is not None:
                self.callback(it, x_min, prev_cost, sub_cost, delta)
            if self.tol is not None and delta <= self.tol:
                status = 1
                message = "Optimization terminated successfully: Parameters converged!"
                break
            prev_cost = cost

        result = OptimizeResult(x=x_min, success=True, status=status, message=message,
                                fun=cost, nit=it+1)
        return result
