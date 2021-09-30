# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import math
import logging
import numpy as np
from ..math import kron, ZERO, ONE

logger = logging.getLogger(__name__)

COMP_EIGVALS = np.array([0, 1])
COMP_EIGVECS = np.array([ZERO, ONE]).T


def get_eigbasis(eigvals=None, eigvecs=None):
    if eigvals is None:
        eigvals = COMP_EIGVALS
        eigvecs = COMP_EIGVECS
    else:
        if eigvecs is None:
            raise ValueError("No Eigenvectors of the measurement-basis are specified!")
    return eigvals, eigvecs


def get_projector(num_qubits, qubit, vec):
    """Builds a projector of a single qubit vector."""
    parts = [np.eye(2) for _ in range(num_qubits)]
    parts[qubit] = np.dot(vec[:, np.newaxis], np.conj(vec[np.newaxis, :]))
    return kron(*parts)


def measure_qubit(psi, qubit, eigvals=None, eigvecs=None):
    r"""Measure the state of a single qubit in a given eigenbasis.

    The probability .math:'p_i' of measuring each eigenstate of the measurement-eigenbasis
    is calculated using the projection .math:'P_i' of the corresponding eigenvector .math:'v_i'

    .. math::
        p(i) = <ψ|P_i|ψ>     P_i = |v_i><v_i|

    The probabilities are used to determine the corresponding eigenvalue .math:'\lambda_i'
    which is the final measurement result. The state after the measurement is defined as:

    .. math::
        |ψ_{new}> = \frac{P_i |ψ>}{\sqrt{<ψ|P_i|ψ>}}

    Parameters
    ----------
    psi : (N) array_like
        The input state-vector ψ.
    qubit : int
        The qubit that is measured
    eigvals : np.ndarray, optional
        The eigenvalues of the basis in which is measured.
        The default is the computational basis with eigenvalues '0' and '1'.
    eigvecs : np.ndarray, optional
        The corresponding eigenvectors of the basis in which is measured.
        The default is the computational basis with eigenvectors '(1, 0)' and '(0, 1)'.

    Returns
    -------
    result : float
        Eigenvalue corresponding to the measured eigenstate.
    psi_post : (N) np.ndarry
        The post-measurement state vector |ψ_{new}>.

    Examples
    --------
    >>> measure_qubit(np.array([1, 0, 0, 0]), qubit=0)
    (0, array([1., 0., 0., 0.]))
    >>> measure_qubit(np.array([0, 0, 1, 0]), qubit=0)
    (1, array([0., 0., 1., 0.]))
    """
    eigvals, eigvecs = get_eigbasis(eigvals, eigvecs)
    v0, v1 = eigvecs.T

    num_qubits = int(math.log2(len(psi)))

    # Calculate probability of getting first eigenstate as result
    proj0 = get_projector(num_qubits, qubit, v0)
    projected = np.dot(proj0, psi)

    p = np.dot(np.conj(psi).T, projected)
    if abs(p.imag) > 1e-15:
        raise RuntimeError(f"Complex probability: {p}")

    # Simulate measurement probability
    index = int(np.random.random() > p.real)
    if index == 1:
        p = 1 - p
        # Project state to other eigenstate of the measurement basis
        proj1 = get_projector(num_qubits, qubit, v1)
        projected = np.dot(proj1, psi)

    logger.debug("Result: %s (%.4f)", index, p)
    # Project measurement result on the state
    psi_post = projected / np.linalg.norm(projected)

    return eigvals[index].real, psi_post


def measure_qubit_rho(rho, qubit, eigvals=None, eigvecs=None):
    r"""Measure the state of a single qubit in a given eigenbasis for a density matrix.

    The probability .math:'p_i' of measuring each eigenstate of the measurement-eigenbasis
    is calculated using the projection .math:'P_i' of the corresponding eigenvector .math:'v_i'

    .. math::
        p(i) = Tr(ρ P_i)     P_i = |v_i><v_i|

    The probabilities are used to determine the corresponding eigenvalue .math:'\lambda_i'
    which is the final measurement result. The state after the measurement is defined as:

    .. math::
        ρ_{new} = \frac{P_i ρ P_i^†}{Tr(ρ P_i)}

    Parameters
    ----------
    rho : (N) array_like
        The input density matrix ρ.
    qubit : int
        The qubit that is measured
    eigvals : np.ndarray, optional
        The eigenvalues of the basis in which is measured.
        The default is the computational basis with eigenvalues '0' and '1'.
    eigvecs : np.ndarray, optional
        The corresponding eigenvectors of the basis in which is measured.
        The default is the computational basis with eigenvectors '(1, 0)' and '(0, 1)'.

    Returns
    -------
    result : float
        Eigenvalue corresponding to the measured eigenstate.
    rho_post : (N) np.ndarry
        The post-measurement density matrix ρ_{new}.

    Examples
    --------
    """
    eigvals, eigvecs = get_eigbasis(eigvals, eigvecs)
    v0, v1 = eigvecs.T

    num_qubits = int(math.log2(len(rho)))

    # Calculate probability of getting first eigenstate as result
    projector = get_projector(num_qubits, qubit, v0)

    p = np.trace(projector @ rho)
    if abs(p.imag) > 1e-15:
        raise RuntimeError(f"Complex probability: {p}")

    # Simulate measurement probability
    index = int(np.random.random() > p.real)
    if index == 1:
        p = 1 - p
        # Project state to other eigenstate of the measurement basis
        projector = get_projector(num_qubits, qubit, v1)
    logger.debug("Result: %s (%.4f)", index, p)
    # Project measurement result on the state
    rho_post = (projector @ rho @ np.conj(projector).T) / np.trace(projector @ rho)

    return eigvals[index].real, rho_post
