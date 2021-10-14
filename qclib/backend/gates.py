# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
from scipy.linalg import expm
from itertools import product
from ..math import kron, P0, P1

# ======================== SINGLE QUBIT GATES =======================

X_GATE = np.array([[0, 1], [1, 0]])
Y_GATE = np.array([[0, -1j], [1j, 0]])
Z_GATE = np.array([[1, 0], [0, -1]])
HADAMARD_GATE = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
PHASE_GATE = np.array([[1, 0], [0, 1j]])
T_GATE = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])


def id_gate(*args):
    """Functional for the single-qubit identity gate."""
    return np.eye(2)


def x_gate(*args):
    """Functional for the single-qubit Pauli-X gate."""
    return X_GATE


def y_gate(*args):
    """Functional for the single-qubit Pauli-Y gate."""
    return Y_GATE


def z_gate(*args):
    """Functional for the single-qubit Pauli-Z gate."""
    return Z_GATE


def h_gate(*args):
    """Functional for the single-qubit Hadamard (H) gate."""
    return HADAMARD_GATE


def s_gate(*args):
    """Functional for the single-qubit phase (S) gate."""
    return PHASE_GATE


def t_gate(*args):
    """Functional for the single-qubit T gate."""
    return T_GATE


def rx_gate(phi: float = 0):
    """Functional for the single-qubit Pauli-X rotation-gate.

    Parameters
    ----------
    phi : float
        Rotation angle (in radians)

    Returns
    -------
    rx : (2, 2) np.ndarray
    """
    arg = phi / 2
    return np.array([[np.cos(arg), -1j*np.sin(arg)], [-1j*np.sin(arg), np.cos(arg)]])


def ry_gate(phi: float = 0):
    """Functional for the single-qubit Pauli-Y rotation-gate.

    Parameters
    ----------
    phi : float
        Rotation angle (in radians)

    Returns
    -------
    ry : (2, 2) np.ndarray
    """
    arg = phi / 2
    return np.array([[np.cos(arg), -np.sin(arg)], [+np.sin(arg), np.cos(arg)]])


def rz_gate(phi: float = 0):
    """Functional for the single-qubit Pauli-Z rotation-gate.

    Parameters
    ----------
    phi : float
        Rotation angle (in radians)

    Returns
    -------
    rz : (2, 2) np.ndarray
    """
    arg = 1j * phi / 2
    return np.array([[np.exp(-arg), 0], [0, np.exp(arg)]])


def single_gate(qbits, gates, num_qubits=None):
    """Builds the matrix of a single-qubit gate.

    Parameters
    ----------
    qbits : int or Sequence of int
        Index of the qubits.
    gates : list of ndarray or ndarray
        Matrix-representation of the single qubit gate.
        A matrix for each qubit index in 'qbits' must be given.
    num_qubits : int, optional
        Total number of qubits. If not specified number of involved qubits is used.

    Returns
    -------
    gate : np.ndarray
        The full matrix of the single-qubit gate.
    """
    if not hasattr(qbits, "__len__"):
        qbits = [qbits]
        gates = [gates]
    if num_qubits is None:
        num_qubits = max(qbits) + 1

    arrays = np.zeros((num_qubits, 2, 2), dtype=np.complex64)
    idx = [0, 1]
    arrays[:, idx, idx] = 1

    for idx, gate in zip(qbits, gates):
        arrays[idx] = gate
    return kron(arrays)


def _get_projection(*items, num_qubits):
    parts = [np.eye(2)] * num_qubits
    for idx, proj in items:
        parts[idx] = proj
    return parts


def cgate(control, target, gate, num_qubits=None, trigger=1):
    """Builds matrix of a n-qubit control gate.

    Parameters
    ----------
    control : int or list of int
        Index of control-qubit(s)
    target : int
        Index of target qubit
    gate : array_like
        Matrix of the gate that is controlled
    num_qubits : int, optional
        Total number of qubits. If not specified
        number of involved qubits is used
    trigger : int, optional
        Value of control qubits that triggers gate. The default is 1.

    Returns
    -------
    gate : np.ndarray
    """
    if not hasattr(control, "__len__"):
        control = [control]
    num_qubits = num_qubits or max(*control, target) + 1
    gate = np.asarray(gate)
    array = 0
    for vals in product([0, 1], repeat=len(control)):
        # Projections without applying gate
        projections = [P1 if x else P0 for x in vals]
        items = list(zip(control, projections))
        if np.all(vals) == trigger:
            # Projection with applying gate
            items.append((target, gate))
        # Build projection array and add to matrix
        array = array + kron(_get_projection(*items, num_qubits=num_qubits))
    return array
