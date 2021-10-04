# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
from typing import Optional, Iterable, Union

# Pauli matrices
si = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
pauli = si, sx, sy, sz

# Initial states
ZERO = np.array([1, 0])
ONE = np.array([0, 1])
PLUS = np.array([1, 1]) / np.sqrt(2)
MINUS = np.array([1, -1]) / np.sqrt(2)
IPLUS = np.array([1, 1j]) / np.sqrt(2)
IMINUS = np.array([1, -1j]) / np.sqrt(2)
STATES = {"0": ZERO, "1": ONE, "+": PLUS, "-": MINUS, "i+": IPLUS, "i-": IMINUS}

# Projections onto |0> and |1>
P0 = np.dot(ZERO[:, np.newaxis], ZERO[np.newaxis, :])
P1 = np.dot(ONE[:, np.newaxis], ONE[np.newaxis, :])
PROJECTIONS = [P0, P1]


def abs2(arr) -> np.ndarray:
    """Computes the element-wise absolute square of the input array.

    Parameters
    ----------
    arr : array_like
        The input array.

    Returns
    -------
    out : np.ndarray
        Element-wise |x*x|, of the same shape and dtype as x. This is a scalar if x is a scalar.

    Examples
    --------
    >>> abs2([-1j, 0.5])
    array([1.  , 0.25])
    """
    return np.square(np.abs(arr))


def kron(*args) -> np.ndarray:
    """Computes the Kronecker product of two or more arrays.

    Parameters
    ----------
    *args : list of array_like or array_like
        Input arrays.

    Returns
    -------
    out : np.ndarray
        The Kronecker product of the input arrays.

    Examples
    --------
    >>> kron([1, 0], [1, 0])
    array([1, 0, 0, 0])
    >>> kron([[1, 0], [1, 0]])
    array([1, 0, 0, 0])
    >>> kron([1, 0], [1, 0], [1, 0])
    array([1, 0, 0, 0, 0, 0, 0, 0])
    """
    if len(args) == 1:
        args = args[0]
    x = 1
    for arg in args:
        x = np.kron(x, arg)
    return x


def density_matrix(psi: np.ndarray) -> np.ndarray:
    r"""Computes the density matrix ρ of a given state-vector |ψ>.

    .. math::
        ρ = |ψ><ψ|

    Parameters
    ----------
    psi: (N) np.ndarray
        The input state-vector ψ.

    Returns
    -------
    rho : (N, N) np.ndarray
        The density matrix ρ.

    Examples
    --------
    >>> density_matrix(np.array([1, 0]))
    array([[1, 0],
           [0, 0]])
    >>> density_matrix(np.array([1, 0]))
    array([[1, 0],
           [0, 0]])
    >>> density_matrix(np.array([1j, 0]))
    array([[1.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j]])
    """
    return np.dot(psi[:, np.newaxis], np.conj(psi[np.newaxis, :]))


def get_projector(psi: np.ndarray):
    r"""Constructs a projection-matrix from a given vector.

    .. math::
        P = |ψ><ψ|

    Parameters
    ----------
    psi : (N) np.ndarray
        The state-vector used for constructing the projector.

    Returns
    -------
    p : (N, N) np.ndarray
    """
    if len(psi.shape) == 1:
        psi = psi[:, np.newaxis]
    return np.dot(psi, np.conj(psi).T)


def project_single(op: np.ndarray, num_qubits: int, qubit: int) -> np.ndarray:
    """Get the projection of the state-vector on a given single-qubit operator

    Parameters
    ----------
    op : (2, 2) array_like
        The single qubit operator.
    num_qubits : int
        The total number of qubits.
    qubit : int
        Index of the qubit.


    Returns
    -------
    proj: (N, N) np.ndarray
        The projection of the operator onto a single qubit of ψ.
    """
    parts = [np.eye(2)] * num_qubits
    parts[qubit] = op
    return kron(*parts)


def apply_statevec(psi: np.ndarray, op: np.ndarray) -> np.ndarray:
    """Applies a operator O to a state-vector |ψ>.

    .. math::
        |ψ'> = O |ψ>

    Parameters
    ----------
    psi : (N) np.ndarray
        The input state-vector |ψ>.
    op : (N, N) np.ndarray
        The operator O to apply on the state-vector.

    Returns
    -------
    out : (N) np.ndarray
        The output state-vector after applying the operator.
    """
    return op @ psi


def apply_density(rho: np.ndarray, op: np.ndarray) -> np.ndarray:
    """Applies a operator O to a density matrix ρ.

    .. math::
        ρ' = O ρ O^†

    Parameters
    ----------
    rho : (N) np.ndarray
        The input density matrix ρ.
    op : (N, N) np.ndarray
        The operator O to apply on the density matrix.

    Returns
    -------
    out : (N) np.ndarray
        The output density matrix ρ' after applying the operator.
    """
    op_conj = np.conj(op).T
    return op @ rho @ op_conj


def ptrace(arr: np.ndarray, n: int, m: int, index: int) -> np.ndarray:
    """Computes the partial trace over a sub-system of a direct product.

    Parameters
    ----------
    arr : (N, N) np.ndarray
        The input matrix. The size N of the array must be the product of the two sub-systems n*m.
    n : int
        The size of the first sub-system.
    m : int
        The size of the second sub-system.
    index : int
        The index of the traced out sub-system.

    Returns
    -------
    trace : (M, M) np.ndarray
        The traced out array.

    Examples
    --------
    >>> a = np.diag([0, 1, 2, 3])
    >>> ptrace(a, 2, 2, index=0)
    array([[2, 0],
           [0, 4]])
    >>> ptrace(a, 2, 2, index=1)
    array([[1, 0],
           [0, 5]])
    """
    if n * m != arr.shape[0]:
        raise ValueError("Sizes of sub-systems don't add up to shape of array")

    reshaped = arr.reshape((n, m, n, m))
    if index == 0:
        return np.trace(reshaped, axis1=0, axis2=2)
    else:
        return np.trace(reshaped, axis1=1, axis2=3)


def expectation(o: np.ndarray, psi: np.ndarray) -> np.ndarray:
    r"""Computes the expectation value of an operator in a given state.

    .. math::
        x = \langle \Psi| \hat{O} |\Psi \rangle

    Parameters
    ----------
    o : (N, N) np.ndarray
        Operator in matrix representation.
    psi : (N) np.ndarray
        State in vector representation.

    Returns
    -------
    x : float
    """
    return np.dot(np.conj(psi).T, o.dot(psi)).real


# =========================================================================


def binstr(num: int, width: Optional[int] = None) -> str:
    """ Returns the binary representation of an integer.

    Parameters
    ----------
    num : int
        The number to convert.
    width: int, optional
        Minimum number of digits used. The default is the global value `BITS`.

    Returns
    -------
    binstr : str
    """
    fill = width or 0
    return f"{num:0{fill}b}"


def binarr(num: int, width: Optional[int] = None,
           dtype: Optional[Union[int, str]] = None) -> np.ndarray:
    """ Returns the bits of an integer as a binary array.

    Parameters
    ----------
    num : int
        The number representing the binary state.
    width : int, optional
        Minimum number of digits used. The default is the global value `BITS`.
    dtype : int or str, optional
        An optional datatype-parameter. The default is the global value `DTYPE`.

    Returns
    -------
    binarr : np.ndarray
    """
    fill = width or 0
    return np.fromiter(f"{num:0{fill}b}"[::-1], dtype=dtype or np.int16)


def binidx(num: int, width: Optional[int] = None) -> Iterable[int]:
    """ Returns the indices of bits with the value `1`.

    Parameters
    ----------
    num : int
        The number representing the binary state.
    width : int, optional
        Minimum number of digits used. The default is the global value `BITS`.

    Returns
    -------
    binidx : list
    """
    fill = width or 0
    return list(sorted(i for i, char in enumerate(f"{num:0{fill}b}"[::-1]) if char == "1"))


class PauliView:

    def __init__(self, matrices, mat):
        self._matrices = matrices
        self._mat = mat

    def _build(self, index, mat):
        matrices = self._matrices.copy()
        matrices[index, :, :] = mat
        return kron(*matrices)

    def __getitem__(self, index):
        return self._build(index, self._mat)

    def __call__(self, index):
        return self._build(index, self._mat)


class PauliOperators:

    def __init__(self, num):
        self.num = num
        self._matrices = self._init_empty()

    @property
    def x(self):
        return PauliView(self._matrices, sx)

    @property
    def y(self):
        return PauliView(self._matrices, sy)

    @property
    def z(self):
        return PauliView(self._matrices, sz)

    def _init_empty(self):
        shape = (self.num, 2, 2)
        idx = np.arange(2)
        matrices = np.zeros(shape, dtype=np.complex)
        matrices[:, idx, idx] = 1
        return matrices

    def _build(self, index, mat):
        matrices = self._matrices.copy()
        matrices[index, :, :] = mat
        return kron(*matrices)
