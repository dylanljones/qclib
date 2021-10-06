# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
from scipy import linalg as la
from functools import partial
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

transpose = partial(np.swapaxes, axis1=-2, axis2=-1)


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


def hermitian(a):
    """Returns the hermitian of an array.

    Parameters
    ----------
    a : array_like
        The input array.

    Returns
    -------
    hermitian : np.ndarray
    """
    return np.conj(np.transpose(a))


def is_hermitian(a, rtol=1e-5, atol=1e-8, equal_nan=False):
    """Checks if an array is hermitian.

    Checks if the hermitian of the given array equals the array within
    a tolerance by calling np.allcloe.

    Parameters
    ----------
    a : array_like
        The array to check.
    rtol : float, optional
        The relative tolerance parameter used in the comparison (see np.allclose).
    atol : float, optional
        The absolute tolerance parameter used in the comparison (see np.allclose).
    equal_nan : bool, optional
        Flag if ``NaN``'s are compared.

    Returns
    -------
    hermitian : bool
    """
    return np.allclose(a, np.conj(np.transpose(a)), rtol, atol, equal_nan)


def fill_diagonal(mat, val, offset=0):
    """Fills the optionally offset diagonal of an array.

    Parameters
    ----------
    mat : np.ndarray
        Array whose diagonal is to be filled, it gets modified in-place.
    val : scalar or array_like
        Value to be written on the diagonal, its type must be compatible
        with that of the array a.
    offset : int, optional
        offset index of diagonal. An offset of 1 results
        in the first upper offdiagonal of the matrix,
        an offset of -1 in the first lower offdiagonal.
    """
    if offset:
        n, m = mat.shape
        if offset > 0:
            # Upper off-diagonal
            idx = slice(0, n), slice(abs(offset), m)
        else:
            # Lower off-diagonal
            idx = slice(abs(offset), n), slice(0, m)
        np.fill_diagonal(mat[idx], val)

    else:
        np.fill_diagonal(mat, val)


def decompose(arr, h=None):
    """Computes the eigen-decomposition of a matrix.

    Finds the eigenvalues and the left and right eigenvectors of a matrix. If the matrix
    is hermitian ``eigh`` is used for computing the right eigenvectors.

    Parameters
    ----------
    arr : (N, N) array_like
        A complex or real matrix whose eigenvalues and eigenvectors will be computed.
    h : bool, optional
        Flag if matrix is hermitian. If ``None`` the matrix is checked. This determines the
        scipy method used for computing the eigenvalue problem.

    Returns
    -------
    vr : (N, N) np.ndarray
        The right eigenvectors of the matrix.
    xi : (N, ) np.ndarray
        The eigenvalues of the matrix.
    vl : (N, N) np.ndarray
        The left eigenvectors of the matrix.
    """
    if h is None:
        h = is_hermitian(arr)
    if h:
        xi, vr = np.linalg.eigh(arr)
        vl = np.conj(vr.T)
    else:
        xi, vl, vr = la.eig(arr, left=True, right=True)
    return vr, xi, vl


def reconstruct(rv, xi, rv_inv, method='full'):
    """Computes a matrix from an eigen-decomposition.

    The matrix is reconstructed using eigenvalues and left and right eigenvectors.

    Parameters
    ----------
    rv : (N, N) np.ndarray
        The right eigenvectors of a matrix.
    xi : (N, ) np.ndarray
        The eigenvalues of a matrix
    rv_inv : (N, N) np.ndarray
        The left eigenvectors of a matrix.
    method : str, optional
        The mode for reconstructing the matrix. If mode is 'full' the original matrix
        is reconstructed, if mode is 'diag' only the diagonal elements are computed.
        The default is 'full'.

    Returns
    -------
    arr : (N, N) np.ndarray
        The reconstructed matrix.
    """
    method = method.lower()
    if 'diag'.startswith(method):
        arr = ((transpose(rv_inv) * rv) @ xi[..., np.newaxis])[..., 0]
    elif 'full'.startswith(method):
        arr = (rv * xi[..., np.newaxis, :]) @ rv_inv
    else:
        arr = np.einsum(method, rv, xi, rv_inv)
    return arr


class Decomposition:
    """Eigen-decomposition of a matrix.

    Parameters
    ----------
    rv : (N, N) np.ndarray
        The right eigenvectors of a matrix.
    xi : (N, ) np.ndarray
        The eigenvalues of a matrix
    rv_inv : (N, N) np.ndarray
        The left eigenvectors of a matrix.
    """
    __slots__ = ('rv', 'xi', 'rv_inv')

    def __init__(self, rv, xi, rv_inv):
        self.rv = rv
        self.xi = xi
        self.rv_inv = rv_inv

    @classmethod
    def decompose(cls, arr, h=None):
        """Computes the decomposition of a matrix and initializes a ``Decomposition``-instance.

        Parameters
        ----------
        arr : (N, N) array_like
            A complex or real matrix whose eigenvalues and eigenvectors will be computed.
        h : bool, optional
            Flag if matrix is hermitian. If ``None`` the matrix is checked. This determines the
            scipy method used for computing the eigenvalue problem.

        Returns
        -------
        decomposition : Decomposition
        """
        rv, xi, rv_inv = decompose(arr, h)
        return cls(rv, xi, rv_inv)

    def reconstrunct(self, xi=None, method='full'):
        """Computes a matrix from the eigen-decomposition.

        Parameters
        ----------
        xi : (N, ) array_like, optional
            Optional eigenvalues to compute the matrix. If ``None`` the eigenvalues of the
            decomposition are used. The default is ``None``.
        method : str, optional
        The mode for reconstructing the matrix. If mode is 'full' the original matrix
        is reconstructed, if mode is 'diag' only the diagonal elements are computed.
        The default is 'full'.

        Returns
        -------
        arr : (N, N) np.ndarray
            The reconstructed matrix.
        """
        xi = self.xi if xi is None else xi
        return reconstruct(self.rv, xi, self.rv_inv, method)

    def normalize(self):
        """Normalizes the eigenstates and creates a new ``Decomposition``-instance."""
        rv = self.rv / np.linalg.norm(self.rv)
        rv_inv = self.rv_inv / np.linalg.norm(self.rv_inv)
        return self.__class__(rv, self.xi, rv_inv)

    def transform_basis(self, transformation):
        """Transforms the eigen-basis into another basis.

        Parameters
        ----------
        transformation : (N, N) array_like
            The transformation-matrix to transform the basis.
        """
        rv = self.rv @ np.asarray(transformation)
        self.rv = rv
        self.rv_inv = np.linalg.inv(rv)

    def __str__(self):
        return f"{self.__class__.__name__}[{self.rv.shape}x{self.xi.shape}x{self.rv_inv.shape}]"


def get_ground_state(eigvals, eigvecs):
    idx = np.argmin(eigvals)
    gs = eigvecs[:, idx]
    gs_energy = eigvals[idx]
    return gs_energy, gs


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
