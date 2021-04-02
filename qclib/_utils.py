# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
from typing import Optional, Union, Iterable


_BITORDER = +1
_ARRORDER = -1


si = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
pauli = si, sx, sy, sz

EIGVALS = np.array([+1, -1])
EV_X = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
EV_Y = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
EV_Z = np.array([[1, 0], [0, 1]])

# Initial states
ZERO = np.array([1, 0])
ONE = np.array([0, 1])
PLUS = np.array([1, 1]) / np.sqrt(2)
MINUS = np.array([1, -1]) / np.sqrt(2)
IPLUS = np.array([1, 1j]) / np.sqrt(2)
IMINUS = np.array([1, -1j]) / np.sqrt(2)

# State dictionary for easy initialization
STATES = {"0": ZERO, "1": ONE, "+": PLUS, "-": MINUS, "i+": IPLUS, "i-": IMINUS}

# Projections onto |0> and |1>
P0 = np.dot(ZERO[:, np.newaxis], ZERO[np.newaxis, :])
P1 = np.dot(ONE[:, np.newaxis], ONE[np.newaxis, :])
PROJECTIONS = [P0, P1]


def kron(*args):
    """Computes the Kronecker product of two or more arrays.

    Parameters
    ----------
    args : list of array_like or array_like
        Arrays used for the kronecker product.

    Returns
    -------
    out : np.ndarray
    """
    if len(args) == 1:
        args = args[0]
    x = 1
    for arg in args:
        x = np.kron(x, arg)
    return x


def is_unitary(a, rtol=1e-5, atol=1e-8):
    """Checks if an array is a unitary matrix.

    Parameters
    ----------
    a : array_like
        The array to check.
    rtol : float
        The relative tolerance parameter.
    atol : float
        The absolute tolerance parameter.

    Returns
    -------
    allclose : bool
    """
    a = np.asarray(a)
    return np.allclose(a.dot(np.conj(a).T), np.eye(a.shape[0]), rtol=rtol, atol=atol)


def get_projector(v):
    r"""Constructs a projection-matrix from a given vector.

    .. math::
        P = | v \rangle \langle v |

    Parameters
    ----------
    v : (N) np.ndarray
        The state-vector used for constructing the projector.

    Returns
    -------
    p : (N, N) np.ndarray
    """
    if len(v.shape) == 1:
        v = v[:, np.newaxis]
    return np.dot(v, np.conj(v).T)


def expectation(o, psi):
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


def density_matrix(psi):
    r"""Computes the density matrix of a given state

    .. math::
        \rho = | \Psi \rangle \langle \Psi |

    Parameters
    ----------
    psi: (N) np.ndarray

    Returns
    -------
    rho : (N, N) np.ndarray
    """
    return np.dot(psi[:, np.newaxis], psi[np.newaxis, :])


# =========================================================================


def chain(items, cycle=False):
    """Create chain between items

    Parameters
    ----------
    items : array_like
        items to join to chain
    cycle : bool, optional
        cycle to the start of the chain if True, default: False

    Returns
    -------
    chain : list
        chain of items

    Example
    -------
    >>> print(chain(["x", "y", "z"]))
    [['x', 'y'], ['y', 'z']]

    >>> print(chain(["x", "y", "z"], True))
    [['x', 'y'], ['y', 'z'], ['z', 'x']]
    """
    result = list()
    for i in range(len(items)-1):
        result.append([items[i], items[i+1]])
    if cycle:
        result.append([items[-1], items[0]])
    return result


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
    return f"{num:0{fill}b}"[::_BITORDER]


def binarr(num: int, width: Optional[int] = None,
           dtype: Optional[Union[int, str]] = None) -> np.ndarray:
    """ Returns the bits of an integer as a binary array.

    Parameters
    ----------
    num : int or Spinstate
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
    return np.fromiter(f"{num:0{fill}b}"[::_ARRORDER], dtype=dtype or np.int16)


def binidx(num, width: Optional[int] = None) -> Iterable[int]:
    """ Returns the indices of bits with the value `1`.

    Parameters
    ----------
    num : int or Spinstate
        The number representing the binary state.
    width : int, optional
        Minimum number of digits used. The default is the global value `BITS`.

    Returns
    -------
    binidx : list
    """
    fill = width or 0
    return list(sorted(i for i, char in enumerate(f"{num:0{fill}b}"[::_ARRORDER]) if char == "1"))


class Binary(int):

    def __init__(self, *args, width=None, **kwargs):  # noqa
        super().__init__()

    def __new__(cls, *args, width: Optional[int] = None, **kwargs):
        self = int.__new__(cls, *args)  # noqa
        self.width = width
        return self

    @property
    def num(self) -> int:
        return int(self)

    @property
    def bin(self) -> bin:
        return bin(self)

    def binstr(self, width: Optional[int] = None) -> str:
        """ Binary representation of the state """
        return binstr(self, width or self.width)

    def binarr(self, width: Optional[int] = None,
               dtype: Optional[Union[int, str]] = None) -> np.ndarray:
        """ Returns the bits of an integer as a binary array. """
        return binarr(self, width or self.width, dtype)

    def binidx(self, width: Optional[int] = None) -> Iterable[int]:
        """ Indices of bits with value `1`. """
        return binidx(self, width or self.width)

    def count(self, value: int = 1) -> int:
        return bin(self).count(str(value))

    def get(self, bit: int) -> int:
        return int(self & (1 << bit))

    def flip(self, bit: int) -> 'Binary':
        return self.__class__(self ^ (1 << bit), width=self.width)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.binstr()})"

    def __str__(self) -> str:
        return self.binstr()

    # -------------- Math operators -------------

    def __add__(self, other: Union[int, 'Binary']) -> 'Binary':
        return self.__class__(super().__add__(other), width=self.width)

    def __radd__(self, other: Union[int, 'Binary']) -> 'Binary':
        return self.__class__(super().__radd__(other), width=self.width)

    def __sub__(self, other: Union[int, 'Binary']) -> 'Binary':
        return self.__class__(super().__sub__(other), width=self.width)

    def __rsub__(self, other: Union[int, 'Binary']) -> 'Binary':
        return self.__class__(super().__rsub__(other), width=self.width)

    def __mul__(self, other: Union[int, 'Binary']) -> 'Binary':
        return self.__class__(super().__mul__(other), width=self.width)

    def __rmul__(self, other: Union[int, 'Binary']) -> 'Binary':
        return self.__class__(super().__rmul__(other), width=self.width)

    # -------------- Binary operators -------------

    def __invert__(self) -> 'Binary':
        return self.__class__(super().__invert__(), width=self.width)

    def __and__(self, other: Union[int, 'Binary']) -> 'Binary':
        return self.__class__(super().__and__(other), width=self.width)

    def __or__(self, other: Union[int, 'Binary']) -> 'Binary':
        return self.__class__(super().__or__(other), width=self.width)

    def __xor__(self, other: Union[int, 'Binary']) -> 'Binary':
        return self.__class__(super().__xor__(other), width=self.width)

    def __lshift__(self, other: Union[int, 'Binary']) -> 'Binary':
        return self.__class__(super().__lshift__(other), width=self.width)

    def __rshift__(self, other: Union[int, 'Binary']) -> 'Binary':
        return self.__class__(super().__rshift__(other), width=self.width)
