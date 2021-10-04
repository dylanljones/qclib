# coding: utf-8
#
# This code is part of qclib.
# 
# Copyright (c) 2021, Dylan Jones

import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import XGate, CXGate, CCXGate
from cmpy import TimeEvolutionOperator
from lattpy import Lattice
from . import libary, run


# =========================================================================
# Model and classical evolution
# =========================================================================

def gaussian_state(x, x0, sigma=1.0):
    """statevector representing a gaussian probability distribution

    Parameters
    ----------
    x : (N, ) array_like
        The x-values for evaluating the function.
    x0 : float
        The center of the Gaussian wavepacket.
    sigma : float, optional
        The width of the probability distribution of the wavefunction.

    Returns
    -------
    psi : (N, ) np.ndarray
    """
    psi = np.exp(-np.power(x - x0, 2.) / (4 * sigma**2))
    norm = np.sqrt(np.sqrt(2 * np.pi) * sigma)
    return psi / norm


def initialize_state(size, index=None, mode="delta", *args, **kwargs):
    """Initialize the statevector of a state

    Parameters
    ----------
    size : int
        The size of the statevector/Hilbert space.
    index : int or float, optional
        The position or site index where the state is localized.
    mode : str, optional
        The mode for initializing the state. valid modes are: 'delta' and 'gauss'
    *args
        Positional arguments for state function.
    **kwargs
        Keyword arguments for the state function.

    Returns
    -------
    psi : (N, ) np.ndarray
    """
    index = int(size // 2) if index is None else index
    if mode == "delta":
        psi = np.zeros(size)
        psi[int(index)] = 1.
    elif mode == "gauss":
        x = np.arange(size)
        psi = gaussian_state(x, index, *args, **kwargs)
    else:
        raise ValueError(f"Mode '{mode}' not supported. Valid modes: 'delta', 'gauss'")
    return psi


def hamiltonian(latt: Lattice, eps: (float, np.ndarray) = 0.,
                t: (float, np.ndarray) = 1.0,
                dense: bool = True) -> (csr_matrix, np.ndarray):
    """Computes the Hamiltonian-matrix of a tight-binding model.

    Parameters
    ----------
    latt : Lattice
        The lattice the tight-binding model is defined on.
    eps : array_like, optional
        The on-site energies of the model.
    t : array_like, optional
        The hopping energies of the model.
    dense : bool, optional
        If ``True`` the hamiltonian matrix is returned as a ``np.ndarray``

    Returns
    -------
    ham : csr_matrix or np.ndarray
        The Hamiltonian-matrix as a sparse or dense matrix.
    """
    dmap = latt.data.map()
    data = np.zeros(dmap.size)
    data[dmap.onsite()] = eps
    data[dmap.hopping()] = t
    ham = csr_matrix((data, dmap.indices))
    if dense:
        ham = ham.toarray()
    return ham


def evolve_state(ham, state0, times):
    """Evolve a state under the given hamiltonian.

    Parameters
    ----------
    ham : (N, N) array_like
    state0 : (N, ) array_like
    times : float or (M, ) array_like

    Returns
    -------
    states : (M, N) np.nd_array or (N, ) np.ndarray
    """
    scalar = not hasattr(times, "__len__")
    if scalar:
        times = [times]
    states = np.zeros((len(times), len(state0)), dtype=np.complex64)
    top = TimeEvolutionOperator(ham)
    for i, t in enumerate(times):
        states[i] = top(t) @ state0
    return states[0] if scalar else states


def compute_time_evolution_cl(latt, steps, w=0., dt=0.2, x0=None, mode="delta", **kwargs):
    times = steps * dt
    state0 = initialize_state(latt.num_sites, index=x0, mode=mode, **kwargs)
    eps = w * np.random.random(latt.num_sites)
    ham = hamiltonian(latt, eps=eps, dense=True)
    states = evolve_state(ham, state0, times)
    return np.square(np.abs(states))


def state_index(latt, pos=None):
    if pos is None:
        shape = np.array(latt.shape)
        pos = shape / 2

    index = list(latt.estimate_index(pos)) + [0]
    res = np.where(np.all(latt.data.indices == index, axis=1))[0]
    ind = res[0]
    return ind


# =========================================================================
# Time evolution circuit
# =========================================================================


def init_binary_state(qc, number):
    binstr = f"{number:0{qc.num_qubits}b}"
    indices = [i for i, c in enumerate(binstr[::-1]) if c == "1"]
    if indices:
        qc.x(indices)
    return qc


def random_gate(gates, qubits, targets=None):
    qubits = list(qubits[:])

    # Pick random gate
    gate = random.choice(gates)()

    # Pick random target qubit
    target_qubits = qubits if targets is None else qubits[targets]
    target = random.choice(target_qubits)

    # Pick random control qubits
    num_control = gate.num_ctrl_qubits if hasattr(gate, "num_ctrl_qubits") else 0
    if num_control:
        qubits.remove(target)
        control = random.sample(qubits, k=num_control)
        qargs = list(control) + [target]
    else:
        qargs = [target]

    return gate, qargs


def random_permutation_circuit(num_qubits, depth, targets=None):
    gates = [XGate, CXGate] if num_qubits == 2 else [XGate, CCXGate]
    qc = QuantumCircuit(num_qubits)
    qubits = np.arange(qc.num_qubits)
    for _ in range(depth):
        gate, qargs = random_gate(gates, qubits, targets)
        qc.append(gate, qargs)
    return qc


def kinetic_operator(qc, qubits, dt, reverse=False, mode="noancilla"):
    index = len(qubits) - 1 if reverse else 0
    qubit = qubits[index]
    # Even part of the kinetic hamiltonian
    qc.rx(2 * dt, qubit)
    # Odd part of the kinetic hamiltonian
    libary.increment(qc, reverse, mode)  # Translate lattice by +1
    qc.rx(2 * dt, qubit)
    libary.decrement(qc, reverse, mode)  # Translate lattice by -1


def potential_operator(qc, qubits, w, dt, depth_exp=2):
    num_qubits = len(qubits)
    p = random_permutation_circuit(num_qubits, int(qc.num_qubits ** depth_exp))
    # perform random permutation
    qc.append(p.to_instruction(), qc.qubits[:])
    # diagonal phase rotation around z-axis
    for i in range(qc.num_qubits):
        q = i
        phi = w * dt / (2 ** (num_qubits - q))
        qc.rz(phi, qc.qubits[i])
    # perform inverse random permutation for next step
    qc.append(p.inverse().to_instruction(), qc.qubits[:])


def trotter_step_circuit(num_qubits, w, dt, depth_exp=2, reverse=False, mode="noancilla"):
    if isinstance(num_qubits, int):
        num_qubits = [num_qubits]

    qc = QuantumCircuit(sum(num_qubits))
    start = 0
    for q in num_qubits:
        stop = start + q
        kinetic_operator(qc, qc.qubits[start:stop], dt, reverse=reverse, mode=mode)
        start = stop
    if w:
        potential_operator(qc, qc.qubits[:], w, dt, depth_exp=depth_exp)
    return qc


def compute_time_evolution_qm(latt, steps, w=0., dt=0.2, x0=None, shots=512):
    if x0 is None:
        x0 = int(latt.num_sites // 2)

    num_qubits = int(np.log2(latt.num_sites))
    u = trotter_step_circuit(num_qubits, w, dt)  # noqa
    probs = np.zeros((len(steps), latt.num_sites))

    print("Measuring...", end="", flush=True)
    num_steps = len(steps)
    for i, nt in enumerate(steps):
        p = (i+1) / num_steps
        print(f"\rMeasuring step {i+1}/{num_steps} ({100*p:.1f}%)", end="", flush=True)
        qc = QuantumCircuit(num_qubits)
        init_binary_state(qc, x0)
        for _ in range(int(nt)):
            qc.append(u.to_instruction(), qc.qubits[:])
        qc.measure_all()
        res = run(qc, shots)
        probs[i] = res.probabilities
    print()
    return probs


# =========================================================================


def random_permutation(qc: QuantumCircuit, depth: int):
    gates = [XGate, CXGate] if qc.num_qubits == 2 else [XGate, CCXGate]
    num_ctrl = int(qc.num_qubits / 2)
    for _ in range(depth):
        gate = random.choice(gates)()
        if hasattr(gate, "num_ctrl_qubits"):
            ctrl_qubits = qc.qubits[:num_ctrl]
            target_qubits = qc.qubits[num_ctrl:]
            qargs = list(random.sample(ctrl_qubits, k=gate.num_ctrl_qubits))
            qargs.append(random.choice(target_qubits))
        else:
            qargs = [random.choice(qc.qubits)]
        qc.append(gate, qargs)


def get_positions(probs, full=False):
    """Computes the positions of the particle during the time evolution.

    The position of the particle is defined by the maximum of the probability distribution.

    Parameters
    ----------
    probs : (..., N, M) np.ndarray
        The array containing the probability distributions for each time step.
    full : bool, optional
        Wheter to use the full probability distribution in both directions.
        If ``False`` only one direction from the starting point is used.
        The default is ``False``.

    Returns
    -------
    positions : (..., N) np.ndarray
        The positions of the particle for each time step.
    """
    num_sites = probs.shape[-1]
    x0 = int(num_sites // 2)
    if full:
        return np.argmax(probs, axis=-1) - x0
    return np.argmax(probs[..., x0:], axis=-1)


def get_max_distance(positions):
    """Computes the maximal distance reached by the particle"""
    distances = np.abs(positions - positions[0])
    return np.max(distances)


def get_final_distance(probs, mode="mean"):
    positions = get_positions(probs, full=False)
    if mode == "mean":
        dist = np.mean(positions, axis=0)[-1]
    elif mode == "maxmean":
        dist = np.max(np.mean(positions, axis=0))
    elif mode == "max":
        dist = np.max(positions)
    else:
        raise ValueError("Mode " + mode + " not supported!")
    return dist
