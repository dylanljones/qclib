# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import os.path
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import qiskit
from qclib import PauliOperators, abs2, fill_diagonal, get_ground_state, compute_ground_state, libary
from qclib.circuit import QuantumCircuit, init_backend
from qclib.vqe import VariationalSolver, VQEFitter
from qclib import jordan_wigner as jw

VQE_FILE = "vqe_state.npz"


def save_opt(sol):
    params = sol.x
    np.savez(VQE_FILE, x=params)


def load_opt():
    data = np.load(VQE_FILE)
    return data["x"]


def prepare_ground_state(num_sites, eps, hop, plot=False, load=False):
    ham = jw.tight_binding_chain(num_sites, eps, hop)
    e_gs, gs = compute_ground_state(ham)
    probs = abs2(gs)

    vqe = VQEFitter(probs, layers=4, gates="rx ry rz", entangle="linear", shots=10 * 1024)
    if os.path.isfile(VQE_FILE) and load:
        vqe.load_popt(VQE_FILE)
    else:
        vqe.optimize(tol=1e-10, method="COBYLA", catol=0.00002, maxiter=10_000)
        vqe.save_popt(VQE_FILE)

    if plot:
        probs_vqe = vqe.get_optimized_probabilities()
        fig, ax = plt.subplots()
        x = np.arange(probs.shape[0])
        ax.bar(x, probs, alpha=0.5)
        ax.bar(x, probs_vqe, alpha=0.5)
        plt.show()

    return vqe.get_optimized_circuit()


def jordan_wigner_string(qc, qubit1, qubit2, reverse=False):
    if reverse:
        qc.cz(qubit2-1, qubit2)
        for i in reversed(range(qubit1+1, qubit2-1)):
            qc.cx(i, i+1)
    else:
        for i in range(qubit1+1, qubit2-1):
            qc.cx(i, i+1)
        qc.cz(qubit2-1, qubit2)


def hopping(qc: qiskit.QuantumCircuit, qubit1, qubit2, hop, dt):
    phi = hop * dt

    qubits = [qubit1, qubit2]

    jordan_wigner_string(qc, qubit1, qubit2)

    qc.h(qubits)
    qc.cx(qubit1, qubit2)
    qc.rz(phi, qubit2)
    qc.cx(qubit1, qubit2)
    qc.h(qubits)

    qc.rx(-np.pi/2, qubits)
    qc.cx(qubit1, qubit2)
    qc.rz(phi, qubit2)
    qc.cx(qubit1, qubit2)
    qc.rx(+np.pi/2, qubits)

    jordan_wigner_string(qc, qubit1, qubit2, reverse=True)


def trotter_step_circuit(num_qubits, eps, hop, dt, xy=False):
    qc = qiskit.QuantumCircuit(num_qubits)

    # on-site terms
    if isinstance(eps, (float, int)):
        eps = eps * np.ones(num_qubits)
    phis = -eps * dt
    for i, phi in enumerate(phis):
        if phi != 0.:
            qc.rz(phi, qc.qubits[i])

    # hopping terms
    if hop != 0.:
        for i in range(qc.num_qubits - 1):
            if xy:
                qc.append(libary.gates.xy_gate(hop * dt / 2), [i, i + 1])
            else:
                hopping(qc, i, i+1, hop, dt)

    return qc


def measure_greens(num_sites, eps, hop, num_steps, dt, shots=1028, index=0, xy=False):
    init_circ = prepare_ground_state(num_sites, eps, hop, plot=False, load=False)
    step_circ = trotter_step_circuit(num_sites, eps, hop, dt, xy=xy)

    # Run interferometer and measure gf
    backend = init_backend(gpu=True)
    inter = libary.TrotterInterferometer(init_circ, step_circ, shots=shots, backend=backend)
    return inter.measure_gf_imag(num_steps, index=1)


def main():
    num_sites = 5
    eps, hop = 0.0, 1.0
    tmax, dt = 20, 0.2
    shots = 10 * 1028

    num_steps = int(tmax / dt) + 1
    times = np.arange(0, tmax + 1e-5, dt)

    gft_imag = measure_greens(num_sites, eps, hop, num_steps, dt, shots, index=0, xy=False)
    plt.plot(times, gft_imag, label=f"Sites: {num_sites} (Rz)")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
