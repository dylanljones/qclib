# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import qiskit
import numpy as np
import matplotlib.pyplot as plt
from collections import Mapping
from typing import Iterator, Dict, Optional, Union


def plot_histogram(bins, hist, labels=None, padding=0.2, color=None, alpha=0.9, scale=False,
                   max_line=True, lc="r", lw=1, rot=0, ax=None, show=False):
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_xlim(-0.5, len(bins) - 0.5)
    ax.set_xticks(bins)
    if not scale:
        ax.set_ylim(0, 1.1)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    if labels is not None:
        ax.set_xticklabels(labels, rotation=rot)
    ax.bar(bins, hist, width=1 - padding, color=color, alpha=alpha)
    if max_line:
        ymax = np.max(hist)
        ax.axhline(ymax, color=lc, lw=lw)
    ax.set_xlabel("State")
    ax.set_ylabel("p")
    if show:
        plt.show()
    return ax


def build_vector(counts: Dict[str, int], num_qubits: Optional[int] = None):
    num_qubits = num_qubits or len(list(counts.keys())[0])
    count_vector = np.zeros((2 ** num_qubits), dtype="int")
    for k, v in counts.items():
        count_vector[int(k, 2)] = v
    return count_vector


class Result(Mapping):

    def __init__(self, data: Dict[str, int]):
        super().__init__()
        self.raw = data
        self.num_qubits = len(list(data.keys())[0])
        self.counts = build_vector(data, self.num_qubits)
        self.samples = np.sum(self.counts)
        self.labels = [f"{x:0{self.num_qubits}b}" for x in range(len(self.counts))]

    @property
    def len(self):
        return len(self.counts)

    @property
    def probabilities(self) -> np.ndarray:
        return self.counts / self.samples

    @property
    def argmax(self) -> int:
        return int(np.argmax(self.counts))

    @property
    def binary(self) -> str:
        return self.labels[self.argmax]

    @property
    def value(self) -> int:
        return int(self.binary, 2)

    def __getitem__(self, k: int) -> int:
        return self.counts[k]

    def __len__(self) -> int:
        return len(self.counts)

    def __iter__(self) -> Iterator[int]:
        return iter(self.counts)

    def histogram(self, normalize=True):
        hist = self.probabilities if normalize else self.counts
        return np.arange(self.__len__()), hist

    def plot_histogram(self, show=True, padding=0.2, color=None, alpha=0.9,
                       scale=False, labels=None, max_line=True, lc="r", lw=1,
                       binary=True, rot=0, fig=None, ax=None):
        bins, hist = self.histogram()
        if not labels:
            label_items = self.labels if binary else range(len(bins))
            labels = [r"$|$" + str(x) + r"$\rangle$" for x in label_items]
        labels = labels or [r"$|$" + str(x) + r"$\rangle$" for x in self.labels]
        return plot_histogram(bins, hist, labels, padding, color, alpha, scale,
                              max_line, lc, lw, rot, ax, show)

    def pformat(self, decimals: int = 2) -> str:
        w = decimals + 2
        labelstr = "[" + "  ".join(f"{x:^{w}}" for x in self.labels) + "]"
        valuestr = "[" + "  ".join(f"{x:.{decimals}f}" for x in self.probabilities) + "]"
        return labelstr + "\n" + valuestr

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.raw})"


def run(qc, shots=1024, backend=None):
    if backend is None:
        backend = qiskit.Aer.get_backend("qasm_simulator")
    elif isinstance(backend, str):
        backend = qiskit.Aer.get_backend(backend)
    # result = qiskit.execute(qc, backend, shots=shots).result()

    # assemble the circuit
    experiments = qiskit.transpile(qc, backend)
    qobj = qiskit.assemble(experiments, shots=shots)

    # run the circuit on the backend
    result = backend.run(qobj).result()
    return Result(result.get_counts(qc))
