# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Dict, Tuple


def plot_histogram(counts: Dict[str, int], color: Union[str, float] = None, alpha: float = 0.9,
                   width: float = 0.8, grid: str = None, bar_labels: bool = True, title: str = None,
                   normalize: bool = True, compress: bool = False, number_to_keep: int = 32,
                   figsize: Tuple[float, float] = (7, 5), ax: plt.Axes = None) -> plt.Axes:
    """Plot a histogram of measurement data.

    Parameters
    ----------
    counts : dict
        A dictionary of the measurement data to represent. The keys are binary strings
        and the values are the corresponding counts.
    color : str or float, optional
        The color of the histogram bars.
    width : float, optional
        The width of the histogram bars.
    alpha : float, optional
        The opacity of the histogram bars.
    grid : str, optional
        Wich grid to draw. Valid arguments are 'x', 'y' and 'both'.
    bar_labels : bool, optional
        If `True` the probability of each result is displayed above the hisogram bars.
    title : str, optional
        Optional title to use for the plot.
    normalize : bool, optional
        If `True` the y-axis will be normalized, displaying the probabilitie.
    compress : bool, optional
        If `True` measurement outcomes with a probability of `0` will not be plotted.
    number_to_keep : int, optional
        The number of terms to plot. If the number of possible outcomes exceedes `number_to_keep`
        the rest of the outcomes are combined in a single bar called 'Rest'.
    figsize : tuple of float, optional
        Figure size of the plot in inches.
    ax: plt.Axes, optional
        An optional `Axes` object to be used for the plot. If none is specified a new
        `Figure` will be created and used.

    Returns
    -------
    ax: plt.Axes
        The `Axes` object used for the plot. This is either the passed `Axes` object
        or a newly created one.
    """
    # Build histogram
    # ---------------
    num_qubits = len(list(counts.keys())[0])
    size = 2 ** num_qubits
    values = np.arange(size)

    # Build full count vector
    count_vector = np.zeros(size, dtype="int")
    for k, v in counts.items():
        count_vector[int(k, 2)] = v

    # Remove results with no counts
    if compress:
        mask = count_vector > 0
        count_vector = count_vector[mask]
        values = values[mask]

    # Create label strings
    if number_to_keep is not None and number_to_keep < len(count_vector):
        thresh = np.sort(count_vector)[::-1][number_to_keep]
        mask = count_vector > thresh
        rest = np.sum(count_vector[~mask])
        count_vector = np.append(count_vector[mask], rest)
        labels = [f"{x:0{num_qubits}b}" for x in values[mask]] + ["Rest"]
    else:
        labels = [f"{x:0{num_qubits}b}" for x in values]

    count_total = np.sum(count_vector)
    prob_vector = count_vector / count_total
    y_vector = prob_vector if normalize else count_vector

    bins = np.arange(count_vector.shape[0])

    # Plot histogram
    # --------------
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(*figsize)
    if title:
        ax.set_title(title)
    if grid is not None:
        ax.set_axisbelow(True)
        ax.grid(axis=grid)

    ax.bar(bins, y_vector, color=color, width=width, alpha=alpha)
    if bar_labels:
        for i, x in enumerate(bins):
            prob = prob_vector[i]
            if prob:
                y = y_vector[i]
                s = f"{prob:.2f}"
                ax.text(x, y, s, horizontalalignment='center', verticalalignment='bottom')

    ax.set_xlim(-0.5, len(bins) - 0.5)
    ax.set_xticks(bins)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Probability" if normalize else "Count")

    return ax
