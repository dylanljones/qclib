# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
from scipy import optimize


def gf_greater(xx, yx, xy, yy):
    return -0.25j * (xx + 1j*yx - 1j*xy + yy)


def gf_lesser(xx, xy, yx, yy):
    return +0.25j * (xx - 1j*xy + 1j*yx + yy)


def gf_fit(t, alpha_1, alpha_2, omega_1, omega_2):
    return 2 * (alpha_1 * np.cos(omega_1 * t) + alpha_2 * np.cos(omega_2 * t))


def fit_gf_measurement(t, data, p0=None, alpha_max=1, omega_max=100):
    bounds = (0, [alpha_max, alpha_max, omega_max, omega_max])
    popt, pcov = optimize.curve_fit(gf_fit, t, data, p0=p0, bounds=bounds)
    errs = np.sqrt(np.diag(pcov))
    return popt, errs


def gf_spectral(z, alpha_1, alpha_2, omega_1, omega_2):
    t1 = alpha_1 * (1 / (z + omega_1) + 1 / (z - omega_1))
    t2 = alpha_2 * (1 / (z + omega_2) + 1 / (z - omega_2))
    return t1 + t2


def fitted_gf_spectral(z, popt):
    return gf_spectral(z, *popt)


def print_popt(popt, errs, dec=2):
    strings = list(["Green's function fit:"])
    strings.append(f"  alpha_1 = {popt[0]:.{dec}f} ± {errs[0]:.{dec}}")
    strings.append(f"  alpha_2 = {popt[1]:.{dec}f} ± {errs[1]:.{dec}}")
    strings.append(f"  omega_1 = {popt[2]:.{dec}f} ± {errs[2]:.{dec}}")
    strings.append(f"  omega_2 = {popt[3]:.{dec}f} ± {errs[3]:.{dec}}")
    line = "-" * (max([len(x) for x in strings]) + 1)
    print(line)
    print("\n".join(strings))
    print(line)


def get_gf_fit_data(popt, tmax, n=100):
    t_fit = np.linspace(0, tmax, n)
    return t_fit, gf_fit(t_fit, *popt)


def get_gf_spectral_data(popt, zmax, eta=0.01, n=1000):
    z = np.linspace(-zmax, zmax, n) + 1j * eta
    return z, fitted_gf_spectral(z, popt)
