# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

import math
import numpy as np
from scipy.optimize import OptimizeResult
from typing import List, Sequence, Callable


class RotoselectOptimizer:

    def __init__(self, generators: Sequence[str] = None, maxiter: int = 100, rtol: float = None,
                 phi: float = 0., callback: Callable = None):
        self.generators = generators or ["X", "Y", "Z", "I"]
        self.maxiter = maxiter
        self.rtol = rtol
        self.phi = phi
        self.callback = callback

        self.costs = list()
        self.sub_costs = list()
