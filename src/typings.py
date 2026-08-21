import numpy as np
import sympy as sym


Unitary = np.ndarray
Unitaries = list[Unitary]
Circuit = list[Unitary]
Barrier = sym.Poly
Timings = dict[str, float]