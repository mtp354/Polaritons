"""
Uniform momentum grid and trapezoid quadrature weights.

The Picard pipeline now operates on a single uniform mesh
``q = linspace(0, K_max, N)``; the previous segmented grid + spline
pre-compute has been removed.
"""

from __future__ import annotations
import numpy as np


def uniform_grid_and_weights(K_max: float, N: int) -> tuple[np.ndarray, np.ndarray]:
	"""
	Uniform momentum grid on [0, K_max] with composite-trapezoid weights.

	Parameters
	----------
	K_max : upper momentum (natural units)
	N     : number of grid points (>= 2)

	Returns
	-------
	q : 1-D array of length N, ``np.linspace(0, K_max, N)``
	w : 1-D array of length N, trapezoid weights
	    (interior dx, endpoints dx/2, sum = K_max)
	"""
	if N < 2:
		raise ValueError("N must be >= 2")
	if K_max <= 0.0:
		raise ValueError("K_max must be positive")

	q  = np.linspace(0.0, float(K_max), int(N))
	dx = K_max / (N - 1)
	w  = np.full(N, dx, dtype=float)
	w[0]  = dx / 2.0
	w[-1] = dx / 2.0
	return q, w
