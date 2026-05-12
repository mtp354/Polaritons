"""
Picard fixed-point iteration for the disorder self-energy.

Solves   X(k) = ∫ dq  K(q,k) * F(q, X(q))

where ``K`` is the disorder kernel and ``F`` is a propagator that has
already had any external parameters (e.g. ``eta``, ``E_ext``) bound by the
caller via ``functools.partial`` or a closure.
"""

from __future__ import annotations
import numpy as np


def picard_iteration(
	Q_init   : np.ndarray,
	q        : np.ndarray,
	K_matrix : np.ndarray,
	weights  : np.ndarray,
	F_func,
	tol      : float = 1e-5,
	max_iter : int   = 5000,
	w        : float = 1.0,
	verbose  : bool  = True,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Vectorised Picard iteration with non-uniform trapezoid quadrature.

	Parameters
	----------
	Q_init   : initial guess for X(k), 1-D complex array of length N
	q        : quadrature nodes (same grid as K_matrix rows), length N
	K_matrix : kernel K(q,k), shape (N, N)
	weights  : trapezoid quadrature weights for q, length N
	F_func   : callable ``F(q, X)`` returning the propagator. Any external
	           parameters such as ``eta`` and ``E_ext`` must already be bound.
	tol      : convergence tolerance (L2 norm of update)
	max_iter : maximum number of iterations
	w        : mixing parameter (w=1 → pure Picard, w<1 → under-relaxation)
	verbose  : print progress every 100 iterations

	Returns
	-------
	Q        : converged (or best) X(k) array
	delta    : array of L2 norms of successive updates, length max_iter
	"""
	delta = np.zeros(max_iter, dtype=complex)
	Q     = Q_init.copy()

	for iteration in range(max_iter):
		F_vec = F_func(q, Q) * weights                  # shape (N,)
		Q_new = (1.0 - w) * Q + w * (K_matrix @ F_vec)  # shape (N,)

		diff           = np.linalg.norm(Q_new - Q)
		delta[iteration] = diff

		if verbose and iteration % 100 == 0:
			print(f"  iter {iteration:5d}  |ΔQ| = {diff:.3e}")

		Q = Q_new
		if diff < tol:
			break

	return Q, delta
