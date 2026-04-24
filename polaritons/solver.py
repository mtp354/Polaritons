"""
Picard fixed-point iteration for the self-energy Q(k, eta).

Solves:   Q(k) = ∫ dq  K(q,k) * F(q, Q(q), eta)

where K is the disorder kernel and F the Lippmann-Schwinger propagator.
"""

from __future__ import annotations
import numpy as np


def picard_iteration(
    Q_init   : np.ndarray,
    q        : np.ndarray,
    K_matrix : np.ndarray,
    weights  : np.ndarray,
    F_func,
    eta      : float = 1.0,
    tol      : float = 1e-5,
    max_iter : int   = 5000,
    w        : float = 1.0,
    verbose  : bool  = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised Picard iteration with non-uniform trapezoid quadrature.

    Parameters
    ----------
    Q_init   : initial guess for Q(k), 1-D complex array of length N
    q        : quadrature nodes (same grid as K_matrix rows), length N
    K_matrix : kernel K(q,k), shape (N, N)
    weights  : trapezoid quadrature weights for q, length N
    F_func   : callable F(q, Q, eta) — the Lippmann-Schwinger propagator
    eta      : disorder strength parameter
    tol      : convergence tolerance (L2 norm of update)
    max_iter : maximum number of iterations
    w        : mixing parameter (w=1 → pure Picard, w<1 → under-relaxation)
    verbose  : print progress every 100 iterations

    Returns
    -------
    Q        : converged (or best) Q(k) array
    delta    : array of L2 norms of successive updates, length max_iter
    """
    delta = np.zeros(max_iter, dtype=complex)
    Q     = Q_init.copy()

    for iteration in range(max_iter):
        F_vec = F_func(q, Q, eta=eta) * weights          # shape (N,)
        Q_new = (1.0 - w) * Q + w * (K_matrix @ F_vec)  # shape (N,)

        diff           = np.linalg.norm(Q_new - Q)
        delta[iteration] = diff

        if verbose and iteration % 100 == 0:
            print(f"  iter {iteration:5d}  |ΔQ| = {diff:.3e}")

        Q = Q_new
        if diff < tol:
            break

    return Q, delta
