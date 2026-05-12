"""
Sigma(k, E_ext) sweep + on-shell Q(k) reconstruction.

Workflow
--------
1. ``sweep_sigma`` runs Picard for every (eta, E_ext) pair on a fixed
   momentum grid. Within an eta block the previous E_ext's converged
   result warm-starts the next; across eta blocks the converged result
   from the same E_ext index of the previous eta seeds the new eta.
2. ``find_E_k_prime`` locates, for each (eta, k), the value E_k' of
   E_ext where ``Re[E_ext - hbar^2 k^2/(2M) - Sigma(k, E_ext)] = 0``,
   by linearly interpolating the bracketing pair on the E_ext grid.
3. ``assemble_Q`` returns ``Q(k, eta) = -Sigma(k, E_k')`` by linearly
   interpolating Sigma along E_ext at E_k'.

Sign convention: Sigma is defined as the negative of the converged
Picard fixed point, so the user-facing identity ``Q = -Sigma`` holds
exactly.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np

from .parameters import Params
from .solver import picard_iteration


def sweep_sigma(
	p           : Params,
	K_matrix    : np.ndarray,
	q           : np.ndarray,
	weights     : np.ndarray,
	E_ext_grid  : np.ndarray,
	eta_grid    : np.ndarray,
	F_factory   : Callable[[Params], Callable] | None = None,
	Q_init      : np.ndarray | None = None,
	tol         : float = 1e-6,
	max_iter    : int   = 5000,
	w           : float = 1.0,
	verbose     : bool  = False,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Sweep Picard over (eta, E_ext) on the momentum grid ``q``.

	Parameters
	----------
	p           : natural-unit Params
	K_matrix    : kernel mesh K(q, q), shape (N, N)
	q           : momentum grid, length N (must match K_matrix)
	weights     : quadrature weights for ``q``, length N
	E_ext_grid  : array of external energies (natural units), length n_E
	eta_grid    : array of disorder amplitudes, length n_eta
	F_factory   : callable returning the propagator F(q, Q, E_ext, eta).
	              Defaults to ``polaritons.kernel.make_propagator(p)``.
	Q_init      : initial Picard guess for the very first solve. Defaults
	              to ``(1+1j) * ones(N)``.
	tol, max_iter, w, verbose
	            : forwarded to ``picard_iteration``.

	Returns
	-------
	Sigma_arr : complex array of shape (n_eta, n_E, N) where
	            ``Sigma_arr[i, j, :] = -Q_picard(k; E_ext_j, eta_i)``
	            so that on-shell ``Q(k, eta) = -Sigma(k, E_k')``.
	iters     : int array of shape (n_eta, n_E) with the iteration count
	            (1 + last index written into ``delta``) for each solve.
	"""
	if F_factory is None:
		from .kernel import make_propagator
		F_factory = make_propagator

	F_full = F_factory(p)

	N      = len(q)
	n_eta  = len(eta_grid)
	n_E    = len(E_ext_grid)

	if K_matrix.shape != (N, N):
		raise ValueError(f"K_matrix must have shape ({N}, {N}); got {K_matrix.shape}")
	if weights.shape != (N,):
		raise ValueError(f"weights must have shape ({N},); got {weights.shape}")

	if Q_init is None:
		Q_init = (1.0 + 1.0j) * np.ones(N, dtype=complex)

	# Sigma_arr stores -Q_picard so that on-shell Q = -Sigma.
	Sigma_arr = np.zeros((n_eta, n_E, N), dtype=complex)
	iters     = np.zeros((n_eta, n_E), dtype=int)

	# Warm-start matrix: prev_eta_Q[j] holds the converged Q from the
	# previous eta at E_ext index j, used to seed eta_{i+1}.
	prev_eta_Q = np.tile(Q_init, (n_E, 1))   # shape (n_E, N)

	for ei, eta in enumerate(eta_grid):
		Q_seed = prev_eta_Q[0].copy()        # seed first E_ext from prev eta
		for ej, E_ext in enumerate(E_ext_grid):
			if eta == 0.0:
				# Picard with eta=0 collapses to Q = 0 in one step.
				Q_conv = np.zeros(N, dtype=complex)
				it_count = 1
			else:
				if ej == 0:
					seed = Q_seed                        # cross-eta seed
				else:
					seed = Sigma_arr[ei, ej - 1].copy()
					seed = -seed                         # back to Q = -Sigma
				F_bound = partial(F_full, E_ext=float(E_ext), eta=float(eta))
				Q_conv, delta = picard_iteration(
					seed, q, K_matrix, weights, F_bound,
					tol=tol, max_iter=max_iter, w=w, verbose=verbose,
				)
				nonzero = np.flatnonzero(delta != 0.0)
				it_count = int(nonzero[-1] + 1) if nonzero.size else 0

			Sigma_arr[ei, ej]  = -Q_conv
			prev_eta_Q[ej]     = Q_conv
			iters[ei, ej]      = it_count

	return Sigma_arr, iters


def find_E_k_prime(
	Sigma_arr  : np.ndarray,
	q          : np.ndarray,
	E_ext_grid : np.ndarray,
	p          : Params,
) -> np.ndarray:
	"""
	On-shell external energy E_k'(eta, k) where the real part of the
	denominator vanishes:

	    f(E_ext) = Re[ E_ext - hbar^2 k^2 / (2 M) - Sigma(k, E_ext) ] = 0

	The first sign change along the E_ext axis is bracketed and a linear
	interpolation gives E_k'. If no sign change occurs anywhere on the
	grid, the entry is set to NaN.

	Parameters
	----------
	Sigma_arr  : (n_eta, n_E, N) complex array from ``sweep_sigma``
	q          : momentum grid, length N
	E_ext_grid : external energy grid, length n_E
	p          : natural-unit Params (uses p.hbar, p.M)

	Returns
	-------
	E_k_prime  : float array of shape (n_eta, N), NaN where no root.
	"""
	n_eta, n_E, N = Sigma_arr.shape
	if len(E_ext_grid) != n_E:
		raise ValueError("E_ext_grid length must equal Sigma_arr.shape[1]")
	if len(q) != N:
		raise ValueError("q length must equal Sigma_arr.shape[2]")

	bare = (p.hbar**2 * np.asarray(q, dtype=float)**2) / (2.0 * p.M)   # (N,)
	E    = np.asarray(E_ext_grid, dtype=float)                         # (n_E,)

	out = np.full((n_eta, N), np.nan, dtype=float)

	for ei in range(n_eta):
		# f[j, k] = E[j] - bare[k] - Re[Sigma[ei, j, k]]
		f = E[:, None] - bare[None, :] - Sigma_arr[ei].real
		# Look for first sign change along axis 0 (E_ext).
		signs = np.sign(f)
		# Treat exact zero as positive so a sign change captures it on
		# the next step; use diff to find indices.
		s = np.where(signs == 0.0, 1.0, signs)
		change = (s[:-1] * s[1:]) < 0.0           # (n_E - 1, N)
		first  = np.argmax(change, axis=0)        # first True index per k
		has    = change.any(axis=0)               # any sign change per k

		# Linear interpolation between j and j+1 where f changes sign.
		j   = first
		k_idx = np.arange(N)
		f0 = f[j,     k_idx]
		f1 = f[j + 1, k_idx]
		E0 = E[j]
		E1 = E[j + 1]
		denom = (f1 - f0)
		# Guard against degenerate flat segments.
		safe = np.where(denom != 0.0, denom, 1.0)
		root = E0 - f0 * (E1 - E0) / safe

		out[ei] = np.where(has, root, np.nan)

	return out


def assemble_Q(
	Sigma_arr  : np.ndarray,
	E_k_prime  : np.ndarray,
	E_ext_grid : np.ndarray,
) -> np.ndarray:
	"""
	On-shell self-energy ``Q(k, eta) = -Sigma(k, E_k'(eta, k))``.

	Sigma is linearly interpolated along the E_ext axis at the per-(eta, k)
	root ``E_k_prime``. Where ``E_k_prime`` is NaN the corresponding entry
	of the output is also NaN.

	Parameters
	----------
	Sigma_arr  : (n_eta, n_E, N) complex array from ``sweep_sigma``
	E_k_prime  : (n_eta, N) float array from ``find_E_k_prime``
	E_ext_grid : monotone array of length n_E

	Returns
	-------
	Q_arr      : (n_eta, N) complex array
	"""
	n_eta, n_E, N = Sigma_arr.shape
	E = np.asarray(E_ext_grid, dtype=float)
	if E.shape != (n_E,):
		raise ValueError("E_ext_grid length must equal Sigma_arr.shape[1]")
	if E_k_prime.shape != (n_eta, N):
		raise ValueError("E_k_prime must have shape (n_eta, N)")
	if not np.all(np.diff(E) > 0):
		raise ValueError("E_ext_grid must be strictly increasing")

	Q_arr = np.full((n_eta, N), np.nan + 1j * np.nan, dtype=complex)

	for ei in range(n_eta):
		Ek = E_k_prime[ei]                       # (N,)
		valid = np.isfinite(Ek)
		if not valid.any():
			continue
		# Bracketing index for each valid k.
		idx = np.searchsorted(E, Ek[valid]) - 1
		idx = np.clip(idx, 0, n_E - 2)
		E0  = E[idx]
		E1  = E[idx + 1]
		t   = (Ek[valid] - E0) / (E1 - E0)
		k_valid = np.flatnonzero(valid)
		S0 = Sigma_arr[ei, idx,     k_valid]
		S1 = Sigma_arr[ei, idx + 1, k_valid]
		Sigma_on = (1.0 - t) * S0 + t * S1
		Q_arr[ei, k_valid] = -Sigma_on

	return Q_arr
