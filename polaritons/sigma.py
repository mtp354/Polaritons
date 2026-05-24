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
from scipy.optimize import brentq

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
				solve_label = f"eta={float(eta):.3g} E_ext={float(E_ext):+.4g}"
				Q_conv, delta = picard_iteration(
					seed, q, K_matrix, weights, F_bound,
					tol=tol, max_iter=max_iter, w=w, verbose=verbose,
					label=solve_label,
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

	    f(E_ext) = E_ext - hbar^2 k^2 / (2 M) - Re[Sigma(k, E_ext)] = 0

	The first sign change of ``f`` along the E_ext grid is bracketed and
	then refined with ``scipy.optimize.brentq`` on a linear interpolation
	of ``Sigma.real`` between the bracketing grid points. If no sign
	change occurs anywhere on the grid, the entry is set to NaN.

	The bare-band reference is ``hbar^2 k^2 / (2 M)``, i.e. the clean
	exciton dispersion measured from the band bottom (no semiconductor
	offset). This matches the band-bottom-relative convention used
	elsewhere in the library.

	Parameters
	----------
	Sigma_arr  : (n_eta, n_E, N) complex array from ``sweep_sigma``
	q          : momentum grid, length N
	E_ext_grid : external energy grid, length n_E (strictly increasing)
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

	E = np.asarray(E_ext_grid, dtype=float)
	if not np.all(np.diff(E) > 0):
		raise ValueError("E_ext_grid must be strictly increasing")

	bare = (p.hbar**2 * np.asarray(q, dtype=float)**2) / (2.0 * p.M)   # (N,)

	out = np.full((n_eta, N), np.nan, dtype=float)

	for ei in range(n_eta):
		Sig_re_all = Sigma_arr[ei].real   # (n_E, N)
		# f[j, k] = E[j] - bare[k] - Re[Sigma[ei, j, k]]
		f = E[:, None] - bare[None, :] - Sig_re_all
		# Treat exact zero as positive so a sign change captures it later.
		s      = np.where(np.sign(f) == 0.0, 1.0, np.sign(f))
		change = (s[:-1] * s[1:]) < 0.0   # (n_E - 1, N)
		first  = np.argmax(change, axis=0)
		has    = change.any(axis=0)

		for k_idx in range(N):
			if not has[k_idx]:
				continue
			j = int(first[k_idx])
			E0, E1 = float(E[j]), float(E[j + 1])
			s0 = float(Sig_re_all[j,     k_idx])
			s1 = float(Sig_re_all[j + 1, k_idx])
			b  = float(bare[k_idx])

			def _f(E_val, _s0=s0, _s1=s1, _E0=E0, _E1=E1, _b=b):
				# Linear interpolation of Re[Sigma] between (E0, s0) and (E1, s1).
				t = (E_val - _E0) / (_E1 - _E0)
				sig = _s0 + t * (_s1 - _s0)
				return E_val - _b - sig

			f0, f1 = _f(E0), _f(E1)
			if f0 == 0.0:
				out[ei, k_idx] = E0
				continue
			if f1 == 0.0:
				out[ei, k_idx] = E1
				continue
			if f0 * f1 > 0.0:
				# Numerical edge: bracket lost after substitution; skip.
				continue
			try:
				root = brentq(_f, E0, E1, xtol=1e-14, rtol=1e-12, maxiter=100)
			except (ValueError, RuntimeError):
				continue
			out[ei, k_idx] = float(root)

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
