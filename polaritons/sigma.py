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


def refine_sigma_per_k(
	p             : Params,
	K_matrix      : np.ndarray,
	q             : np.ndarray,
	weights       : np.ndarray,
	eta           : float,
	E_seed        : np.ndarray,
	half_width    : float,
	n_E_pk        : int,
	F_factory     : Callable[[Params], Callable] | None = None,
	Q_init        : np.ndarray | None = None,
	tol           : float = 1e-6,
	max_iter      : int   = 5000,
	w             : float = 1.0,
	verbose       : bool  = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""
	Pass-2 per-k refinement.

	For each momentum index ``k_idx`` we build a tight, uniform external-energy
	window of width ``2 * half_width`` centred on ``E_seed[k_idx]`` (the pass-1
	estimate of E_k') and run ``n_E_pk`` Picard solves on that window. The
	root of ``E_ext - bare(k) - Re[Sigma(k, E_ext)] = 0`` is then refined with
	``scipy.optimize.brentq`` inside this narrow window.

	Each (k_idx, j) is one Picard solve at a scalar E_ext. Picard returns the
	full Sigma vector for all k, but only the ``k_idx``-th column is retained
	(the entry whose external energy actually sits inside the per-k window).
	Cost scales as ``N * n_E_pk`` Picard solves per eta value.

	If ``E_seed[k_idx]`` is non-finite, the bare exciton energy
	``hbar^2 k^2 / (2 M)`` is used as the centre instead so that no input
	NaN propagates into a missing root.

	Parameters
	----------
	p              : natural-unit Params
	K_matrix       : kernel mesh K(q, q), shape (N, N)
	q, weights     : Picard quadrature grid and weights, length N each
	eta            : disorder amplitude (scalar)
	E_seed         : per-k centre energies from pass 1, shape (N,)
	half_width     : per-k window half-width in natural energy units
	n_E_pk         : number of E_ext samples in each per-k window
	F_factory      : as in ``sweep_sigma``
	Q_init         : initial Picard seed (default: ``(1+1j) * ones(N)``)
	tol, max_iter, w, verbose
	               : forwarded to ``picard_iteration``.

	Returns
	-------
	E_k_prime  : (N,) float    -- refined on-shell energies (NaN if no root)
	Q_out      : (N,) complex  -- Q(k) = -Sigma(k, E_k_prime[k])
	Sigma_col  : (n_E_pk, N)   -- Sigma(k, E_per_k[j, k]) at the k-th column
	E_per_k    : (n_E_pk, N)   -- per-k external-energy grid
	iters      : (n_E_pk, N)   -- per-solve Picard iteration counts
	"""
	if F_factory is None:
		from .kernel import make_propagator
		F_factory = make_propagator

	F_full = F_factory(p)

	N = len(q)
	if K_matrix.shape != (N, N):
		raise ValueError(f"K_matrix must have shape ({N}, {N}); got {K_matrix.shape}")
	if weights.shape != (N,):
		raise ValueError(f"weights must have shape ({N},); got {weights.shape}")
	if np.asarray(E_seed).shape != (N,):
		raise ValueError(f"E_seed must have shape ({N},); got {np.asarray(E_seed).shape}")
	if n_E_pk < 2:
		raise ValueError("n_E_pk must be at least 2")
	if half_width <= 0.0:
		raise ValueError("half_width must be positive")

	q_arr = np.asarray(q, dtype=float)
	bare  = (p.hbar**2 * q_arr**2) / (2.0 * p.M)            # (N,)

	# Fall back to bare-band energy where pass-1 produced no estimate.
	E_centre = np.asarray(E_seed, dtype=float).copy()
	bad      = ~np.isfinite(E_centre)
	if bad.any():
		E_centre[bad] = bare[bad]

	offsets = np.linspace(-half_width, +half_width, n_E_pk)
	E_per_k = E_centre[None, :] + offsets[:, None]          # (n_E_pk, N)

	Sigma_col = np.zeros((n_E_pk, N), dtype=complex)
	iters     = np.zeros((n_E_pk, N), dtype=int)

	if Q_init is None:
		Q_init = (1.0 + 1.0j) * np.ones(N, dtype=complex)

	# Warm-start chain: Q from the previous k's centre solve seeds the next
	# k's first solve; within a k, the previous j's converged Q seeds j+1.
	Q_carry = Q_init.copy()

	for k_idx in range(N):
		Q_seed = Q_carry.copy()
		for j in range(n_E_pk):
			E_val = float(E_per_k[j, k_idx])
			if eta == 0.0:
				Q_conv   = np.zeros(N, dtype=complex)
				it_count = 1
			else:
				F_bound = partial(F_full, E_ext=E_val, eta=float(eta))
				Q_conv, delta = picard_iteration(
					Q_seed, q, K_matrix, weights, F_bound,
					tol=tol, max_iter=max_iter, w=w, verbose=False,
				)
				nonzero  = np.flatnonzero(delta != 0.0)
				it_count = int(nonzero[-1] + 1) if nonzero.size else 0
				Q_seed   = Q_conv.copy()
			Sigma_col[j, k_idx] = -Q_conv[k_idx]
			iters[j, k_idx]     = it_count
			if j == n_E_pk // 2:
				Q_carry = Q_conv.copy()
		if verbose and (k_idx % 200 == 0):
			print(f"  refine_per_k: k_idx={k_idx}/{N}, mean iters={iters[:, k_idx].mean():.1f}")

	E_k_prime = np.full(N, np.nan, dtype=float)
	Q_out     = np.full(N, np.nan + 1j * np.nan, dtype=complex)

	for k_idx in range(N):
		E_local  = E_per_k[:, k_idx]
		Sig_full = Sigma_col[:, k_idx]
		Sig_re   = Sig_full.real
		b        = float(bare[k_idx])
		f        = E_local - b - Sig_re
		s        = np.where(np.sign(f) == 0.0, 1.0, np.sign(f))
		change   = (s[:-1] * s[1:]) < 0.0
		if not change.any():
			continue
		j  = int(np.argmax(change))
		E0 = float(E_local[j]);     E1 = float(E_local[j + 1])
		s0 = float(Sig_re[j]);      s1 = float(Sig_re[j + 1])

		def _f(E_val, _s0=s0, _s1=s1, _E0=E0, _E1=E1, _b=b):
			t   = (E_val - _E0) / (_E1 - _E0)
			sig = _s0 + t * (_s1 - _s0)
			return E_val - _b - sig

		f0, f1 = _f(E0), _f(E1)
		if f0 == 0.0:
			root = E0
		elif f1 == 0.0:
			root = E1
		elif f0 * f1 > 0.0:
			continue
		else:
			try:
				root = brentq(_f, E0, E1, xtol=1e-14, rtol=1e-12, maxiter=100)
			except (ValueError, RuntimeError):
				continue
		t        = (root - E0) / (E1 - E0)
		Sigma_on = (1.0 - t) * Sig_full[j] + t * Sig_full[j + 1]
		E_k_prime[k_idx] = float(root)
		Q_out[k_idx]     = -Sigma_on

	return E_k_prime, Q_out, Sigma_col, E_per_k, iters


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
