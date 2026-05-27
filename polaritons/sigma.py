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
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from .parameters import Params
from .solver import picard_iteration


# ---------------------------------------------------------------------------
# Interpolation helpers (linear / cubic) for on-shell Sigma reconstruction.
# ---------------------------------------------------------------------------

def _interp_window(j: int, n_E: int, width: int = 4) -> tuple[int, int]:
	"""
	Return ``(lo, hi)`` such that ``[lo, hi]`` is a ``width``-point window
	on a length-``n_E`` grid that contains the bracket ``[j, j+1]`` and
	stays inside ``[0, n_E-1]``.  ``hi`` is exclusive in slicing terms.
	"""
	if n_E < 2:
		raise ValueError("Need at least 2 E_ext points")
	w  = min(width, n_E)
	lo = j - (w - 2) // 2
	lo = max(0, min(lo, n_E - w))
	return lo, lo + w


def sweep_sigma(
	p           : Params,
	K_matrix    : np.ndarray,
	q           : np.ndarray,
	weights     : np.ndarray,
	E_ext_grid  : np.ndarray,
	eta_grid    : np.ndarray,
	F_factory   : Callable[[Params], Callable] | None = None,
	Q_init      : np.ndarray | None = None,
	Sigma_seed  : np.ndarray | None = None,
	solve_mask  : np.ndarray | None = None,
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
	Q_init      : initial Picard guess. Accepts either a 1-D array of
	              length N (global initial seed for the very first solve,
	              backwards-compatible behaviour) or a 3-D array of shape
	              ``(n_eta, n_E, N)`` giving a per-cell warm start.
	Sigma_seed  : optional ``(n_eta, n_E, N)`` array of previously
	              computed ``Sigma`` values. Cells with ``solve_mask`` =
	              False are not re-solved and instead take their output
	              from ``Sigma_seed``. When ``solve_mask`` is False at a
	              cell, ``Sigma_seed`` must be provided for that cell.
	solve_mask  : optional ``(n_eta, n_E)`` bool array. ``True`` cells
	              run Picard; ``False`` cells copy ``Sigma_seed``. The
	              eta=0 short-circuit (Sigma=0) still applies regardless.
	tol, max_iter, w, verbose
	            : forwarded to ``picard_iteration``.

	Returns
	-------
	Sigma_arr : complex array of shape (n_eta, n_E, N) where
	            ``Sigma_arr[i, j, :] = -Q_picard(k; E_ext_j, eta_i)``
	            so that on-shell ``Q(k, eta) = -Sigma(k, E_k')``.
	iters     : int array of shape (n_eta, n_E) with the iteration count
	            (1 + last index written into ``delta``) for each solve.
	            Masked-out cells report 0 iterations.
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

	# Normalise Q_init: support either a single (N,) seed (legacy) or a
	# per-cell (n_eta, n_E, N) warm-start tensor.
	if Q_init is None:
		Q_init_arr = (1.0 + 1.0j) * np.ones(N, dtype=complex)
		per_cell_init = False
	else:
		Q_init_arr = np.asarray(Q_init, dtype=complex)
		if Q_init_arr.shape == (N,):
			per_cell_init = False
		elif Q_init_arr.shape == (n_eta, n_E, N):
			per_cell_init = True
		else:
			raise ValueError(
				f"Q_init must have shape ({N},) or ({n_eta}, {n_E}, {N}); "
				f"got {Q_init_arr.shape}"
			)

	if Sigma_seed is not None:
		Sigma_seed = np.asarray(Sigma_seed, dtype=complex)
		if Sigma_seed.shape != (n_eta, n_E, N):
			raise ValueError(
				f"Sigma_seed must have shape ({n_eta}, {n_E}, {N}); "
				f"got {Sigma_seed.shape}"
			)

	if solve_mask is not None:
		solve_mask = np.asarray(solve_mask, dtype=bool)
		if solve_mask.shape != (n_eta, n_E):
			raise ValueError(
				f"solve_mask must have shape ({n_eta}, {n_E}); "
				f"got {solve_mask.shape}"
			)
		if Sigma_seed is None and not solve_mask.all():
			raise ValueError(
				"solve_mask has False entries but Sigma_seed was not "
				"provided; cannot fill skipped cells."
			)

	# Sigma_arr stores -Q_picard so that on-shell Q = -Sigma.
	Sigma_arr = np.zeros((n_eta, n_E, N), dtype=complex)
	iters     = np.zeros((n_eta, n_E), dtype=int)

	# Warm-start matrix: prev_eta_Q[j] holds the converged Q from the
	# previous eta at E_ext index j, used to seed eta_{i+1}.
	if per_cell_init:
		prev_eta_Q = Q_init_arr[0].copy()      # shape (n_E, N)
	else:
		prev_eta_Q = np.tile(Q_init_arr, (n_E, 1))   # shape (n_E, N)

	for ei, eta in enumerate(eta_grid):
		Q_seed_cross = prev_eta_Q[0].copy()   # seed first E_ext from prev eta
		for ej, E_ext in enumerate(E_ext_grid):
			cell_active = True if solve_mask is None else bool(solve_mask[ei, ej])

			if not cell_active:
				# Copy the seeded Sigma straight through; no Picard work.
				Sigma_arr[ei, ej] = Sigma_seed[ei, ej]
				prev_eta_Q[ej]    = -Sigma_seed[ei, ej]
				iters[ei, ej]     = 0
				continue

			if eta == 0.0:
				# Picard with eta=0 collapses to Q = 0 in one step.
				Q_conv = np.zeros(N, dtype=complex)
				it_count = 1
			else:
				# Choose warm-start seed for this Picard solve.
				if per_cell_init or Sigma_seed is not None:
					# Prefer per-cell information when supplied.
					if Sigma_seed is not None:
						seed = -Sigma_seed[ei, ej].copy()
					else:
						seed = Q_init_arr[ei, ej].copy()
				elif ej == 0:
					seed = Q_seed_cross                 # cross-eta seed
				else:
					seed = -Sigma_arr[ei, ej - 1].copy()  # within-eta seed
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
	interp     : str = "linear",
) -> np.ndarray:
	"""
	On-shell external energy E_k'(eta, k) where the real part of the
	denominator vanishes:

	    f(E_ext) = E_ext - hbar^2 k^2 / (2 M) - Re[Sigma(k, E_ext)] = 0

	The first sign change of ``f`` along the E_ext grid is bracketed and
	then refined with ``scipy.optimize.brentq``.  Between the bracketing
	grid samples ``Re[Sigma]`` is reconstructed via either:

	- ``interp="linear"`` (default): straight segment between the two
	  bracketing grid samples;
	- ``interp="cubic"``: a monotone PCHIP fit over a 4-point window
	  spanning the bracket plus one neighbour on each side (clipped to
	  the grid).  This recovers ``E_k'`` more accurately when ``Re[Sigma]``
	  is small relative to the grid spacing, which is the regime where
	  linear segmentation produces an artificial bump in ``Q(k)``.

	If no sign change occurs anywhere on the grid the entry is set to NaN.

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
	interp     : ``"linear"`` (default) or ``"cubic"``.

	Returns
	-------
	E_k_prime  : float array of shape (n_eta, N), NaN where no root.
	"""
	if interp not in ("linear", "cubic"):
		raise ValueError(f"interp must be 'linear' or 'cubic'; got {interp!r}")

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

			if interp == "cubic" and n_E >= 4:
				lo, hi = _interp_window(j, n_E, width=4)
				pchip = PchipInterpolator(
					E[lo:hi],
					Sig_re_all[lo:hi, k_idx],
					extrapolate=False,
				)

				def _f(E_val, _pchip=pchip, _b=b):
					return float(E_val - _b - _pchip(E_val))
			else:
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
	interp     : str = "linear",
) -> np.ndarray:
	"""
	On-shell self-energy ``Q(k, eta) = -Sigma(k, E_k'(eta, k))``.

	Sigma is interpolated along the E_ext axis at the per-(eta, k) root
	``E_k_prime`` using either:

	- ``interp="linear"`` (default): linear interpolation between the two
	  bracketing E_ext samples;
	- ``interp="cubic"``: monotone PCHIP on a 4-point window spanning the
	  bracket and one neighbour on each side (clipped to grid bounds).
	  Real and imaginary parts of ``Sigma`` are interpolated separately.

	Where ``E_k_prime`` is NaN the corresponding entry of the output is
	also NaN.

	Parameters
	----------
	Sigma_arr  : (n_eta, n_E, N) complex array from ``sweep_sigma``
	E_k_prime  : (n_eta, N) float array from ``find_E_k_prime``
	E_ext_grid : monotone array of length n_E
	interp     : ``"linear"`` (default) or ``"cubic"``.

	Returns
	-------
	Q_arr      : (n_eta, N) complex array
	"""
	if interp not in ("linear", "cubic"):
		raise ValueError(f"interp must be 'linear' or 'cubic'; got {interp!r}")

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
		k_valid = np.flatnonzero(valid)

		if interp == "linear" or n_E < 4:
			E0  = E[idx]
			E1  = E[idx + 1]
			t   = (Ek[valid] - E0) / (E1 - E0)
			S0 = Sigma_arr[ei, idx,     k_valid]
			S1 = Sigma_arr[ei, idx + 1, k_valid]
			Sigma_on = (1.0 - t) * S0 + t * S1
		else:
			Sigma_on = np.empty(k_valid.size, dtype=complex)
			Sig_slice = Sigma_arr[ei]               # (n_E, N)
			for m, k_idx in enumerate(k_valid):
				j = int(idx[m])
				lo, hi = _interp_window(j, n_E, width=4)
				col = Sig_slice[lo:hi, k_idx]
				re = PchipInterpolator(E[lo:hi], col.real, extrapolate=False)
				im = PchipInterpolator(E[lo:hi], col.imag, extrapolate=False)
				Sigma_on[m] = float(re(Ek[k_idx])) + 1j * float(im(Ek[k_idx]))

		Q_arr[ei, k_valid] = -Sigma_on

	return Q_arr


# ---------------------------------------------------------------------------
# Spectral-function exciton dispersion
# ---------------------------------------------------------------------------

def assemble_E_ex_spectral(
	Sigma_arr  : np.ndarray,
	q          : np.ndarray,
	E_ext_grid : np.ndarray,
	p          : Params,
	*,
	eta_grid   : np.ndarray | None = None,
	refine     : bool = True,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Real and imaginary exciton dispersion from the spectral function.

	For each ``(eta, k)`` the retarded spectral function is

	    A(k, E) = -(1/π) Im Σ(k, E)
	              / [(E - ε_k - Re Σ(k, E))² + (Im Σ(k, E))²]

	with ε_k = ħ² k² / (2 M) the bare exciton dispersion (band-bottom
	relative).

	- ``Re[E_ex(k)]`` is the ``E`` that maximises ``A(k, ·)``.  The
	  discrete argmax over ``E_ext_grid`` is bracketed and (if
	  ``refine=True`` and the bracket is interior) refined with a
	  bounded Brent search over monotone PCHIP interpolants of ``Re Σ``
	  and ``Im Σ``.
	- ``Im[E_ex(k)]`` is taken as ``Z(k) · Im Σ(k, E*)`` where
	  ``Z(k) = 1 / (1 - ∂ Re Σ/∂E |_{E*})`` is the quasiparticle
	  residue.  This is the (signed) HWHM of the Lorentzian obtained by
	  linearising the inverse Green's function about the peak; with the
	  retarded convention ``Im Σ ≤ 0`` it gives ``Im[E_ex] ≤ 0``
	  (decaying excitations).

	``eta = 0`` rows (Σ ≡ 0) short-circuit to ``E_ex = ε_k``, ``Z = 1``.

	Parameters
	----------
	Sigma_arr   : (n_eta, n_E, N) complex Σ from ``sweep_sigma``
	q           : momentum grid, length N
	E_ext_grid  : external-energy grid, length n_E (strictly increasing)
	p           : natural-unit Params (uses ``p.hbar``, ``p.M``)
	eta_grid    : optional (n_eta,) array; used only to detect and
	              short-circuit ``eta = 0`` rows.
	refine      : if True, Brent-refine the peak inside the bracketing
	              triple ``[E_{j*-1}, E_{j*}, E_{j*+1}]`` when ``j*`` is
	              an interior grid index.

	Returns
	-------
	E_ex_arr : (n_eta, N) complex; real part = peak energy, imaginary
	           part = Z · Im Σ at the peak (negative for decay).
	Z_arr    : (n_eta, N) float; quasiparticle residue at the peak.
	"""
	from scipy.optimize import minimize_scalar

	Sigma_arr = np.asarray(Sigma_arr, dtype=complex)
	q         = np.asarray(q, dtype=float)
	E         = np.asarray(E_ext_grid, dtype=float)

	if Sigma_arr.ndim != 3:
		raise ValueError("Sigma_arr must be 3-D (n_eta, n_E, N)")
	n_eta, n_E, N = Sigma_arr.shape
	if E.shape != (n_E,):
		raise ValueError("E_ext_grid length must equal Sigma_arr.shape[1]")
	if q.shape != (N,):
		raise ValueError("q length must equal Sigma_arr.shape[2]")
	if not np.all(np.diff(E) > 0):
		raise ValueError("E_ext_grid must be strictly increasing")

	if eta_grid is not None:
		eta_grid = np.asarray(eta_grid, dtype=float)
		if eta_grid.shape != (n_eta,):
			raise ValueError("eta_grid length must equal Sigma_arr.shape[0]")

	bare = (p.hbar**2 * q**2) / (2.0 * p.M)            # (N,)

	E_ex_arr = np.full((n_eta, N), np.nan + 1j * np.nan, dtype=complex)
	Z_arr    = np.full((n_eta, N), np.nan, dtype=float)

	for ei in range(n_eta):
		# eta = 0 short-circuit: bare dispersion, no broadening.
		if (eta_grid is not None and float(eta_grid[ei]) == 0.0) \
				or not np.any(Sigma_arr[ei]):
			E_ex_arr[ei, :] = bare.astype(complex)
			Z_arr[ei, :]    = 1.0
			continue

		Sig_re_all = Sigma_arr[ei].real                # (n_E, N)
		Sig_im_all = Sigma_arr[ei].imag                # (n_E, N)

		for k_idx in range(N):
			col_re = Sig_re_all[:, k_idx]
			col_im = Sig_im_all[:, k_idx]
			if not (np.all(np.isfinite(col_re)) and np.all(np.isfinite(col_im))):
				continue

			b = float(bare[k_idx])

			# Discrete A on the grid; argmax gives the bracket.
			denom_grid = (E - b - col_re) ** 2 + col_im ** 2
			with np.errstate(divide="ignore", invalid="ignore"):
				A_grid = np.where(denom_grid > 0.0, -col_im / denom_grid, -np.inf)
			j_star = int(np.argmax(A_grid))

			re_spline = PchipInterpolator(E, col_re, extrapolate=False)
			im_spline = PchipInterpolator(E, col_im, extrapolate=False)

			def neg_A(Ev, _re=re_spline, _im=im_spline, _b=b):
				rS = float(_re(Ev))
				iS = float(_im(Ev))
				d  = (Ev - _b - rS) ** 2 + iS ** 2
				if d == 0.0:
					return np.inf
				return -(-iS) / d           # minimise -A; A = -ImΣ / d

			if refine and 0 < j_star < n_E - 1:
				lo, hi = float(E[j_star - 1]), float(E[j_star + 1])
				try:
					res = minimize_scalar(
						neg_A, bounds=(lo, hi), method="bounded",
						options={"xatol": 1e-12},
					)
					E_star = float(res.x) if (lo <= float(res.x) <= hi) else float(E[j_star])
				except (ValueError, RuntimeError):
					E_star = float(E[j_star])
			else:
				E_star = float(E[j_star])

			im_at_star = float(im_spline(E_star))
			d_re_dE    = float(re_spline.derivative()(E_star))
			one_minus  = 1.0 - d_re_dE
			Z_val      = (1.0 / one_minus) if one_minus != 0.0 else np.nan

			E_ex_arr[ei, k_idx] = complex(E_star, Z_val * im_at_star)
			Z_arr[ei, k_idx]    = Z_val

	return E_ex_arr, Z_arr


# ---------------------------------------------------------------------------
# Reuse helpers: resume / reseed / refine a previously-saved Sigma surface
# ---------------------------------------------------------------------------

def resume_unconverged(
	p             : Params,
	K_matrix      : np.ndarray,
	q             : np.ndarray,
	weights       : np.ndarray,
	E_ext_grid    : np.ndarray,
	eta_grid      : np.ndarray,
	Sigma_prev    : np.ndarray,
	iters_prev    : np.ndarray,
	prev_max_iter : int,
	*,
	w             : float,
	max_iter      : int,
	tol           : float = 1e-6,
	F_factory     : Callable[[Params], Callable] | None = None,
	verbose       : bool = False,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Re-run Picard only at cells flagged non-converged in a previous sweep.

	A cell ``(ei, ej)`` is considered non-converged when
	``iters_prev[ei, ej] == prev_max_iter`` (Picard ran out of iterations).
	The eta=0 row is always excluded since its solution is exact by
	construction.

	The resulting Sigma surface equals ``Sigma_prev`` everywhere except
	at the re-solved cells, which are warm-started by ``-Sigma_prev``
	and integrated with the new damping ``w`` and ``max_iter``.
	"""
	n_eta = len(eta_grid)
	n_E   = len(E_ext_grid)
	if Sigma_prev.shape != (n_eta, n_E, len(q)):
		raise ValueError(
			f"Sigma_prev must have shape ({n_eta}, {n_E}, {len(q)}); "
			f"got {Sigma_prev.shape}"
		)
	if iters_prev.shape != (n_eta, n_E):
		raise ValueError(
			f"iters_prev must have shape ({n_eta}, {n_E}); "
			f"got {iters_prev.shape}"
		)

	mask = (iters_prev == int(prev_max_iter))
	for ei, eta in enumerate(eta_grid):
		if float(eta) == 0.0:
			mask[ei, :] = False

	n_to_solve = int(mask.sum())
	if verbose:
		print(f"resume_unconverged: re-solving {n_to_solve} of {mask.size} cells "
		      f"(w={w}, max_iter={max_iter}, tol={tol:g})")

	if n_to_solve == 0:
		# Nothing to do: return inputs unchanged.
		return Sigma_prev.copy(), iters_prev.copy()

	Sigma_new, iters_new = sweep_sigma(
		p, K_matrix, q, weights, E_ext_grid, eta_grid,
		F_factory=F_factory,
		Q_init=-Sigma_prev,
		Sigma_seed=Sigma_prev,
		solve_mask=mask,
		tol=tol, max_iter=max_iter, w=w, verbose=verbose,
	)

	# Carry over the original iteration counts at the cells we did not
	# touch so the caller can tell apart "previously converged" (small
	# count) from "skipped in this resume pass" (0).
	keep = ~mask
	iters_new[keep] = iters_prev[keep]

	if verbose:
		still_stuck = int((iters_new[mask] == max_iter).sum())
		print(f"resume_unconverged: {still_stuck} cells still hit max_iter "
		      f"after re-solve")

	return Sigma_new, iters_new


def rerun_with_seed(
	p             : Params,
	K_matrix      : np.ndarray,
	q             : np.ndarray,
	weights       : np.ndarray,
	E_ext_grid    : np.ndarray,
	eta_grid      : np.ndarray,
	Sigma_prev    : np.ndarray,
	*,
	w             : float,
	max_iter      : int,
	tol           : float = 1e-6,
	F_factory     : Callable[[Params], Callable] | None = None,
	verbose       : bool = False,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Re-run the full Picard sweep, warm-starting every cell from
	``-Sigma_prev``. Use this to re-solve with weaker (or stronger)
	damping ``w`` from a previously converged surface.
	"""
	n_eta = len(eta_grid)
	n_E   = len(E_ext_grid)
	if Sigma_prev.shape != (n_eta, n_E, len(q)):
		raise ValueError(
			f"Sigma_prev must have shape ({n_eta}, {n_E}, {len(q)}); "
			f"got {Sigma_prev.shape}"
		)
	return sweep_sigma(
		p, K_matrix, q, weights, E_ext_grid, eta_grid,
		F_factory=F_factory,
		Q_init=-Sigma_prev,
		tol=tol, max_iter=max_iter, w=w, verbose=verbose,
	)


def refine_E_ext_grid(
	E_ext_grid_old : np.ndarray,
	p              : Params,
	t_local        : float,
	*,
	n_add          : int,
	band_pad_omega : float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Densify an existing E_ext grid by inserting uniform points across
	the bare-parabola band ``[-pad*Omega, bare_max + pad*Omega]``.

	Parameters
	----------
	E_ext_grid_old : strictly-increasing existing grid, length n_E_old
	p              : natural-unit Params
	t_local        : kernel truncation in natural units (defines bare_max)
	n_add          : number of uniform points to attempt to insert in the
	                 band (duplicates within rel-tol of an existing point
	                 are discarded)
	band_pad_omega : extra Omega-widths above and below the parabola band

	Returns
	-------
	E_ext_grid_new : merged strictly-increasing grid, length n_E_new
	new_mask       : bool array of length n_E_new, True at indices that
	                 were freshly inserted (not present in the old grid)
	old_to_new_idx : int array of length n_E_old mapping each old E_ext
	                 value to its position in ``E_ext_grid_new``
	"""
	if n_add <= 0:
		raise ValueError(f"n_add must be positive; got {n_add}")

	E_old = np.asarray(E_ext_grid_old, dtype=float)
	if E_old.ndim != 1 or np.any(np.diff(E_old) <= 0.0):
		raise ValueError("E_ext_grid_old must be strictly increasing 1-D array.")

	bare_max = float(p.hbar**2 * float(t_local)**2 / (2.0 * p.M))
	pad      = float(band_pad_omega) * float(p.Omega)
	E_lo     = -pad
	E_hi     = bare_max + pad

	# Generate `n_add` strictly-interior uniform candidate points so they
	# never coincide with the band endpoints (the existing grid already
	# samples those when concentrated near the parabola).
	candidates = np.linspace(E_lo, E_hi, int(n_add) + 2)[1:-1]

	# Drop candidates that match an existing grid point within a relative
	# tolerance (avoid spurious near-duplicates that np.unique would keep).
	span = float(E_old[-1] - E_old[0]) if E_old.size > 1 else 1.0
	tol  = 1e-9 * max(span, 1.0)
	keep = []
	for c in candidates:
		j = int(np.searchsorted(E_old, c))
		nearest = []
		if j > 0:
			nearest.append(E_old[j - 1])
		if j < E_old.size:
			nearest.append(E_old[j])
		if all(abs(c - n) > tol for n in nearest):
			keep.append(c)
	fresh = np.array(keep, dtype=float)

	merged = np.unique(np.concatenate([E_old, fresh]))
	# Identify which merged indices correspond to fresh inserts.
	new_mask       = np.ones(merged.size, dtype=bool)
	old_to_new_idx = np.searchsorted(merged, E_old)
	new_mask[old_to_new_idx] = False

	return merged, new_mask, old_to_new_idx


def _interp_sigma_along_E(
	Sigma_prev   : np.ndarray,
	E_old        : np.ndarray,
	E_new_points : np.ndarray,
) -> np.ndarray:
	"""
	Linear interpolation of ``Sigma_prev`` along its E_ext axis at
	``E_new_points``. Returns an array of shape
	``(n_eta, len(E_new_points), N)``.
	"""
	n_eta, n_E_old, N = Sigma_prev.shape
	if E_old.shape != (n_E_old,):
		raise ValueError("E_old length must match Sigma_prev.shape[1].")

	# Bracketing indices for each new E value.
	idx = np.searchsorted(E_old, E_new_points) - 1
	idx = np.clip(idx, 0, n_E_old - 2)
	E0  = E_old[idx]                              # (n_new,)
	E1  = E_old[idx + 1]                          # (n_new,)
	t   = ((E_new_points - E0) / (E1 - E0))[None, :, None]  # (1, n_new, 1)

	S0 = Sigma_prev[:, idx,     :]                # (n_eta, n_new, N)
	S1 = Sigma_prev[:, idx + 1, :]                # (n_eta, n_new, N)
	return (1.0 - t) * S0 + t * S1


def refine_sigma(
	p              : Params,
	K_matrix       : np.ndarray,
	q              : np.ndarray,
	weights        : np.ndarray,
	E_ext_grid_old : np.ndarray,
	eta_grid       : np.ndarray,
	Sigma_prev     : np.ndarray,
	t_local        : float,
	*,
	n_add          : int,
	band_pad_omega : float = 0.5,
	w              : float,
	max_iter       : int,
	tol            : float = 1e-6,
	F_factory      : Callable[[Params], Callable] | None = None,
	verbose        : bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Add ``n_add`` E_ext samples in the bare-parabola band and solve Picard
	only at the freshly inserted columns. Existing columns are copied
	through unchanged.

	Returns
	-------
	Sigma_new      : (n_eta, n_E_new, N) complex
	iters_new      : (n_eta, n_E_new) int (0 at copied columns)
	E_ext_grid_new : (n_E_new,) float, strictly increasing
	"""
	E_old = np.asarray(E_ext_grid_old, dtype=float)
	n_eta = len(eta_grid)
	N     = len(q)
	if Sigma_prev.shape != (n_eta, E_old.size, N):
		raise ValueError(
			f"Sigma_prev must have shape ({n_eta}, {E_old.size}, {N}); "
			f"got {Sigma_prev.shape}"
		)

	E_new, new_mask, old_to_new_idx = refine_E_ext_grid(
		E_old, p, t_local,
		n_add=n_add, band_pad_omega=band_pad_omega,
	)
	n_E_new = E_new.size

	# Assemble the full seed surface on the refined grid.
	Sigma_seed = np.empty((n_eta, n_E_new, N), dtype=complex)
	Sigma_seed[:, old_to_new_idx, :] = Sigma_prev
	if new_mask.any():
		Sigma_seed[:, new_mask, :] = _interp_sigma_along_E(
			Sigma_prev, E_old, E_new[new_mask],
		)

	solve_mask = np.broadcast_to(new_mask, (n_eta, n_E_new)).copy()

	if verbose:
		print(f"refine_sigma: inserted {int(new_mask.sum())} new E_ext points "
		      f"(total {n_E_new}); solving Picard on {int(solve_mask.sum())} cells")

	Sigma_new, iters_new = sweep_sigma(
		p, K_matrix, q, weights, E_new, eta_grid,
		F_factory=F_factory,
		Q_init=-Sigma_seed,
		Sigma_seed=Sigma_seed,
		solve_mask=solve_mask,
		tol=tol, max_iter=max_iter, w=w, verbose=verbose,
	)
	return Sigma_new, iters_new, E_new
