"""
Dispersion relations for excitons, photons, and lower polaritons.

Functions here operate in natural units:
	- momentum is measured in units of 1/a
	- energy is measured in units of E_bind

Plotting and reporting code is responsible for converting these values back
to SI-derived units such as cm^-1, eV, meV, or micro-eV um^2.
"""

from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline
from .parameters import Params


# ---------------------------------------------------------------------------
# Spline interpolants built from saved Q results
# ---------------------------------------------------------------------------

class DispersionModel:
	"""
	Encapsulates E_ex, E_ph, E_LP given a loaded set of Q(k, eta) results.

	Q(k) is interpolated only along the momentum axis.  The disorder parameter
	eta selects one of the solved eta slices; it is not interpolated.

	Parameters
	----------
	p          : Params in natural units
	q_grid_nat : momentum grid in natural units (from the Picard solve)
	eta_grid   : array of disorder parameter values
	Q_results  : complex array, shape (len(eta_grid), len(q_grid_nat))
	"""

	def __init__(
		self,
		p          : Params,
		q_grid_nat : np.ndarray,
		eta_grid   : np.ndarray,
		Q_results  : np.ndarray,
	):
		if not p.in_natural_units:
			raise ValueError("DispersionModel expects Params in natural units.")

		self.p        = p
		self.eta_grid = np.asarray(eta_grid, dtype=float)
		self.q_grid_nat = np.asarray(q_grid_nat, dtype=float)
		self.q_grid = self.q_grid_nat

		Q_results = np.asarray(Q_results, dtype=complex)
		expected_shape = (len(self.eta_grid), len(self.q_grid_nat))
		if Q_results.shape != expected_shape:
			raise ValueError(
				f"Q_results must have shape {expected_shape}, got {Q_results.shape}."
			)
		if self.q_grid_nat.ndim != 1 or np.any(np.diff(self.q_grid_nat) <= 0.0):
			raise ValueError("q_grid_nat must be a strictly increasing 1-D array.")

		# Build one k-only spline for each solved eta value.  NaN samples
		# (k-points where the on-shell root could not be located) are
		# dropped before fitting; extrapolation covers the gaps.
		self._Q_splines = []
		for i in range(len(self.eta_grid)):
			row    = Q_results[i]
			finite = np.isfinite(row.real) & np.isfinite(row.imag)
			if finite.sum() < 2:
				raise ValueError(
					f"Q_results[{i}] (eta={self.eta_grid[i]}) has fewer than two finite samples; "
					"cannot build a spline."
				)
			self._Q_splines.append(
				CubicSpline(self.q_grid_nat[finite], row[finite], extrapolate=True)
			)

	def _eta_bracket(self, eta: float) -> tuple[int, int, float]:
		"""
		Return ``(i_lo, i_hi, w)`` such that
		``Q(k, eta) ≈ (1-w) * Q_spline[i_lo](k) + w * Q_spline[i_hi](k)``.

		``eta`` is clamped to ``[eta_grid[0], eta_grid[-1]]``; exact grid
		points yield ``w == 0`` (or ``w == 1``) so the on-grid result is
		exactly the saved spline value.
		"""
		eta_grid = self.eta_grid
		eta_clamped = float(np.clip(eta, eta_grid[0], eta_grid[-1]))
		i_hi = int(np.searchsorted(eta_grid, eta_clamped, side="left"))
		if i_hi <= 0:
			return 0, 0, 0.0
		if i_hi >= len(eta_grid):
			return len(eta_grid) - 1, len(eta_grid) - 1, 0.0
		if eta_clamped == eta_grid[i_hi]:
			return i_hi, i_hi, 0.0
		i_lo = i_hi - 1
		span = eta_grid[i_hi] - eta_grid[i_lo]
		w    = 0.0 if span == 0.0 else (eta_clamped - eta_grid[i_lo]) / span
		return i_lo, i_hi, float(w)

	# ------------------------------------------------------------------
	# Self-energy
	# ------------------------------------------------------------------

	def Q(self, k_nat: float | np.ndarray, eta: float) -> complex | np.ndarray:
		"""
		Self-energy Q(k, eta) interpolated over k (cubic spline) and over
		eta (linear between solved eta slices).

		``eta`` outside the solved grid is clamped to the grid endpoints.
		Real and imaginary parts are interpolated together via complex
		linear blending of the two bracketing spline evaluations.

		Parameters
		----------
		k_nat : momentum in natural units
		eta   : disorder parameter value (any real number; interpolated)

		Returns
		-------
		Q     : complex energy shift in natural energy units
		"""
		k_nat = np.asarray(k_nat, dtype=float)
		if eta == 0.0:
			return np.zeros_like(k_nat, dtype=complex)
		i_lo, i_hi, w = self._eta_bracket(float(eta))
		Q_lo = self._Q_splines[i_lo](k_nat)
		if i_lo == i_hi or w == 0.0:
			return Q_lo
		Q_hi = self._Q_splines[i_hi](k_nat)
		return (1.0 - w) * Q_lo + w * Q_hi

	# ------------------------------------------------------------------
	# Dispersion relations
	# ------------------------------------------------------------------

	def E_ex(self, k_nat: float | np.ndarray, eta: float) -> np.ndarray:
		"""
		Complex exciton dispersion in natural energy units, measured *relative
		to the band bottom* (E_ex(0, η=0) = 0).

		The absolute semiconductor offset ``E_gap_bare - E_bind`` is intentionally
		excluded; add it back in plotting code via
		``polaritons.plotting.semiconductor_offset_natural`` when an absolute
		energy axis is required.
		"""
		p  = self.p
		k  = np.asarray(k_nat, dtype=float)
		Ek = (p.hbar**2 * k**2) / (2.0 * p.M) - self.Q(k, eta)
		return Ek

	def E_ph(self, k_nat: float | np.ndarray, eta: float) -> np.ndarray:
		"""
		Photon dispersion in the band-bottom-relative convention,
		**tuned to the real part of the exciton energy at k=0**:

		    E_ph(k) = sqrt((hbar*c*k/n)^2 + E_0_abs^2) - E_band_bottom,

		where ``E_0_abs = E_band_bottom + Re[E_ex(0, eta)]`` and
		``E_band_bottom = E_gap_bare - E_bind`` is the semiconductor offset.

		Only the real part of the disorder-shifted exciton energy is used to
		tune the cavity, so the cavity photon dispersion is purely real and
		does not pick up the exciton's disorder-induced linewidth.
		"""
		p             = self.p
		k             = np.asarray(k_nat, dtype=float)
		E_band_bottom = float(p.E_gap_bare - p.E_bind)
		E_0_abs       = E_band_bottom + float(np.real(self.E_ex(0.0, eta)))
		return np.sqrt((p.hbar * p.c * k / p.n_refr)**2 + E_0_abs**2) - E_band_bottom

	def E_ph_untuned(self, k_nat: float | np.ndarray) -> np.ndarray:
		"""Photon dispersion tuned to the disorder-free (eta=0) exciton energy
		(band-bottom-relative; see E_ph for the convention).

		``E_ex(0, 0) = 0`` in the band-bottom-relative convention with no
		disorder, so this dispersion is purely real.
		"""
		p             = self.p
		k             = np.asarray(k_nat, dtype=float)
		E_band_bottom = float(p.E_gap_bare - p.E_bind)
		E_0_abs       = E_band_bottom + float(np.real(self.E_ex(0.0, 0.0)))
		return np.sqrt((p.hbar * p.c * k / p.n_refr)**2 + E_0_abs**2) - E_band_bottom

	def _branch_anchor_at_k0(
		self,
		eta_target    : float,
		disorder_tuned: bool,
		branch        : str,
		n_eta_steps   : int = 32,
	) -> complex:
		"""
		LP or UP eigenvalue at ``k=0`` for the requested ``eta_target``,
		obtained by **adiabatic continuation in eta** from the clean limit.

		Strategy
		--------
		At ``eta=0`` (Q=0) the 2x2 polariton Hamiltonian is Hermitian and
		the requested branch is unambiguous:

		    λ_LP(η=0, k=0) = E_ph(0,0)/2 − sqrt((E_ph(0,0)/2)² + (Ω/2)²)
		    λ_UP(η=0, k=0) = E_ph(0,0)/2 + sqrt((E_ph(0,0)/2)² + (Ω/2)²)

		(in the band-bottom-relative convention with ``E_ex(0,0)=0`` these
		reduce to ``∓Ω/2``).  For finite ``eta_target`` we walk along a
		dense eta path ``[0, eta_target]`` and at each step pick the
		eigenvalue closest to the previous-step value.  This gives the
		unique analytic continuation of the chosen branch across the
		exceptional point ``|Im[E_ex]| = Ω`` where naive "smaller/larger
		real part" picking fails.

		The walk is cheap (one 2x2 eig per step) and used only at k=0.
		"""
		if branch not in ("LP", "UP"):
			raise ValueError(f"branch must be 'LP' or 'UP', got {branch!r}.")
		sign = -1.0 if branch == "LP" else +1.0

		p = self.p
		g = 0.5 * p.Omega

		# Cavity at k=0 in the relevant tuning convention.
		Eph0_tuned = lambda η: float(np.real(self.E_ex(0.0, η))) if disorder_tuned else 0.0
		# (E_ph_untuned(0) == 0 in BBR since E_ex(0, 0) == 0.)

		# Exact clean branch eigenvalue at eta=0.
		Eph0_clean = Eph0_tuned(0.0)
		lam_prev   = 0.5 * Eph0_clean + sign * np.sqrt((0.5 * Eph0_clean)**2 + g**2 + 0j)

		if eta_target == 0.0:
			return complex(lam_prev)

		# Dense eta path; n_eta_steps + 1 points including endpoints.
		n_steps = max(int(n_eta_steps), 1)
		eta_path = np.linspace(0.0, float(eta_target), n_steps + 1)

		for η in eta_path[1:]:
			Eex0 = complex(self.E_ex(0.0, float(η)))
			Eph0 = complex(Eph0_tuned(float(η)))
			# Closed-form 2x2 eigenvalues.
			half_sum  = 0.5 * (Eex0 + Eph0)
			half_diff = 0.5 * (Eex0 - Eph0)
			disc      = np.sqrt(half_diff * half_diff + g * g)
			cand0     = half_sum + disc
			cand1     = half_sum - disc
			# Nearest-to-previous LP.
			lam_prev  = cand0 if abs(cand0 - lam_prev) <= abs(cand1 - lam_prev) else cand1

		return complex(lam_prev)

	def E_LP(
		self,
		k_nat         : np.ndarray,
		eta           : float,
		disorder_tuned: bool = True,
	) -> np.ndarray:
		"""
		Lower polariton dispersion from the 2×2 exciton-photon eigenvalue
		problem, with branches assigned by eigenvalue continuity.

		Algorithm
		---------
		1. Build the 2×2 Hamiltonian H(k) = [[E_ex, Ω/2], [Ω/2, E_ph]]
		   for every requested k (sorted ascending; k=0 prepended as an
		   anchor if not already in the input).
		2. At the anchor k=0 the LP is the eigenvalue obtained by
		   **adiabatic continuation in eta from the clean limit** — see
		   ``_branch_anchor_at_k0``.  This is correct even past the
		   exceptional point ``|Im[E_ex]| ≈ Ω`` where naive "smaller
		   real part" picking selects an unphysical branch.
		3. For each subsequent k the LP is the eigenvalue whose complex
		   distance to the previous-k LP eigenvalue is smaller.

		Parameters
		----------
		disorder_tuned : if True, cavity is tuned to Re[E_ex(0,eta)];
						if False, cavity is tuned to E_ex(0, 0) (disorder-free)
		"""
		return self._branch_dispersion(k_nat, eta, disorder_tuned, branch="LP")

	def E_UP(
		self,
		k_nat         : np.ndarray,
		eta           : float,
		disorder_tuned: bool = True,
	) -> np.ndarray:
		"""
		Upper polariton dispersion from the same 2×2 exciton-photon
		eigenvalue problem as ``E_LP``, but anchored to the upper branch
		at k=0 (clean-limit eigenvalue ``+Ω/2`` at resonance).  Branch
		tracking and the disorder-tuning convention are identical to
		``E_LP``.
		"""
		return self._branch_dispersion(k_nat, eta, disorder_tuned, branch="UP")

	def _branch_dispersion(
		self,
		k_nat         : np.ndarray,
		eta           : float,
		disorder_tuned: bool,
		branch        : str,
	) -> np.ndarray:
		k_in     = np.asarray(k_nat, dtype=float)
		in_shape = k_in.shape
		k_flat   = k_in.reshape(-1)

		# Sort ascending and prepend a k=0 anchor if not present.
		order        = np.argsort(k_flat, kind="stable")
		k_sorted     = k_flat[order]
		anchor_added = False
		if k_sorted.size == 0 or k_sorted[0] > 0.0:
			k_solve      = np.concatenate(([0.0], k_sorted))
			anchor_added = True
		else:
			k_solve = k_sorted

		Eex = np.asarray(self.E_ex(k_solve, eta), dtype=complex)
		Eph = np.asarray(
			self.E_ph(k_solve, eta) if disorder_tuned
			else self.E_ph_untuned(k_solve),
			dtype=complex,
		)
		# Params.Omega is the Rabi splitting (LP-UP gap at resonance), so the
		# off-diagonal coupling in the 2x2 exciton-photon block is Omega/2.
		g = 0.5 * self.p.Omega

		# Stacked 2x2 Hamiltonians, shape (N, 2, 2).
		N         = k_solve.size
		H         = np.empty((N, 2, 2), dtype=complex)
		H[:, 0, 0] = Eex
		H[:, 0, 1] = g
		H[:, 1, 0] = g
		H[:, 1, 1] = Eph
		evals, evecs = np.linalg.eig(H)  # evals (N, 2), evecs (N, 2, 2)

		# Anchor at k=0 by adiabatic continuation in eta from Q=0.  Pick by
		# eigenvalue distance to the analytic anchor (eigenvalues at k=0 are
		# well separated by ~Omega, so this is unambiguous).  Subsequent k
		# points use eigenvector-overlap tracking, which is robust against
		# large eigenvalue jumps between the two branches.
		lam_anchor = self._branch_anchor_at_k0(float(eta), disorder_tuned, branch)
		d0 = abs(evals[0, 0] - lam_anchor)
		d1 = abs(evals[0, 1] - lam_anchor)
		branch_idx0 = 0 if d0 <= d1 else 1

		branch_vals     = np.empty(N, dtype=complex)
		branch_vals[0]  = evals[0, branch_idx0]
		prev_vec        = evecs[0, :, branch_idx0]
		# Branch tracking via right-eigenvector overlap: pick the eigenpair
		# whose eigenvector has the largest |<v_prev | v_i>| with the
		# previous-k branch eigenvector.  This follows the same Riemann
		# sheet across both branch-eigenvalue crossings and large gaps,
		# whereas the simple "nearest eigenvalue" rule can switch branches
		# when one eigenvalue jumps faster than the other (e.g. UP at small
		# k, where the photon-like branch shoots up steeply).
		for i in range(1, N):
			v0 = evecs[i, :, 0]
			v1 = evecs[i, :, 1]
			o0 = abs(np.vdot(prev_vec, v0))
			o1 = abs(np.vdot(prev_vec, v1))
			pick = 0 if o0 >= o1 else 1
			branch_vals[i] = evals[i, pick]
			prev_vec       = evecs[i, :, pick]

		# Drop the prepended anchor (if any) and undo the sort.
		if anchor_added:
			branch_vals = branch_vals[1:]
		out_flat = np.empty_like(branch_vals)
		out_flat[order] = branch_vals
		return out_flat.reshape(in_shape)
