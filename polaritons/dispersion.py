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

	def _eta_index(self, eta: float) -> int:
		"""Index of an already-solved eta value."""
		matches = np.flatnonzero(np.isclose(self.eta_grid, eta, rtol=1e-12, atol=1e-12))
		if len(matches) != 1:
			raise ValueError(
				f"eta={eta!r} is not in eta_grid; Q is only interpolated over k."
			)
		return int(matches[0])

	# ------------------------------------------------------------------
	# Self-energy
	# ------------------------------------------------------------------

	def Q(self, k_nat: float | np.ndarray, eta: float) -> complex | np.ndarray:
		"""
		Self-energy Q(k, eta) interpolated over k for a solved eta value.

		Parameters
		----------
		k_nat : momentum in natural units
		eta   : solved disorder parameter value

		Returns
		-------
		Q     : complex energy shift in natural energy units
		"""
		k_nat = np.asarray(k_nat, dtype=float)
		if eta == 0.0:
			return np.zeros_like(k_nat, dtype=complex)
		return self._Q_splines[self._eta_index(eta)](k_nat)

	# ------------------------------------------------------------------
	# Dispersion relations
	# ------------------------------------------------------------------

	def E_ex(self, k_nat: float | np.ndarray, eta: float) -> np.ndarray:
		"""
		Complex exciton dispersion in natural energy units, measured *relative
		to the band bottom* (E_ex(0, η=0) = 0).

		The absolute semiconductor offset ``E_gap - E_bind`` is intentionally
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
		``E_band_bottom = E_gap - E_bind`` is the semiconductor offset.

		Only the real part of the disorder-shifted exciton energy is used to
		tune the cavity, so the cavity photon dispersion is purely real and
		does not pick up the exciton's disorder-induced linewidth.
		"""
		p             = self.p
		k             = np.asarray(k_nat, dtype=float)
		E_band_bottom = float(p.E_gap - p.E_bind)
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
		E_band_bottom = float(p.E_gap - p.E_bind)
		E_0_abs       = E_band_bottom + float(np.real(self.E_ex(0.0, 0.0)))
		return np.sqrt((p.hbar * p.c * k / p.n_refr)**2 + E_0_abs**2) - E_band_bottom

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
		2. At the anchor k=0 the LP is the eigenvalue with the smaller real
		   part; on a real-part tie (over-damped regime) the more-negative
		   imaginary part wins.
		3. For each subsequent k the LP is the eigenvalue whose complex
		   distance to the previous-k LP eigenvalue is smaller.

		Parameters
		----------
		disorder_tuned : if True, cavity is tuned to Re[E_ex(0,eta)];
						if False, cavity is tuned to E_ex(0, 0) (disorder-free)
		"""
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
		evals, _  = np.linalg.eig(H)  # shape (N, 2)

		# Anchor at k=0: smaller real part, more-negative imag on tie.
		scale     = max(abs(g), 1.0)
		real_diff = (evals[0, 0].real - evals[0, 1].real) / scale
		if abs(real_diff) > 1e-10:
			lp_idx0 = 0 if real_diff < 0 else 1
		else:
			lp_idx0 = 0 if evals[0, 0].imag <= evals[0, 1].imag else 1

		lp_vals     = np.empty(N, dtype=complex)
		lp_vals[0]  = evals[0, lp_idx0]
		prev        = lp_vals[0]
		# Branch tracking: pick the eigenvalue closest (in the complex
		# plane) to the previous LP eigenvalue.
		for i in range(1, N):
			d0 = abs(evals[i, 0] - prev)
			d1 = abs(evals[i, 1] - prev)
			lp_vals[i] = evals[i, 0] if d0 <= d1 else evals[i, 1]
			prev       = lp_vals[i]

		# Drop the prepended anchor (if any) and undo the sort.
		if anchor_added:
			lp_vals = lp_vals[1:]
		out_flat = np.empty_like(lp_vals)
		out_flat[order] = lp_vals
		return out_flat.reshape(in_shape)
