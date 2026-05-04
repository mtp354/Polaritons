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

		# Build one k-only spline for each solved eta value.
		self._Q_splines = [
			CubicSpline(self.q_grid_nat, Q_results[i], extrapolate=True)
			for i in range(len(self.eta_grid))
		]

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
		"""Complex exciton dispersion in natural energy units."""
		p  = self.p
		k  = np.asarray(k_nat, dtype=float)
		Ek = p.E_gap - p.E_bind + (p.hbar**2 * k**2) / (2.0 * p.M) - self.Q(k, eta)
		return Ek

	def E_ph(self, k_nat: float | np.ndarray, eta: float) -> np.ndarray:
		"""
		Photon dispersion  E_ph = sqrt((hbar*c*k/n)^2 + E_0^2)

		Cavity is tuned so E_ph(0) = Re[E_ex(0, eta)].
		"""
		p   = self.p
		k   = np.asarray(k_nat, dtype=float)
		E_0 = float(np.real(self.E_ex(0.0, eta)))
		return np.sqrt((p.hbar * p.c * k / p.n_refr)**2 + E_0**2)

	def E_ph_untuned(self, k_nat: float | np.ndarray) -> np.ndarray:
		"""Photon dispersion tuned to the disorder-free (eta=0) exciton energy."""
		p   = self.p
		k   = np.asarray(k_nat, dtype=float)
		E_0 = float(np.real(self.E_ex(0.0, 0.0)))
		return np.sqrt((p.hbar * p.c * k / p.n_refr)**2 + E_0**2)

	def E_LP(
		self,
		k_nat         : np.ndarray,
		eta           : float,
		disorder_tuned: bool = True,
	) -> np.ndarray:
		"""
		Lower polariton dispersion from the 2×2 exciton-photon eigenvalue problem.

		Parameters
		----------
		disorder_tuned : if True, cavity is tuned to Re[E_ex(0,eta)];
						if False, cavity is tuned to E_ex(0, 0) (disorder-free)
		"""
		k    = np.asarray(k_nat, dtype=float)
		Eex  = np.asarray(self.E_ex(k, eta),    dtype=complex)
		Eph  = (
			np.asarray(self.E_ph(k, eta),        dtype=complex)
			if disorder_tuned
			else np.asarray(self.E_ph_untuned(k), dtype=complex)
		)
		Omega = self.p.Omega

		if k.ndim == 0:
			vals = np.linalg.eigvals(
				np.array([[Eex, Omega], [Omega, Eph]], dtype=complex)
			)
			return vals[np.argmin(vals.real)]

		out = np.empty_like(Eex, dtype=complex)
		for i in range(len(k)):
			vals = np.linalg.eigvals(
				np.array([[Eex[i], Omega], [Omega, Eph[i]]], dtype=complex)
			)
			if i == 0:
				out[i] = vals[np.argmin(vals.real)]
			else:
				out[i] = vals[np.argmin(np.abs(vals - out[i-1]))]
		return out
