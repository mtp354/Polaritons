"""
Dispersion relations for excitons, photons, and lower polaritons.

Functions here operate in SI units (k in m^-1, energies in eV).
They read complex Q(k, eta) from saved result files via the io module
and construct bivariate spline interpolants at load time.
"""

from __future__ import annotations
import numpy as np
from scipy.interpolate import RectBivariateSpline
from .parameters import Params


# ---------------------------------------------------------------------------
# Spline interpolants built from saved Q results
# ---------------------------------------------------------------------------

class DispersionModel:
	"""
	Encapsulates E_ex, E_ph, E_LP given a loaded set of Q(k, eta) results.

	Parameters
	----------
	p          : Params in SI units
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
		if p.in_natural_units:
			raise ValueError("DispersionModel expects Params in SI units, not natural units.")

		self.p        = p
		self.eta_grid = np.asarray(eta_grid, dtype=float)
		self.q_grid_nat = np.asarray(q_grid_nat, dtype=float)

		# Convert momentum grid to SI for external callers
		self.q_grid_si = self.q_grid_nat / p.a

		# Build bivariate splines  (axes: q_natural, eta)
		self._spline_real = RectBivariateSpline(
			self.q_grid_nat, self.eta_grid, np.real(Q_results).T
		)
		self._spline_imag = RectBivariateSpline(
			self.q_grid_nat, self.eta_grid, np.imag(Q_results).T
		)

	# ------------------------------------------------------------------
	# Self-energy
	# ------------------------------------------------------------------

	def Q(self, k_si: float | np.ndarray, eta: float) -> complex | np.ndarray:
		"""
		Self-energy Q(k, eta) interpolated from saved results.

		Parameters
		----------
		k_si : momentum in m^-1 (SI)
		eta  : disorder parameter

		Returns
		-------
		Q    : complex energy shift in eV (SI)
		"""
		p    = self.p
		k_nat = np.asarray(k_si, dtype=float) * p.a
		Q_nat = (
			self._spline_real(k_nat, eta)
			+ 1j * self._spline_imag(k_nat, eta)
		)
		# spline returns 2-D array; squeeze to scalar / 1-D
		Q_nat = Q_nat.squeeze()
		return p.E_bind * Q_nat

	# ------------------------------------------------------------------
	# Dispersion relations
	# ------------------------------------------------------------------

	def E_ex(self, k_si: float | np.ndarray, eta: float) -> np.ndarray:
		"""Complex exciton dispersion (eV) at given k (m^-1) and eta."""
		p  = self.p
		k  = np.asarray(k_si, dtype=float)
		Ek = p.E_gap - p.E_bind + (p.hbar**2 * k**2) / (2.0 * p.M) - self.Q(k, eta)
		return Ek

	def E_ph(self, k_si: float | np.ndarray, eta: float) -> np.ndarray:
		"""
		Photon dispersion  E_ph = sqrt((hbar*c*k/n)^2 + E_0^2)

		Cavity is tuned so E_ph(0) = Re[E_ex(0, eta)].
		"""
		p   = self.p
		k   = np.asarray(k_si, dtype=float)
		E_0 = float(np.real(self.E_ex(0.0, eta)))
		return np.sqrt((p.hbar * p.c * k / p.n_refr)**2 + E_0**2)

	def E_ph_untuned(self, k_si: float | np.ndarray) -> np.ndarray:
		"""Photon dispersion tuned to the disorder-free (eta=0) exciton energy."""
		p   = self.p
		k   = np.asarray(k_si, dtype=float)
		E_0 = float(np.real(self.E_ex(0.0, 0.0)))
		return np.sqrt((p.hbar * p.c * k / p.n_refr)**2 + E_0**2)

	def E_LP(
		self,
		k_si          : np.ndarray,
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
		k    = np.asarray(k_si, dtype=float)
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
