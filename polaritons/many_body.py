"""
Finite-temperature many-body quantities:
	- effective LP mass
	- chemical potential (Bose-Einstein fugacity series)
	- Hopfield coefficients
	- Matsubara bubble Pi0
	- bare and screened polariton interaction strength
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from .parameters import Params
from .dispersion import DispersionModel


# np.trapezoid was added in NumPy 2.0; np.trapz is deprecated there but still
# present.  Pick the modern name when available, falling back transparently.
_trapz = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]


def effective_mass(model: DispersionModel, eta: float, k_grid: np.ndarray,
				disorder_tuned: bool = True) -> float:
	"""
	LP effective mass at k=0 from a local even-polynomial fit.

		m* = hbar^2 / (d^2 E_LP / dk^2)|_{k=0}

	Parameters
	----------
	model          : DispersionModel (natural units)
	eta            : disorder parameter
	k_grid         : dense natural-unit k array including or starting near 0
	disorder_tuned : cavity tuning flag passed to E_LP
	"""
	k = np.asarray(k_grid, dtype=float)
	if k.ndim != 1:
		raise ValueError("k_grid must be a 1-D array.")

	k = np.unique(k[np.isfinite(k) & (k >= 0.0)])
	if len(k) < 3:
		raise ValueError("effective_mass requires at least three non-negative k points.")

	k.sort()
	n_fit = min(8, len(k))
	k_fit = k[:n_fit]
	E_fit = np.real(model.E_LP(k_fit, eta, disorder_tuned=disorder_tuned))

	x = k_fit**2
	x_scale = float(np.max(x))
	if x_scale <= 0.0:
		raise ValueError("effective_mass requires at least one positive k point.")

	degree = min(2, len(k_fit) - 1)
	coeffs = np.polyfit(x / x_scale, E_fit - E_fit[0], deg=degree)
	slope_at_zero = coeffs[-2] / x_scale
	d2E = 2.0 * slope_at_zero
	if not np.isfinite(d2E) or d2E <= 0.0:
		raise ValueError(f"Non-positive LP curvature at k=0: {d2E!r}")
	return float(model.p.hbar**2 / d2E)


def _fugacity_series_coefficients(
	dE_vals  : np.ndarray,
	ks       : np.ndarray,
	beta     : float,
	L_terms  : int,
	block_size: int = 200,
) -> np.ndarray:
	"""
	Bose-Einstein fugacity expansion coefficients a_l = ∫ k dk/(2pi) exp(-l*beta*dE).
	"""
	a = np.zeros(L_terms, dtype=float)
	idx = 0
	for l_start in range(1, L_terms + 1, block_size):
		l_stop  = min(l_start + block_size, L_terms + 1)
		l_block = np.arange(l_start, l_stop)[:, None]
		exp_block = ks[None, :] * np.exp(-l_block * beta * dE_vals[None, :])
		block_coeffs = _trapz(exp_block, ks, axis=1) / (2.0 * np.pi)
		a[idx:idx + len(block_coeffs)] = block_coeffs
		idx += len(block_coeffs)
	return a


def chemical_potential(
	model          : DispersionModel,
	eta            : float,
	beta           : float | None = None,
	L_terms        : int   = 100,
	k_upper        : float = 1.0,
	n_k            : int   = 100_000,
	disorder_tuned : bool  = True,
) -> float:
	"""
	Chemical potential mu at given disorder strength and temperature via
	bisection on the Bose-Einstein fugacity series.

	Parameters
	----------
	model          : DispersionModel (natural units)
	eta            : disorder parameter
	beta           : inverse temperature 1/(k_B*T); defaults to model.p.beta
	L_terms        : truncation order of the fugacity series
	k_upper        : upper natural momentum limit for density integral
	n_k            : number of k points in density integral
	disorder_tuned : cavity tuning flag

	Returns
	-------
	mu : chemical potential in natural energy units
	"""
	p    = model.p
	beta = beta if beta is not None else p.beta
	ks   = np.linspace(0.0, k_upper, n_k)
	E_0  = float(np.real(model.E_LP(np.array([0.0]), eta, disorder_tuned=disorder_tuned)[0]))
	dE   = np.maximum(
		np.real(model.E_LP(ks, eta, disorder_tuned=disorder_tuned)) - E_0, 0.0
	)

	a = _fugacity_series_coefficients(dE, ks, beta, L_terms)
	if a.sum() < p.concentration:
		raise ValueError(
			"Increase L_terms: truncated series cannot reach the target density."
		)

	powers = np.arange(1, L_terms + 1)
	z_hi   = 1.0 - 1e-14
	z      = brentq(
		lambda z: float(np.dot(a, z**powers) - p.concentration),
		0.0, z_hi,
		xtol=1e-14, rtol=1e-12, maxiter=200,
	)
	return np.log(z) / beta


def hopfield_coefficients(
	model     : DispersionModel,
	eta       : float,
	k_grid    : np.ndarray,
	E_lp_vals : np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Hopfield exciton (X_LP) and photon (C_LP) fractions.

	Returns
	-------
	X_LP : exciton fraction (complex array, same shape as k_grid)
	C_LP : photon fraction  (complex array)
	"""
	E_ex_vals = model.E_ex(k_grid, eta)
	ratio     = (E_lp_vals - E_ex_vals) / model.p.Omega
	X_LP = 1.0 / np.sqrt(1.0 + np.abs(ratio)**2)
	C_LP = ratio / np.sqrt(1.0 + np.abs(ratio)**2)
	return X_LP, C_LP


def Pi0(
	model          : DispersionModel,
	eta            : float,
	beta           : float | None = None,
	L_terms        : int   = 100,
	k_upper        : float = 1.0,
	n_k            : int   = 100_000,
	k_upper_mass   : float | None = None,
	n_k_mass       : int   = 64,
	disorder_tuned : bool  = True,
) -> float:
	"""
	Zero-frequency Matsubara bubble Π₀ at finite temperature T.

		Π₀ = -(m*_LP / pi*hbar^2) * 1/(exp(-mu*beta) - 1)

	`n_k` controls the chemical-potential density integral.  The effective
	mass is a local k=0 fit, so it uses a separate, smaller near-zero grid.
	"""
	p    = model.p
	beta = beta if beta is not None else p.beta
	mu   = chemical_potential(model, eta, beta=beta, L_terms=L_terms, k_upper=k_upper, n_k=n_k, disorder_tuned=disorder_tuned)
	k_upper_mass = min(k_upper, 1e-3) if k_upper_mass is None else k_upper_mass
	m_eff = effective_mass(
		model, eta,
		k_grid=np.linspace(0.0, k_upper_mass, n_k_mass),
		disorder_tuned=disorder_tuned,
	)
	return (-m_eff / (np.pi * p.hbar**2)) / (np.exp(-mu * beta) - 1.0)


def polariton_interaction_strength(
	model          : DispersionModel,
	eta            : float,
	k_grid         : np.ndarray,
	beta           : float | None = None,
	L_terms        : int   = 100,
	k_upper        : float = 1.0,
	n_k            : int   = 100_000,
	k_upper_mass   : float | None = None,
	n_k_mass       : int   = 64,
	disorder_tuned : bool  = True,
	bubble         : float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Bare and screened polariton-polariton interaction strengths in natural
	energy-area units.

	Bare:     g(k)  = (g_ex / N_qw) * |X_LP(k)|^4
	Screened: g'(k) = g(k) * (1 - g(k)*Π₀) / (1 - 2*g(k)*Π₀)

	Parameters
	----------
	model          : DispersionModel (natural units)
	eta            : disorder parameter
	k_grid         : natural-unit momentum array
	beta           : inverse temperature; defaults to model.p.beta
	n_k            : momentum grid size for chemical-potential integration
	n_k_mass       : local near-zero grid size for effective-mass fitting
	disorder_tuned : cavity tuning flag
	bubble         : optional precomputed Pi0 value for repeated calls

	Returns
	-------
	g_bare, g_screened : real arrays, shape matching k_grid
	"""
	p        = model.p
	beta     = beta if beta is not None else p.beta
	E_lp     = model.E_LP(k_grid, eta, disorder_tuned=disorder_tuned)
	X_LP, _  = hopfield_coefficients(model, eta, k_grid, E_lp)
	g_bare   = (p.g_ex / p.N_qw) * np.abs(X_LP)**4

	bubble_value = (
		Pi0(model, eta, beta=beta, L_terms=L_terms,
			k_upper=k_upper, n_k=n_k,
			k_upper_mass=k_upper_mass, n_k_mass=n_k_mass,
			disorder_tuned=disorder_tuned)
		if bubble is None
		else bubble
	)
	g_screened = g_bare * (1.0 - g_bare * bubble_value) / (1.0 - 2.0 * g_bare * bubble_value)
	return np.real(g_bare), np.real(g_screened)
