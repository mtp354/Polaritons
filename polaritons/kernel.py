"""
Disorder scattering kernel K(q, k) and propagator F(q, Q, eta).

All inputs/outputs are in natural units.  Pass a Params object that has
already been converted via Params.to_natural().

Both kernel factories return a single callable::

	K = make_kernel_gaussian(p)
	matrix = K(q_array, k_array)        # shape (N_q, N_k)

Angular quadrature is performed with Gauss-Legendre nodes on [0, 2π] and
the computation is blocked along the q-axis so peak memory is bounded.
"""

from __future__ import annotations
import numpy as np
from .parameters import Params


def make_propagator(p: Params, epsilon: float = 1e-9):
	"""
	Return a vectorised propagator function ``F(q, Q, E_ext, eta)``.

	    F(q, Q, E_ext, eta) = -eta * q
	                          / (E_ext - hbar^2 q^2 / (2 M) + Q + i*epsilon)

	``E_ext`` is the external energy *measured relative to the exciton band
	bottom* ``E_gap - E_bind`` (the constant offset is no longer baked into
	the denominator). ``epsilon`` is the imaginary regulator (in natural
	energy units) that keeps the propagator finite at the resonance.
	"""
	hbar    = p.hbar
	M       = p.M
	i_eps   = 1j * float(epsilon)

	def F(q, Q, E_ext, eta=1.0):
		return -eta * q / (E_ext - (hbar**2 * q**2) / (2.0 * M) + Q + i_eps)

	return F


def _gauss_legendre_theta(n_gauss: int) -> tuple[np.ndarray, np.ndarray]:
	"""Gauss-Legendre nodes mapped from [-1,1] -> [0, 2pi]."""
	x, w = np.polynomial.legendre.leggauss(n_gauss)
	theta_w   = np.pi * w
	cos_theta = np.cos(np.pi * (x + 1.0))
	return cos_theta, theta_w


def _kernel_shifts(p: Params) -> tuple[float, float]:
	"""Shifts (shift_e, shift_h) common to both Gaussian and non-Gaussian kernels."""
	M, a, m_e, m_h = p.M, p.a, p.m_e, p.m_h
	shift_e = 4 * M**2 / (a**2 * m_e**2)
	shift_h = 4 * M**2 / (a**2 * m_h**2)
	return shift_e, shift_h


def _block_iter_q(N_q: int, block_size: int):
	for i0 in range(0, N_q, block_size):
		yield i0, min(i0 + block_size, N_q)


def make_kernel_gaussian(p: Params, n_gauss: int = 96):
	"""
	Return a vectorised kernel function K(q, k) for Gaussian-correlated disorder.

	K(q, k)[i, j] = prefactor * ∫₀²π dθ  exp(−ξ²/2 p²) × [bracket(p²)]²

	where p² = q[i]² + k[j]² − 2 q[i] k[j] cosθ  and
	bracket(p²) = (p²+shift_h)^{−3/2}/m_h² − (p²+shift_e)^{−3/2}/m_e²

	Angular integration uses ``n_gauss``-point Gauss-Legendre quadrature.
	The q-axis is processed in blocks of ``block_size`` to bound peak memory.

	Parameters
	----------
	p       : Params in natural units
	n_gauss : number of Gauss-Legendre nodes for the θ integration

	Returns
	-------
	K : callable  K(q, k, block_size=64) → float64 array of shape (len(q), len(k))
	"""
	D_0, M, m_prime, m_rest = p.D_0, p.M, p.m_prime, p.m_rest
	m_e, m_h, a, xi         = p.m_e, p.m_h, p.a, p.xi

	prefactor = (
		2 * D_0 * M**6 * m_prime**2 * m_rest**2
		/ (np.pi**2 * a**6 * m_e**2 * m_h**2)
	)
	shift_e, shift_h = _kernel_shifts(p)
	cos_theta, theta_w = _gauss_legendre_theta(n_gauss)

	def K(q: np.ndarray, k: np.ndarray, block_size: int = 64) -> np.ndarray:
		q = np.asarray(q, dtype=float)
		k = np.asarray(k, dtype=float)
		N_q, N_k = len(q), len(k)
		out = np.empty((N_q, N_k), dtype=float)

		for i0, i1 in _block_iter_q(N_q, block_size):
			q_b = q[i0:i1]
			# p²[b, j, l] = q_b[b]² + k[j]² − 2 q_b[b] k[j] cosθ[l]
			p2 = (q_b[:, None, None]**2 + k[None, :, None]**2
				  - 2.0 * q_b[:, None, None] * k[None, :, None] * cos_theta[None, None, :])
			gauss   = np.exp(-0.5 * xi**2 * p2)
			bracket = ((p2 + shift_h)**(-1.5) / m_h**2 - (p2 + shift_e)**(-1.5) / m_e**2)
			out[i0:i1] = prefactor * (gauss * bracket**2 @ theta_w)

		return out

	return K


def make_kernel_nongaussian(p: Params, n_gauss: int = 96):
	"""
	Return a vectorised kernel function K(q, k) for white-noise disorder.

	The kernel has three terms:
	t1, t2 — analytic closed-form outer products (no integration)
	t3     — angular integral evaluated with ``n_gauss``-point GL quadrature

	Parameters
	----------
	p       : Params in natural units
	n_gauss : number of Gauss-Legendre nodes for the θ integration in t3

	Returns
	-------
	K : callable  K(q, k, block_size=64) → float64 array of shape (len(q), len(k))
	"""
	D_0, M, m_prime, m_rest = p.D_0, p.M, p.m_prime, p.m_rest
	m_e, m_h, a             = p.m_e, p.m_h, p.a

	prefactor = (
		4 * D_0 * M**6 * m_prime**2 * m_rest**2
		/ (np.pi * a**6 * m_e**2 * m_h**2)
	)
	shift_e, shift_h = _kernel_shifts(p)
	cos_theta, theta_w = _gauss_legendre_theta(n_gauss)

	def K(q: np.ndarray, k: np.ndarray, block_size: int = 64) -> np.ndarray:
		q = np.asarray(q, dtype=float)
		k = np.asarray(k, dtype=float)
		N_q = len(q)

		# -- Analytic t1, t2 (no angular integration) -----------------------
		q2  = q[:, None]
		k2  = k[None, :]
		A_e = q2**2 + k2**2 + shift_e
		A_h = q2**2 + k2**2 + shift_h
		kq2 = q2**2 * k2**2

		t1 = (A_e**2 + 2*kq2) / (m_e**4 * (A_e**2 - 4*kq2)**2.5)
		t2 = (A_h**2 + 2*kq2) / (m_h**4 * (A_h**2 - 4*kq2)**2.5)

		# -- t3 cross-term: blocked angular integration ----------------------
		out = np.empty_like(t1)

		for i0, i1 in _block_iter_q(N_q, block_size):
			q_b = q[i0:i1]
			base = (q_b[:, None, None]**2
					+ k[None, :, None]**2
					- 2.0 * q_b[:, None, None] * k[None, :, None] * cos_theta[None, None, :])

			p2_e = base + shift_e
			p2_h = base + shift_h

			integrand = (-1.0 / np.pi) * m_e**2 * m_h**2 * p2_e**(-1.5) * p2_h**(-1.5)
			t3 = integrand @ theta_w

			out[i0:i1] = prefactor * (t1[i0:i1] + t2[i0:i1] + t3)

		return out

	return K


_KERNEL_FACTORIES = {
	"gaussian":    make_kernel_gaussian,
	"nongaussian": make_kernel_nongaussian,
}


def make_kernel(p: Params, kind: str = "gaussian", n_gauss: int = 96):
	"""Dispatch to make_kernel_gaussian / make_kernel_nongaussian by name."""
	try:
		factory = _KERNEL_FACTORIES[kind]
	except KeyError:
		raise ValueError(
			f"Unknown kernel kind: {kind!r}. Expected one of {list(_KERNEL_FACTORIES)}."
		)
	return factory(p, n_gauss=n_gauss)


def _kernel_diagonal(K_fn, t_probe: np.ndarray) -> np.ndarray:
	"""Evaluate K(t, t) along ``t_probe`` without forming the full mesh."""
	t_probe = np.asarray(t_probe, dtype=float)
	out = np.empty(t_probe.shape, dtype=float)
	# Most kernel factories accept paired arrays of shape (N, M); evaluate one
	# probe point at a time to keep memory bounded and to avoid relying on
	# diagonal-extraction conventions.
	for i, t in enumerate(t_probe):
		out[i] = float(K_fn(np.array([t]), np.array([t]))[0, 0])
	return out


def find_kernel_truncation(
	K_fn,
	*,
	threshold: float,
	k_lower: float,
	n_probe: int = 4096,
	k_start: float | None = None,
	max_expansions: int = 12,
) -> float:
	"""
	Find the truncation momentum ``t`` for a kernel ``K_fn``.

	``t`` is the smallest k > 0 satisfying
	``|K(t, t)| <= threshold * |K(0, 0)|`` (i.e. the kernel diagonal falls
	below a fraction of its zero-momentum value), clamped from below by
	``k_lower``. The search bound starts at ``max(k_start, 2 * k_lower)``
	and doubles up to ``max_expansions`` times until the threshold is
	crossed.

	Parameters
	----------
	K_fn       : callable ``K(q, k) -> array`` (vectorised in q and k).
	threshold  : relative threshold (e.g. 1e-3).
	k_lower    : lower bound on the returned truncation (natural units).
	n_probe    : number of probe points on the linear grid [0, k_search_max].
	k_start    : initial upper bound for the search (natural units). Defaults
	             to ``20 * k_lower``.
	max_expansions : number of times to double ``k_search_max`` if the
	                 threshold has not been crossed.

	Returns
	-------
	t : float, the truncation momentum in natural units.
	"""
	if threshold <= 0.0 or threshold >= 1.0:
		raise ValueError(f"threshold must be in (0, 1); got {threshold}")
	if k_lower <= 0.0:
		raise ValueError(f"k_lower must be positive; got {k_lower}")

	K00 = float(K_fn(np.array([0.0]), np.array([0.0]))[0, 0])
	if not np.isfinite(K00) or K00 == 0.0:
		raise ValueError(f"K(0, 0) must be finite and non-zero; got {K00}")
	target = threshold * abs(K00)

	k_search_max = float(k_start) if k_start is not None else 20.0 * k_lower
	k_search_max = max(k_search_max, 2.0 * k_lower)

	for _ in range(max_expansions + 1):
		t_probe = np.linspace(0.0, k_search_max, int(n_probe))
		K_diag  = np.abs(_kernel_diagonal(K_fn, t_probe))
		below   = np.flatnonzero(K_diag <= target)
		if below.size:
			j = int(below[0])
			if j == 0:
				t = float(t_probe[1])
			else:
				k0, k1 = t_probe[j - 1], t_probe[j]
				K0, K1 = K_diag[j - 1], K_diag[j]
				if K0 == K1:
					t = float(k1)
				else:
					t = float(k0 + (target - K0) * (k1 - k0) / (K1 - K0))
			return max(t, float(k_lower))
		k_search_max *= 2.0

	raise RuntimeError(
		f"|K(t,t)| did not fall below {threshold}*|K(0,0)| within t <= {k_search_max:g}; "
		"increase max_expansions or k_start."
	)
