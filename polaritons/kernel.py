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
	Return a vectorised propagator function F(q, Q, eta).

	F(q, Q, eta) = -eta*q / (E_gap - E_bind - hbar^2*q^2/(2M) + Q + i*epsilon)

	`epsilon` is the imaginary regulator (in natural energy units) that
	keeps the propagator finite at the resonance.
	"""
	E_gap   = p.E_gap
	E_bind  = p.E_bind
	hbar    = p.hbar
	M       = p.M
	i_eps   = 1j * float(epsilon)

	def F(q, Q, eta=1.0):
		return -eta * q / (E_gap - E_bind - (hbar**2 * q**2) / (2.0 * M) + Q + i_eps)

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
