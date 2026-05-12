"""
Unit tests for polaritons.kernel.
"""
import numpy as np
import pytest

from polaritons.parameters import Params
from polaritons.kernel import (
	make_propagator,
	make_kernel_gaussian,
	make_kernel_nongaussian,
	make_kernel,
)


@pytest.fixture(scope="module")
def p_nat():
	return Params().to_natural()


@pytest.fixture(scope="module")
def small_q():
	"""Coarse momentum grid in natural units."""
	return np.linspace(0.01, 30.0, 8)


class TestMakePropagator:
	def test_returns_callable(self, p_nat):
		F = make_propagator(p_nat)
		assert callable(F)

	def test_eta_zero_gives_zero(self, p_nat, small_q):
		"""F ∝ eta, so F(q, Q, eta=0) = 0."""
		F = make_propagator(p_nat)
		Q = np.zeros_like(small_q)
		result = F(small_q, Q, E_ext=0.0, eta=0.0)
		np.testing.assert_allclose(np.abs(result), 0.0, atol=1e-30)

	def test_scales_linearly_with_eta(self, p_nat, small_q):
		"""F(q, Q, 2*eta) = 2 * F(q, Q, eta)."""
		F = make_propagator(p_nat)
		Q = np.zeros_like(small_q)
		f1 = F(small_q, Q, E_ext=0.0, eta=1.0)
		f2 = F(small_q, Q, E_ext=0.0, eta=2.0)
		np.testing.assert_allclose(f2, 2.0 * f1, rtol=1e-10)

	def test_output_shape(self, p_nat, small_q):
		F = make_propagator(p_nat)
		Q = np.zeros_like(small_q)
		result = F(small_q, Q, E_ext=0.0, eta=1.0)
		assert result.shape == small_q.shape

	def test_complex_output(self, p_nat, small_q):
		"""F has a small imaginary regularizer, so result is complex."""
		F = make_propagator(p_nat)
		Q = np.zeros_like(small_q, dtype=complex)
		result = F(small_q, Q, E_ext=0.0, eta=1.0)
		assert np.iscomplexobj(result)

	def test_finite_at_resonance(self, p_nat, small_q):
		"""The i*epsilon regulator prevents divergence at the resonance."""
		F = make_propagator(p_nat)
		# Choose Q so that the real part of the denominator is exactly zero
		# at every q (E_ext = 0, so denom_re = -hbar^2 q^2/(2M) + Q.real).
		Q = (p_nat.hbar**2 * small_q**2) / (2.0 * p_nat.M) + 0j
		result = F(small_q, Q, E_ext=0.0, eta=1.0)
		assert np.all(np.isfinite(result))


class TestMakeKernelGaussian:
	def test_returns_callable(self, p_nat):
		K = make_kernel_gaussian(p_nat)
		assert callable(K)

	def test_K_matrix_shape(self, p_nat, small_q):
		K = make_kernel_gaussian(p_nat)
		mat = K(small_q, small_q)
		assert mat.shape == (len(small_q), len(small_q))

	def test_K_real(self, p_nat, small_q):
		K = make_kernel_gaussian(p_nat)
		mat = K(small_q, small_q)
		# The kernel is real-valued
		assert np.all(np.isreal(mat))

	def test_K_symmetric(self, p_nat, small_q):
		"""K(q, k) == K(k, q) — symmetry of the disorder kernel."""
		K = make_kernel_gaussian(p_nat)
		mat = K(small_q, small_q)
		np.testing.assert_allclose(mat, mat.T, rtol=1e-8)

	def test_single_element_consistent(self, p_nat, small_q):
		"""K called with single-element arrays matches the value in the full matrix."""
		K = make_kernel_gaussian(p_nat)
		mat = K(small_q, small_q)
		q_single = np.array([small_q[2]])
		k_single = np.array([small_q[4]])
		single = K(q_single, k_single)
		assert single.shape == (1, 1)
		assert float(single[0, 0]) == pytest.approx(mat[2, 4], rel=1e-10)

	def test_n_gauss_convergence(self, p_nat, small_q):
		"""Higher n_gauss should give similar results to default.

		n_gauss=32 vs n_gauss=96 typically agree to within ~5% on this grid;
		the key property is that the two kernels are proportional, not exact.
		"""
		K32 = make_kernel_gaussian(p_nat, n_gauss=32)
		K96 = make_kernel_gaussian(p_nat, n_gauss=96)
		mat32 = K32(small_q, small_q)
		mat96 = K96(small_q, small_q)
		np.testing.assert_allclose(mat32, mat96, rtol=0.05)

	def test_finite_values(self, p_nat, small_q):
		K = make_kernel_gaussian(p_nat)
		mat = K(small_q, small_q)
		assert np.all(np.isfinite(mat))

	def test_block_size_invariant(self, p_nat, small_q):
		"""block_size should not affect the result."""
		K = make_kernel_gaussian(p_nat)
		mat1 = K(small_q, small_q, block_size=2)
		mat8 = K(small_q, small_q, block_size=len(small_q))
		np.testing.assert_allclose(mat1, mat8, rtol=1e-12)


class TestMakeKernelNongaussian:
	def test_returns_callable(self, p_nat):
		K = make_kernel_nongaussian(p_nat)
		assert callable(K)

	def test_K_matrix_shape(self, p_nat, small_q):
		K = make_kernel_nongaussian(p_nat)
		mat = K(small_q, small_q)
		assert mat.shape == (len(small_q), len(small_q))

	def test_K_finite(self, p_nat, small_q):
		K = make_kernel_nongaussian(p_nat)
		mat = K(small_q, small_q)
		assert np.all(np.isfinite(mat))

	def test_K_symmetric(self, p_nat, small_q):
		K = make_kernel_nongaussian(p_nat)
		mat = K(small_q, small_q)
		np.testing.assert_allclose(mat, mat.T, rtol=1e-6)

	def test_block_size_invariant(self, p_nat, small_q):
		"""block_size should not affect the result."""
		K = make_kernel_nongaussian(p_nat)
		mat1 = K(small_q, small_q, block_size=2)
		mat8 = K(small_q, small_q, block_size=len(small_q))
		np.testing.assert_allclose(mat1, mat8, rtol=1e-12)


# ---------------------------------------------------------------------------
# make_kernel dispatcher
# ---------------------------------------------------------------------------

class TestMakeKernelDispatch:
	def test_gaussian_matches_factory(self, p_nat, small_q):
		K_a = make_kernel(p_nat, kind="gaussian")
		K_b = make_kernel_gaussian(p_nat)
		np.testing.assert_allclose(K_a(small_q, small_q), K_b(small_q, small_q), rtol=1e-12)

	def test_nongaussian_matches_factory(self, p_nat, small_q):
		K_a = make_kernel(p_nat, kind="nongaussian")
		K_b = make_kernel_nongaussian(p_nat)
		np.testing.assert_allclose(K_a(small_q, small_q), K_b(small_q, small_q), rtol=1e-12)

	def test_unknown_kind_raises(self, p_nat):
		with pytest.raises(ValueError, match="Unknown kernel kind"):
			make_kernel(p_nat, kind="bogus")


# ---------------------------------------------------------------------------
# Propagator epsilon kwarg
# ---------------------------------------------------------------------------

class TestPropagatorEpsilon:
	def test_default_matches_explicit_default(self, p_nat, small_q):
		F1 = make_propagator(p_nat)
		F2 = make_propagator(p_nat, epsilon=1e-9)
		Q  = np.zeros_like(small_q, dtype=complex)
		np.testing.assert_array_equal(
			F1(small_q, Q, E_ext=0.0, eta=1.0),
			F2(small_q, Q, E_ext=0.0, eta=1.0),
		)

	def test_zero_epsilon_changes_result(self, p_nat, small_q):
		# Imaginary part of F scales with epsilon when the denominator is real:
		# Im[F] = eta*q*epsilon / (Re_denom^2 + epsilon^2).  Compare imag parts.
		Q  = np.zeros_like(small_q, dtype=complex)
		F0 = make_propagator(p_nat, epsilon=1e-12)
		F1 = make_propagator(p_nat, epsilon=1e-3)
		im0 = np.abs(F0(small_q, Q, E_ext=0.0, eta=1.0).imag).max()
		im1 = np.abs(F1(small_q, Q, E_ext=0.0, eta=1.0).imag).max()
		assert im1 > 1e3 * im0


# ---------------------------------------------------------------------------
# Numerical correctness of make_kernel_nongaussian against direct quadrature
# ---------------------------------------------------------------------------

class TestNonGaussianAgainstDirectQuadrature:
	def test_single_qk_matches_dense_simpson(self, p_nat):
		"""
		For a single (q, k) pair, recompute the angular integrand on a dense
		uniform theta grid and integrate with Simpson's rule; require the
		Gauss-Legendre kernel to agree.
		"""
		from scipy.integrate import simpson

		# Pick a single off-diagonal pair away from any singularity.
		q, k = 0.7, 1.3
		K = make_kernel_nongaussian(p_nat, n_gauss=128)
		val = float(K(np.array([q]), np.array([k]))[0, 0])

		# Recompute analytically for cross-check using the same algebra.
		D_0, M, m_prime, m_rest = p_nat.D_0, p_nat.M, p_nat.m_prime, p_nat.m_rest
		m_e, m_h, a = p_nat.m_e, p_nat.m_h, p_nat.a
		prefactor = (4 * D_0 * M**6 * m_prime**2 * m_rest**2
					 / (np.pi * a**6 * m_e**2 * m_h**2))
		shift_e = 4 * M**2 / (a**2 * m_e**2)
		shift_h = 4 * M**2 / (a**2 * m_h**2)

		A_e = q**2 + k**2 + shift_e
		A_h = q**2 + k**2 + shift_h
		kq2 = q**2 * k**2
		t1 = (A_e**2 + 2*kq2) / (m_e**4 * (A_e**2 - 4*kq2)**2.5)
		t2 = (A_h**2 + 2*kq2) / (m_h**4 * (A_h**2 - 4*kq2)**2.5)

		theta = np.linspace(0.0, 2.0 * np.pi, 4001)
		base  = q**2 + k**2 - 2.0 * q * k * np.cos(theta)
		integrand = (-1.0 / np.pi) * m_e**2 * m_h**2 * (base + shift_e)**(-1.5) * (base + shift_h)**(-1.5)
		t3 = simpson(integrand, x=theta)
		expected = prefactor * (t1 + t2 + t3)

		assert val == pytest.approx(expected, rel=1e-4)
