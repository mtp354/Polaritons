"""
Unit tests for polaritons.kernel.
"""
import numpy as np
import pytest

from polaritons.parameters import Params
from polaritons.kernel import make_propagator, make_kernel_gaussian, make_kernel_nongaussian


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
        result = F(small_q, Q, eta=0.0)
        np.testing.assert_allclose(np.abs(result), 0.0, atol=1e-30)

    def test_scales_linearly_with_eta(self, p_nat, small_q):
        """F(q, Q, 2*eta) = 2 * F(q, Q, eta)."""
        F = make_propagator(p_nat)
        Q = np.zeros_like(small_q)
        f1 = F(small_q, Q, eta=1.0)
        f2 = F(small_q, Q, eta=2.0)
        np.testing.assert_allclose(f2, 2.0 * f1, rtol=1e-10)

    def test_output_shape(self, p_nat, small_q):
        F = make_propagator(p_nat)
        Q = np.zeros_like(small_q)
        result = F(small_q, Q, eta=1.0)
        assert result.shape == small_q.shape

    def test_complex_output(self, p_nat, small_q):
        """F has a small imaginary regularizer, so result is complex."""
        F = make_propagator(p_nat)
        Q = np.zeros_like(small_q, dtype=complex)
        result = F(small_q, Q, eta=1.0)
        assert np.iscomplexobj(result)

    def test_finite_at_resonance(self, p_nat, small_q):
        """The i*epsilon regulator prevents divergence at the resonance."""
        F = make_propagator(p_nat)
        # Q chosen so denominator is nearly zero for mid-grid q
        Q = np.ones_like(small_q) * p_nat.E_gap
        result = F(small_q, Q, eta=1.0)
        assert np.all(np.isfinite(result))


class TestMakeKernelGaussian:
    def test_returns_two_callables(self, p_nat):
        K, K_vec = make_kernel_gaussian(p_nat)
        assert callable(K)
        assert callable(K_vec)

    def test_K_matrix_shape(self, p_nat, small_q):
        K, _ = make_kernel_gaussian(p_nat)
        mat = K(small_q, small_q)
        assert mat.shape == (len(small_q), len(small_q))

    def test_K_real(self, p_nat, small_q):
        K, _ = make_kernel_gaussian(p_nat)
        mat = K(small_q, small_q)
        # The kernel is real-valued
        assert np.all(np.isreal(mat))

    def test_K_symmetric(self, p_nat, small_q):
        """K(q, k) == K(k, q) — symmetry of the disorder kernel."""
        K, _ = make_kernel_gaussian(p_nat)
        mat = K(small_q, small_q)
        np.testing.assert_allclose(mat, mat.T, rtol=1e-8)

    def test_K_vec_consistent_with_K(self, p_nat, small_q):
        """K_vec(q, k) should match the diagonal of K(q, k) evaluated element-wise."""
        K, K_vec = make_kernel_gaussian(p_nat)
        # K_vec(q[i], k[j]) for a single pair
        q_single = np.array([small_q[2]])
        k_single = np.array([small_q[4]])
        mat_val  = K(q_single, k_single)[0, 0]
        vec_val  = K_vec(float(small_q[2]), float(small_q[4]))
        assert float(mat_val) == pytest.approx(float(vec_val), rel=1e-6)

    def test_n_gauss_convergence(self, p_nat, small_q):
        """Higher n_gauss should give similar results to default."""
        K32, _  = make_kernel_gaussian(p_nat, n_gauss=32)
        K96, _  = make_kernel_gaussian(p_nat, n_gauss=96)
        mat32 = K32(small_q, small_q)
        mat96 = K96(small_q, small_q)
        np.testing.assert_allclose(mat32, mat96, rtol=1e-3)

    def test_finite_values(self, p_nat, small_q):
        K, _ = make_kernel_gaussian(p_nat)
        mat = K(small_q, small_q)
        assert np.all(np.isfinite(mat))


class TestMakeKernelNongaussian:
    def test_returns_two_callables(self, p_nat):
        K, K_vec = make_kernel_nongaussian(p_nat)
        assert callable(K)
        assert callable(K_vec)

    def test_K_matrix_shape(self, p_nat, small_q):
        K, _ = make_kernel_nongaussian(p_nat)
        mat = K(small_q, small_q)
        assert mat.shape == (len(small_q), len(small_q))

    def test_K_finite(self, p_nat, small_q):
        K, _ = make_kernel_nongaussian(p_nat)
        mat = K(small_q, small_q)
        assert np.all(np.isfinite(mat))

    def test_K_symmetric(self, p_nat, small_q):
        K, _ = make_kernel_nongaussian(p_nat)
        mat = K(small_q, small_q)
        np.testing.assert_allclose(mat, mat.T, rtol=1e-6)
