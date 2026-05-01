"""
Unit tests for polaritons.solver (picard_iteration).
"""
import numpy as np
import pytest

from polaritons.solver import picard_iteration


N = 10  # small grid size for fast tests


def _zero_F(q, Q, *, eta):
    """Propagator that returns zero — simplest convergence case."""
    return np.zeros_like(Q, dtype=complex)


def _identity_F(q, Q, *, eta):
    """Returns Q unchanged — K @ F_vec = K @ (Q * weights)."""
    return Q.astype(complex)


class TestPicardConvergence:
    def test_zero_kernel_converges_immediately(self):
        """With K=0 the update Q_new = 0 regardless of init."""
        q       = np.linspace(0.01, 5.0, N)
        Q_init  = np.ones(N, dtype=complex)
        K       = np.zeros((N, N))
        weights = np.ones(N) * 0.1

        Q, delta = picard_iteration(
            Q_init, q, K, weights, _identity_F,
            eta=1.0, tol=1e-12, max_iter=200, verbose=False,
        )
        np.testing.assert_allclose(Q, 0.0, atol=1e-12)

    def test_eta_zero_gives_zero_Q(self):
        """F ∝ eta, so with eta=0 the iteration maps Q → 0 in one step."""
        q       = np.linspace(0.01, 5.0, N)
        Q_init  = np.random.default_rng(0).normal(size=N).astype(complex)
        K       = np.eye(N)
        weights = np.ones(N)

        Q, delta = picard_iteration(
            Q_init, q, K, weights, _zero_F,
            eta=0.0, tol=1e-14, max_iter=10, verbose=False,
        )
        np.testing.assert_allclose(Q, 0.0, atol=1e-14)

    def test_converges_from_zero_init(self):
        """Starting from Q=0 with a zero kernel stays at zero."""
        q       = np.linspace(0.01, 5.0, N)
        Q_init  = np.zeros(N, dtype=complex)
        K       = np.zeros((N, N))
        weights = np.ones(N) * 0.1

        Q, delta = picard_iteration(
            Q_init, q, K, weights, _zero_F,
            eta=1.0, tol=1e-14, max_iter=50, verbose=False,
        )
        np.testing.assert_allclose(Q, 0.0, atol=1e-14)


class TestPicardOutputShape:
    def test_Q_shape(self):
        q       = np.linspace(0.01, 5.0, N)
        Q_init  = np.zeros(N, dtype=complex)
        K       = np.zeros((N, N))
        weights = np.ones(N)

        Q, delta = picard_iteration(
            Q_init, q, K, weights, _zero_F,
            eta=1.0, max_iter=20, verbose=False,
        )
        assert Q.shape == (N,)

    def test_delta_length(self):
        max_iter = 50
        q        = np.linspace(0.01, 5.0, N)
        Q_init   = np.zeros(N, dtype=complex)
        K        = np.zeros((N, N))
        weights  = np.ones(N)

        Q, delta = picard_iteration(
            Q_init, q, K, weights, _zero_F,
            eta=1.0, max_iter=max_iter, verbose=False,
        )
        assert delta.shape == (max_iter,)

    def test_does_not_modify_Q_init(self):
        q       = np.linspace(0.01, 5.0, N)
        Q_init  = np.ones(N, dtype=complex) * 3.0
        K       = np.zeros((N, N))
        weights = np.ones(N)

        Q_init_copy = Q_init.copy()
        picard_iteration(
            Q_init, q, K, weights, _zero_F,
            eta=1.0, max_iter=10, verbose=False,
        )
        np.testing.assert_array_equal(Q_init, Q_init_copy)


class TestPicardRelaxation:
    def test_under_relaxation_converges(self):
        """w < 1 under-relaxation should still converge to zero for K=0."""
        q       = np.linspace(0.01, 5.0, N)
        Q_init  = np.ones(N, dtype=complex)
        K       = np.zeros((N, N))
        weights = np.ones(N)

        Q, delta = picard_iteration(
            Q_init, q, K, weights, _identity_F,
            eta=1.0, tol=1e-10, max_iter=500, w=0.5, verbose=False,
        )
        np.testing.assert_allclose(Q, 0.0, atol=1e-9)

    def test_w_zero_leaves_Q_unchanged(self):
        """w=0 means no update — Q stays at Q_init forever."""
        q       = np.linspace(0.01, 5.0, N)
        Q_init  = np.ones(N, dtype=complex) * 7.0
        K       = np.eye(N)
        weights = np.ones(N)

        Q, _ = picard_iteration(
            Q_init, q, K, weights, _identity_F,
            eta=1.0, tol=1e-14, max_iter=100, w=0.0, verbose=False,
        )
        np.testing.assert_array_almost_equal(Q, Q_init)


class TestPicardDelta:
    def test_delta_zero_after_convergence(self):
        """Once converged (Q=0 from K=0), all subsequent deltas are zero."""
        q       = np.linspace(0.01, 5.0, N)
        Q_init  = np.zeros(N, dtype=complex)
        K       = np.zeros((N, N))
        weights = np.ones(N)
        max_iter = 30

        Q, delta = picard_iteration(
            Q_init, q, K, weights, _zero_F,
            eta=1.0, tol=1e-20, max_iter=max_iter, verbose=False,
        )
        np.testing.assert_allclose(delta, 0.0, atol=1e-20)
